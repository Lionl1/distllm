//! distllm relay in Rust: a drop-in replacement for `distllm relay` (Python).
//!
//! Same HTTP protocol, same msgpack/JSON bodies and status codes, so the Python
//! workers, `Cluster` and the examples work unchanged. See docs/architecture.md.

// Handlers return a ready HTTP response as the error of their parameter helpers; it is
// only built on the error path, so its size does not matter.
#![allow(clippy::result_large_err)]

mod state;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use serde_json::json;

use state::{Relay, ResultView, Settings};

#[derive(Parser, Debug)]
#[command(name = "distllm-relay", about = "distllm relay server (Rust)")]
struct Args {
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Interface to bind.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    /// Bearer token required from clients (and workers, unless --worker-token).
    #[arg(long, env = "DISTLLM_TOKEN")]
    token: Option<String>,
    /// Separate bearer token for workers.
    #[arg(long, env = "DISTLLM_WORKER_TOKEN")]
    worker_token: Option<String>,
    /// Add artificial round-trip latency to worker requests (experiments only).
    #[arg(long, default_value_t = 0.0)]
    simulate_worker_rtt_ms: f64,
    /// Add artificial round-trip latency to client requests (experiments only).
    #[arg(long, default_value_t = 0.0)]
    simulate_client_rtt_ms: f64,
    /// Seconds before an assigned task without a result is re-queued.
    #[arg(long, default_value_t = 30.0)]
    task_timeout: f64,
    /// Seconds without a heartbeat before a worker is considered dead.
    #[arg(long, default_value_t = 30.0)]
    node_timeout: f64,
}

type Shared = Arc<Relay>;
type Params = Query<HashMap<String, String>>;

const WORKER_PATHS: [&str; 6] = [
    "/register",
    "/deregister",
    "/heartbeat",
    "/poll",
    "/result",
    "/fail",
];

fn is_worker_path(path: &str) -> bool {
    WORKER_PATHS.iter().any(|p| path.starts_with(p))
}

// ---------------------------------------------------------------------------
// Responses shaped like FastAPI's
// ---------------------------------------------------------------------------

fn json_response(status: StatusCode, value: serde_json::Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        value.to_string(),
    )
        .into_response()
}

fn ok(status: &str) -> Response {
    json_response(StatusCode::OK, json!({ "status": status }))
}

fn not_found() -> Response {
    json_response(StatusCode::NOT_FOUND, json!({ "detail": "Not Found" }))
}

fn octet(bytes: Vec<u8>) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        bytes,
    )
        .into_response()
}

/// A missing or malformed query parameter, reported like FastAPI's validation error.
fn invalid(name: &str, problem: &str) -> Response {
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        json!({ "detail": [{ "loc": ["query", name], "msg": problem, "type": "value_error" }] }),
    )
}

fn required<'a>(params: &'a HashMap<String, String>, name: &str) -> Result<&'a str, Response> {
    params
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| invalid(name, "Field required"))
}

fn parse_bool(params: &HashMap<String, String>, name: &str) -> Result<bool, Response> {
    match params.get(name).map(|v| v.to_ascii_lowercase()) {
        None => Ok(false),
        Some(v) if ["true", "1", "yes", "on", "t", "y"].contains(&v.as_str()) => Ok(true),
        Some(v) if ["false", "0", "no", "off", "f", "n"].contains(&v.as_str()) => Ok(false),
        Some(_) => Err(invalid(name, "Input should be a valid boolean")),
    }
}

fn parse_num<T: std::str::FromStr>(
    params: &HashMap<String, String>,
    name: &str,
    default: T,
) -> Result<T, Response> {
    match params.get(name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| invalid(name, "Input should be a valid number")),
    }
}

// ---------------------------------------------------------------------------
// Middleware: authentication, latency simulation
// ---------------------------------------------------------------------------

async fn require_token(State(relay): State<Shared>, request: Request, next: Next) -> Response {
    let worker = is_worker_path(request.uri().path());
    let expected = if worker {
        &relay.settings.worker_token
    } else {
        &relay.settings.client_token
    };
    if let Some(expected) = expected {
        let header_value = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let (scheme, value) = header_value.split_once(' ').unwrap_or((header_value, ""));
        if !scheme.eq_ignore_ascii_case("bearer")
            || !constant_time_eq(value.trim().as_bytes(), expected.as_bytes())
        {
            return json_response(
                StatusCode::UNAUTHORIZED,
                json!({ "detail": "Invalid or missing token" }),
            );
        }
    }
    next.run(request).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn simulate_latency(State(relay): State<Shared>, request: Request, next: Next) -> Response {
    let rtt = if is_worker_path(request.uri().path()) {
        relay.settings.worker_rtt
    } else {
        relay.settings.client_rtt
    };
    if rtt.is_zero() {
        return next.run(request).await;
    }
    tokio::time::sleep(rtt / 2).await;
    let response = next.run(request).await;
    tokio::time::sleep(rtt / 2).await;
    response
}

// ---------------------------------------------------------------------------
// Worker endpoints
// ---------------------------------------------------------------------------

async fn register(State(relay): State<Shared>, Query(p): Params) -> Response {
    let (node_id, role) = match (required(&p, "node_id"), required(&p, "role")) {
        (Ok(n), Ok(r)) => (n, r),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    relay.register(node_id, role);
    ok("ok")
}

async fn deregister(State(relay): State<Shared>, Query(p): Params) -> Response {
    let node_id = match required(&p, "node_id") {
        Ok(n) => n,
        Err(e) => return e,
    };
    relay.deregister(node_id);
    ok("ok")
}

async fn heartbeat(State(relay): State<Shared>, Query(p): Params) -> Response {
    let node_id = match required(&p, "node_id") {
        Ok(n) => n,
        Err(e) => return e,
    };
    if relay.heartbeat(node_id) {
        ok("ok")
    } else {
        not_found()
    }
}

async fn poll(State(relay): State<Shared>, Path(role): Path<String>, Query(p): Params) -> Response {
    let worker_id = match required(&p, "worker_id") {
        Ok(w) => w.to_string(),
        Err(e) => return e,
    };
    let max_tasks: i64 = match parse_num(&p, "max_tasks", 1) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let taken = relay
        .poll(
            &role,
            &worker_id,
            max_tasks.max(1) as usize,
            Duration::from_secs(5),
        )
        .await;
    if taken.is_empty() {
        return StatusCode::NO_CONTENT.into_response();
    }
    // {"task_id", "payload"} for one task (the classic format), {"tasks": [...]} for batches.
    let mut out = Vec::with_capacity(taken.iter().map(|(_, b)| b.len() + 64).sum());
    if max_tasks <= 1 {
        write_task(&mut out, &taken[0].0, &taken[0].1);
    } else {
        rmp::encode::write_map_len(&mut out, 1).unwrap();
        rmp::encode::write_str(&mut out, "tasks").unwrap();
        rmp::encode::write_array_len(&mut out, taken.len() as u32).unwrap();
        for (task_id, payload) in &taken {
            write_task(&mut out, task_id, payload);
        }
    }
    octet(out)
}

fn write_task(out: &mut Vec<u8>, task_id: &str, payload: &[u8]) {
    rmp::encode::write_map_len(out, 2).unwrap();
    rmp::encode::write_str(out, "task_id").unwrap();
    rmp::encode::write_str(out, task_id).unwrap();
    rmp::encode::write_str(out, "payload").unwrap();
    rmp::encode::write_bin(out, payload).unwrap();
}

async fn post_result(
    State(relay): State<Shared>,
    Path(task_id): Path<String>,
    Query(p): Params,
    body: Bytes,
) -> Response {
    let stop_chain = match parse_bool(&p, "stop_chain") {
        Ok(v) => v,
        Err(e) => return e,
    };
    match relay.post_result(
        &task_id,
        p.get("worker_id").map(String::as_str),
        stop_chain,
        body.to_vec(),
    ) {
        state::Outcome::NotFound => not_found(),
        state::Outcome::WrongWorker => conflict(),
        state::Outcome::Status(s) => ok(s),
    }
}

async fn post_failure(
    State(relay): State<Shared>,
    Path(task_id): Path<String>,
    Query(p): Params,
    body: Bytes,
) -> Response {
    let text = String::from_utf8_lossy(&body).to_string();
    let message = if !text.is_empty() {
        text
    } else {
        p.get("error_msg")
            .cloned()
            .unwrap_or_else(|| "unknown error".to_string())
    };
    match relay.post_failure(&task_id, p.get("worker_id").map(String::as_str), message) {
        state::Outcome::NotFound => not_found(),
        state::Outcome::WrongWorker => conflict(),
        state::Outcome::Status(s) => ok(s),
    }
}

fn conflict() -> Response {
    json_response(
        StatusCode::CONFLICT,
        json!({ "detail": "task is assigned to another worker" }),
    )
}

// ---------------------------------------------------------------------------
// Client endpoints
// ---------------------------------------------------------------------------

async fn submit(State(relay): State<Shared>, Query(p): Params, body: Bytes) -> Response {
    let (target_role, _sender_id) = match (required(&p, "target_role"), required(&p, "sender_id")) {
        (Ok(t), Ok(s)) => (t, s),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    let route: Vec<String> = p
        .get("route")
        .map(|r| {
            r.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    let session = p.get("session").filter(|s| !s.is_empty()).cloned();
    let task_id = relay.submit(target_role, body.to_vec(), route, session);
    json_response(StatusCode::OK, json!({ "task_id": task_id }))
}

async fn get_result(
    State(relay): State<Shared>,
    Path(task_id): Path<String>,
    Query(p): Params,
) -> Response {
    let wait: f64 = match parse_num(&p, "wait", 0.0) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match relay.get_result(&task_id, wait).await {
        ResultView::NotFound => not_found(),
        ResultView::Completed(bytes) => octet(bytes),
        ResultView::Failed(error) => json_response(
            StatusCode::OK,
            json!({ "status": "FAILED", "error": error }),
        ),
        ResultView::Pending(status) => json_response(StatusCode::OK, json!({ "status": status })),
    }
}

// ---------------------------------------------------------------------------

fn app(relay: Shared) -> Router {
    Router::new()
        .route("/register", post(register))
        .route("/deregister", post(deregister))
        .route("/heartbeat", post(heartbeat))
        .route("/poll/{role}", get(poll))
        .route("/result/{task_id}", post(post_result))
        .route("/fail/{task_id}", post(post_failure))
        .route("/submit", post(submit))
        .route("/get_result/{task_id}", get(get_result))
        .fallback(|| async { not_found() })
        // Prefill hidden states of big models are tens of MB: no body limit.
        .layer(DefaultBodyLimit::disable())
        .layer(middleware::from_fn_with_state(
            relay.clone(),
            simulate_latency,
        ))
        // Outermost: rejected requests do not pay the simulated latency.
        .layer(middleware::from_fn_with_state(relay.clone(), require_token))
        .with_state(relay)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let client_token = args.token.filter(|t| !t.is_empty());
    let worker_token = args
        .worker_token
        .filter(|t| !t.is_empty())
        .or_else(|| client_token.clone());
    if client_token.is_none()
        && worker_token.is_none()
        && !["127.0.0.1", "localhost", "::1"].contains(&args.host.as_str())
    {
        eprintln!(
            "[!] WARNING: relay is reachable from the network without a token: anyone who knows the URL \
             can submit tasks or register as a worker. Set DISTLLM_TOKEN."
        );
    }
    let settings = Settings {
        client_token,
        worker_token,
        worker_rtt: Duration::from_secs_f64(args.simulate_worker_rtt_ms.max(0.0) / 1000.0),
        client_rtt: Duration::from_secs_f64(args.simulate_client_rtt_ms.max(0.0) / 1000.0),
        task_timeout: Duration::from_secs_f64(args.task_timeout.max(0.1)),
        node_timeout: Duration::from_secs_f64(args.node_timeout.max(0.1)),
    };
    let relay = Arc::new(Relay::new(settings));
    tokio::spawn(state::janitor(relay.clone()));

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .or_else(|_| format!("[{}]:{}", args.host, args.port).parse())
        .expect("invalid --host/--port");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("cannot bind");
    eprintln!("distllm relay (rust) listening on {addr}");
    axum::serve(listener, app(relay))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("server error");
}
