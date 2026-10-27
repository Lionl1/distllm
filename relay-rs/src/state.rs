//! Relay state: nodes, tasks, queues, session affinity — the same rules as the
//! Python relay (src/distllm/relay.py).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, watch};
use tokio::time::Instant as TokioInstant;

pub const MAX_RESULT_WAIT: f64 = 60.0;
pub const FINISHED_TTL: Duration = Duration::from_secs(300);
pub const DANGLING_TTL: Duration = Duration::from_secs(1800);
pub const SESSION_AFFINITY_TTL: Duration = Duration::from_secs(900);
pub const STOP_CHAIN_KEY: &str = "stop_chain";
pub const STOP_CHAIN_SNIFF_LIMIT: usize = 4096;
pub const MAX_ERROR_LENGTH: usize = 8000;

pub struct Settings {
    pub client_token: Option<String>,
    pub worker_token: Option<String>,
    pub worker_rtt: Duration,
    pub client_rtt: Duration,
    /// An assigned task without a result for this long is re-queued.
    pub task_timeout: Duration,
    /// A worker without a heartbeat for this long is considered dead.
    pub node_timeout: Duration,
}

impl Settings {
    pub fn janitor_interval(&self) -> Duration {
        Duration::from_secs(5)
            .min(self.task_timeout / 3)
            .min(self.node_timeout / 3)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Pending,
    Assigned,
    Forwarded,
    Completed,
    Failed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pending => "PENDING",
            Status::Assigned => "ASSIGNED",
            Status::Forwarded => "FORWARDED",
            Status::Completed => "COMPLETED",
            Status::Failed => "FAILED",
        }
    }

    fn finished(self) -> bool {
        matches!(self, Status::Completed | Status::Failed | Status::Forwarded)
    }
}

struct Task {
    target_role: String,
    payload: Arc<Vec<u8>>,
    /// Roles that receive this task's result next (chained pipeline stages).
    route: Vec<String>,
    /// The task the sender waits on; the final result of the chain lands there.
    root_id: String,
    /// Tasks of one session go to the same worker of a role (it holds the KV cache).
    session: Option<String>,
    status: Status,
    worker_id: Option<String>,
    assigned_at: Option<Instant>,
    result: Option<Vec<u8>>,
    error: Option<String>,
    created_at: Instant,
    completed_at: Option<Instant>,
    /// True when no more work is needed (a chain root only when the chain ends).
    done: watch::Sender<bool>,
}

impl Task {
    fn is_done(&self) -> bool {
        *self.done.borrow()
    }
}

struct Node {
    role: String,
    last_heartbeat: Instant,
}

#[derive(Default)]
struct Inner {
    nodes: HashMap<String, Node>,
    tasks: HashMap<String, Task>,
    role_queues: HashMap<String, VecDeque<String>>,
    worker_queues: HashMap<String, VecDeque<String>>,
    /// (role, session) -> (worker_id, last used)
    affinity: HashMap<(String, String), (String, Instant)>,
    /// Wakes pollers: "role:<name>" / "worker:<id>".
    notify: HashMap<String, Arc<Notify>>,
}

pub enum Outcome {
    NotFound,
    WrongWorker,
    Status(&'static str),
}

pub enum ResultView {
    NotFound,
    Completed(Vec<u8>),
    Failed(String),
    Pending(&'static str),
}

pub struct Relay {
    pub settings: Settings,
    inner: Mutex<Inner>,
}

fn role_key(role: &str) -> String {
    format!("role:{role}")
}

fn worker_key(worker: &str) -> String {
    format!("worker:{worker}")
}

impl Inner {
    fn notifier(&mut self, key: String) -> Arc<Notify> {
        self.notify
            .entry(key)
            .or_insert_with(|| Arc::new(Notify::new()))
            .clone()
    }

    fn push_role(&mut self, role: &str, task_id: String) {
        self.role_queues
            .entry(role.to_string())
            .or_default()
            .push_back(task_id);
        self.notifier(role_key(role)).notify_one();
    }

    fn push_worker(&mut self, worker: &str, task_id: String) {
        self.worker_queues
            .entry(worker.to_string())
            .or_default()
            .push_back(task_id);
        self.notifier(worker_key(worker)).notify_one();
    }

    /// The live worker holding this task's session for its role, if any.
    fn bound_worker(&mut self, role: &str, session: Option<&str>) -> Option<String> {
        let key = (role.to_string(), session?.to_string());
        let worker = self.affinity.get(&key)?.0.clone();
        match self.nodes.get(&worker) {
            Some(node) if node.role == role => Some(worker),
            _ => {
                self.affinity.remove(&key);
                None
            }
        }
    }

    fn enqueue(&mut self, task_id: String, task: Task) {
        let bound = self.bound_worker(&task.target_role, task.session.as_deref());
        let role = task.target_role.clone();
        self.tasks.insert(task_id.clone(), task);
        match bound {
            Some(worker) => self.push_worker(&worker, task_id),
            None => self.push_role(&role, task_id),
        }
    }

    /// Return an assigned task to its role queue (any replica may take it).
    fn requeue(&mut self, task_id: &str, reason: &str) {
        let Some(task) = self.tasks.get_mut(task_id) else {
            return;
        };
        eprintln!("Task {task_id} {reason}. Re-queuing...");
        if let (Some(session), Some(worker)) = (task.session.clone(), task.worker_id.clone()) {
            let key = (task.target_role.clone(), session);
            if self
                .affinity
                .get(&key)
                .map(|(w, _)| w == &worker)
                .unwrap_or(false)
            {
                self.affinity.remove(&key);
            }
        }
        let task = self.tasks.get_mut(task_id).unwrap();
        task.status = Status::Pending;
        task.worker_id = None;
        task.assigned_at = None;
        let role = task.target_role.clone();
        self.push_role(&role, task_id.to_string());
    }

    /// Re-queue everything a gone worker held or had queued, drop its sessions.
    fn forget_worker(&mut self, worker: &str) {
        let assigned: Vec<String> = self
            .tasks
            .iter()
            .filter(|(_, t)| t.status == Status::Assigned && t.worker_id.as_deref() == Some(worker))
            .map(|(id, _)| id.clone())
            .collect();
        for id in assigned {
            self.requeue(&id, &format!("was assigned to gone node {worker}"));
        }
        if let Some(queue) = self.worker_queues.remove(worker) {
            for id in queue {
                let role = match self.tasks.get(&id) {
                    Some(t) if t.status == Status::Pending => t.target_role.clone(),
                    _ => continue,
                };
                self.push_role(&role, id);
            }
        }
        self.affinity.retain(|_, (w, _)| w != worker);
        self.notify.remove(&worker_key(worker));
    }

    /// Pop the next pending task: this worker's sessions first, then the role queue.
    fn take_next(&mut self, worker: &str, role: &str) -> Option<String> {
        for queue_is_worker in [true, false] {
            loop {
                let queue = if queue_is_worker {
                    self.worker_queues.get_mut(worker)
                } else {
                    self.role_queues.get_mut(role)
                };
                let Some(id) = queue.and_then(VecDeque::pop_front) else {
                    break;
                };
                let now = Instant::now();
                let Some(task) = self.tasks.get_mut(&id) else {
                    continue;
                };
                if task.status != Status::Pending {
                    continue;
                }
                task.status = Status::Assigned;
                task.worker_id = Some(worker.to_string());
                task.assigned_at = Some(now);
                if let Some(session) = task.session.clone() {
                    let key = (task.target_role.clone(), session);
                    self.affinity.insert(key, (worker.to_string(), now));
                }
                return Some(id);
            }
        }
        None
    }

    /// Complete a task and, for the last step of a chain, its root task too.
    fn finish(
        &mut self,
        task_id: &str,
        status: Status,
        result: Option<Vec<u8>>,
        error: Option<String>,
    ) {
        let root_id = self.tasks.get(task_id).map(|t| t.root_id.clone());
        let now = Instant::now();
        for id in [Some(task_id.to_string()), root_id].into_iter().flatten() {
            if let Some(task) = self.tasks.get_mut(&id) {
                if task.is_done() {
                    continue;
                }
                task.status = status;
                task.result = result.clone();
                task.error = error.clone();
                task.completed_at = Some(now);
                task.done.send_replace(true);
            }
        }
    }
}

fn stops_chain(body: &[u8], flagged: bool) -> bool {
    if flagged {
        return true;
    }
    // Workers flag it explicitly; sniff only tiny bodies (a cache-miss answer is a few
    // bytes) so that hidden states are never decoded by the relay.
    if body.len() > STOP_CHAIN_SNIFF_LIMIT {
        return false;
    }
    let Ok(rmpv::Value::Map(entries)) = rmpv::decode::read_value(&mut &body[..]) else {
        return false;
    };
    entries.iter().any(|(k, v)| {
        k.as_str() == Some(STOP_CHAIN_KEY)
            && match v {
                rmpv::Value::Nil => false,
                rmpv::Value::Boolean(b) => *b,
                rmpv::Value::Integer(i) => i.as_i64() != Some(0) && i.as_u64() != Some(0),
                rmpv::Value::F32(f) => *f != 0.0,
                rmpv::Value::F64(f) => *f != 0.0,
                rmpv::Value::String(s) => !s.as_bytes().is_empty(),
                rmpv::Value::Binary(b) => !b.is_empty(),
                rmpv::Value::Array(a) => !a.is_empty(),
                rmpv::Value::Map(m) => !m.is_empty(),
                rmpv::Value::Ext(_, _) => true,
            }
    })
}

impl Relay {
    pub fn new(settings: Settings) -> Self {
        Relay {
            settings,
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn register(&self, node_id: &str, role: &str) {
        let mut g = self.lock();
        g.nodes.insert(
            node_id.to_string(),
            Node {
                role: role.to_string(),
                last_heartbeat: Instant::now(),
            },
        );
        g.role_queues.entry(role.to_string()).or_default();
        g.worker_queues.entry(node_id.to_string()).or_default();
    }

    pub fn deregister(&self, node_id: &str) {
        let mut g = self.lock();
        if let Some(node) = g.nodes.remove(node_id) {
            eprintln!("Node {node_id} ({}) deregistered gracefully.", node.role);
            g.forget_worker(node_id);
        }
    }

    pub fn heartbeat(&self, node_id: &str) -> bool {
        let mut g = self.lock();
        match g.nodes.get_mut(node_id) {
            Some(node) => {
                node.last_heartbeat = Instant::now();
                true
            }
            None => false,
        }
    }

    pub fn submit(
        &self,
        target_role: &str,
        body: Vec<u8>,
        route: Vec<String>,
        session: Option<String>,
    ) -> String {
        let task_id = uuid::Uuid::new_v4().to_string();
        let (done, _) = watch::channel(false);
        let task = Task {
            target_role: target_role.to_string(),
            payload: Arc::new(body),
            route,
            root_id: task_id.clone(),
            session,
            status: Status::Pending,
            worker_id: None,
            assigned_at: None,
            result: None,
            error: None,
            created_at: Instant::now(),
            completed_at: None,
            done,
        };
        self.lock().enqueue(task_id.clone(), task);
        task_id
    }

    /// Long-poll for up to `max_tasks` tasks; waits only for the first one.
    pub async fn poll(
        &self,
        role: &str,
        worker: &str,
        max_tasks: usize,
        timeout: Duration,
    ) -> Vec<(String, Arc<Vec<u8>>)> {
        let deadline = TokioInstant::now() + timeout;
        loop {
            let (own, shared) = {
                let mut g = self.lock();
                g.role_queues.entry(role.to_string()).or_default();
                g.worker_queues.entry(worker.to_string()).or_default();
                if let Some(first) = g.take_next(worker, role) {
                    let mut ids = vec![first];
                    while ids.len() < max_tasks {
                        match g.take_next(worker, role) {
                            Some(id) => ids.push(id),
                            None => break,
                        }
                    }
                    return ids
                        .into_iter()
                        .map(|id| {
                            let payload = g.tasks[&id].payload.clone();
                            (id, payload)
                        })
                        .collect();
                }
                (g.notifier(worker_key(worker)), g.notifier(role_key(role)))
            };
            // notify_one() stores a permit when nobody waits, so a push between the
            // unlock above and these futures is not missed.
            tokio::select! {
                _ = own.notified() => {}
                _ = shared.notified() => {}
                _ = tokio::time::sleep_until(deadline) => {
                    // One last look before answering "no task".
                    let mut g = self.lock();
                    return match g.take_next(worker, role) {
                        Some(id) => {
                            let payload = g.tasks[&id].payload.clone();
                            vec![(id, payload)]
                        }
                        None => Vec::new(),
                    };
                }
            }
        }
    }

    fn check_worker(task: &Task, worker_id: Option<&str>) -> bool {
        match &task.worker_id {
            Some(assigned) => worker_id == Some(assigned.as_str()),
            None => true,
        }
    }

    pub fn post_result(
        &self,
        task_id: &str,
        worker_id: Option<&str>,
        stop_chain: bool,
        body: Vec<u8>,
    ) -> Outcome {
        let mut g = self.lock();
        let Some(task) = g.tasks.get_mut(task_id) else {
            return Outcome::NotFound;
        };
        // A re-queued task may be computed twice: first result wins.
        if task.status.finished() {
            return Outcome::Status("ignored");
        }
        if !Self::check_worker(task, worker_id) {
            return Outcome::WrongWorker;
        }
        if !task.route.is_empty() && !stops_chain(&body, stop_chain) {
            task.status = Status::Forwarded;
            task.completed_at = Some(Instant::now());
            if task.root_id != task_id {
                task.done.send_replace(true); // nobody waits on an intermediate step
            }
            let (done, _) = watch::channel(false);
            let next = Task {
                target_role: task.route[0].clone(),
                payload: Arc::new(body),
                route: task.route[1..].to_vec(),
                root_id: task.root_id.clone(),
                session: task.session.clone(),
                status: Status::Pending,
                worker_id: None,
                assigned_at: None,
                result: None,
                error: None,
                created_at: Instant::now(),
                completed_at: None,
                done,
            };
            g.enqueue(uuid::Uuid::new_v4().to_string(), next);
            return Outcome::Status("forwarded");
        }
        g.finish(task_id, Status::Completed, Some(body), None);
        Outcome::Status("ok")
    }

    pub fn post_failure(&self, task_id: &str, worker_id: Option<&str>, message: String) -> Outcome {
        let mut g = self.lock();
        let Some(task) = g.tasks.get(task_id) else {
            return Outcome::NotFound;
        };
        if task.status.finished() {
            return Outcome::Status("ignored");
        }
        if !Self::check_worker(task, worker_id) {
            return Outcome::WrongWorker;
        }
        let message: String = message.chars().take(MAX_ERROR_LENGTH).collect();
        g.finish(task_id, Status::Failed, None, Some(message));
        Outcome::Status("ok")
    }

    /// With `wait > 0` long-polls until the task is done or `wait` (<= 60 s) passes.
    pub async fn get_result(&self, task_id: &str, wait: f64) -> ResultView {
        let mut done_rx = {
            let g = self.lock();
            let Some(task) = g.tasks.get(task_id) else {
                return ResultView::NotFound;
            };
            task.done.subscribe()
        };
        if wait > 0.0 {
            let limit = Duration::from_secs_f64(wait.min(MAX_RESULT_WAIT));
            // wait_for() checks the current value first: a finish before this is not missed.
            let _ = tokio::time::timeout(limit, done_rx.wait_for(|done| *done)).await;
        }
        let g = self.lock();
        let Some(task) = g.tasks.get(task_id) else {
            return ResultView::NotFound;
        };
        match task.status {
            Status::Completed => ResultView::Completed(task.result.clone().unwrap_or_default()),
            Status::Failed => ResultView::Failed(task.error.clone().unwrap_or_default()),
            other => ResultView::Pending(other.as_str()),
        }
    }

    /// Dead nodes, stuck tasks, old tasks, idle session bindings.
    pub fn sweep(&self) {
        let mut g = self.lock();
        let now = Instant::now();

        let dead: Vec<String> = g
            .nodes
            .iter()
            .filter(|(_, n)| now.duration_since(n.last_heartbeat) > self.settings.node_timeout)
            .map(|(id, _)| id.clone())
            .collect();
        for id in dead {
            if let Some(node) = g.nodes.remove(&id) {
                eprintln!("Node {id} ({}) timed out.", node.role);
                g.forget_worker(&id);
            }
        }

        let stuck: Vec<(String, String)> = g
            .tasks
            .iter()
            .filter(|(_, t)| {
                t.status == Status::Assigned
                    && t.assigned_at
                        .map(|a| now.duration_since(a) > self.settings.task_timeout)
                        .unwrap_or(false)
            })
            .map(|(id, t)| (id.clone(), t.worker_id.clone().unwrap_or_default()))
            .collect();
        for (id, worker) in stuck {
            g.requeue(&id, &format!("timed out on worker {worker}"));
        }

        // A chain root stays FORWARDED (not done) while later stages run: keep it.
        g.tasks.retain(|_, t| {
            if t.is_done() {
                t.completed_at
                    .map(|c| now.duration_since(c) <= FINISHED_TTL)
                    .unwrap_or(true)
            } else {
                now.duration_since(t.created_at) <= DANGLING_TTL
            }
        });
        g.affinity
            .retain(|_, (_, used)| now.duration_since(*used) <= SESSION_AFFINITY_TTL);
    }
}

pub async fn janitor(relay: Arc<Relay>) {
    let mut interval = tokio::time::interval(relay.settings.janitor_interval());
    loop {
        interval.tick().await;
        relay.sweep();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed_map(key: &str, value: rmpv::Value) -> Vec<u8> {
        let mut out = Vec::new();
        rmpv::encode::write_value(
            &mut out,
            &rmpv::Value::Map(vec![(rmpv::Value::from(key), value)]),
        )
        .unwrap();
        out
    }

    #[test]
    fn stop_chain_detection() {
        assert!(stops_chain(b"", true));
        assert!(stops_chain(
            &packed_map("stop_chain", rmpv::Value::Boolean(true)),
            false
        ));
        assert!(!stops_chain(
            &packed_map("stop_chain", rmpv::Value::Boolean(false)),
            false
        ));
        assert!(!stops_chain(
            &packed_map("other", rmpv::Value::Boolean(true)),
            false
        ));
        assert!(!stops_chain(&vec![0u8; STOP_CHAIN_SNIFF_LIMIT + 1], false));
        assert!(!stops_chain(b"\xc1not msgpack", false));
    }
}
