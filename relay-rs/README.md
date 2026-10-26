# distllm-relay (Rust)

A drop-in replacement for `distllm relay`: same HTTP protocol, bodies and status codes, so the
Python workers, `Cluster` and the examples work unchanged.

```bash
cargo build --release
./target/release/distllm-relay --port 8000                  # reads DISTLLM_TOKEN / DISTLLM_WORKER_TOKEN
./target/release/distllm-relay --help
```

Implements everything the Python relay does: long-poll for tasks and results, routes with
`stop_chain`, first result wins and worker checks, session affinity for replicas, batched
`/poll?max_tasks=`, bearer tokens for clients and workers, failure messages in the body,
latency simulation, the janitor (dead nodes, stuck tasks, cleanup) — with no body size limit.

Conformance is checked by the Python suites, run against this binary:

```bash
cd .. && uv run pytest tests/test_relay_http.py                 # HTTP conformance, both relays
DISTLLM_TEST_RELAY=rust uv run pytest tests/test_integration.py  # real workers and models
```

Numbers: [../benchmarks/results_relay.md](../benchmarks/results_relay.md) — ~2.2× lower
per-hop latency than the Python relay, 3.6 MB of memory instead of ~240 MB.
