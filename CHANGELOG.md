# Changelog

## Unreleased

- Benchmark against llama.cpp RPC (`benchmarks/compare_llamacpp.sh`, `delay_proxy.py`): speed
  on loopback and at 60 ms RTT, and what happens when a node dies; Petals attempt documented.
- Offline loading: checkpoint files are requested by name instead of listing the
  repository, so a cached (pinned) model loads with `HF_HUB_OFFLINE=1`; an unknown revision
  is diagnosed only after a failed load.
- CI on Linux, macOS and Windows for the Python tests and the Rust relay.
- Model revision pinning: `load_stage(..., revision=)`, `resolve_revision()`, `--revision` in
  the examples; `plan_split.py` pins the current commit in every command it prints.
- Rust relay (`relay-rs/`, axum + tokio): drop-in replacement with the same protocol;
  ~2.2× lower per-hop latency, 3.6 MB RSS instead of ~240 MB. HTTP conformance suite
  `tests/test_relay_http.py` runs against both relays.
- Batching: `PipelineStage.handle_batch` runs decode steps of concurrent sessions in one
  forward (padded caches + attention mask); relay `/poll?max_tasks=`; `Worker(max_batch,
  batched)`; `worker_hf.py --max-batch` (default 8). Up to 2.8× throughput on MPS with 8
  sessions.
- Auditing: `AuditConfig(roles, rate, min_similarity)` spot-checks the pipeline against
  trusted auditor stages and locates a stage that computes differently
  (`VerificationError`); client flags `--audit-roles --audit-rate`.
- Apple Silicon: `device="auto"` picks MPS after CUDA, with float16; quantization on MPS
  is rejected with a clear error.
- Session affinity for replicas: tasks of a session go to the worker of each role that
  holds its KV cache (`?session=`, set by `Cluster` from `session_id`); re-bound when that
  worker is gone.
- Sampling: `SamplingParams(temperature, top_k, top_p, seed)` for `PipelineGenerator`;
  client flags `--temperature --top-k --top-p --seed`.
- `docs/self-hosted.md`: running on your own machines with friends (relay placement,
  mixed hardware, privacy and trust).
- `examples/plan_split.py`: memory-based layer split across machines, prints the commands.

## 0.2.0

The pipeline example did not work on current transformers and every node loaded the whole
model; 0.2 rebuilds the Hugging Face path and hardens the relay.

### Pipeline inference (`distllm.hf`)
- `load_stage`: a stage is a smaller model of the same architecture; only its tensors (and
  shards) are read, layer by layer. Exact: generates the same tokens as the full model
  (Llama, Qwen2/2.5/3, Mistral; safetensors and `pytorch_model.bin`).
- Per-session KV cache on every stage with LRU/TTL eviction; `cache_miss` + prefix replay
  when a stage lost its cache; session dropped if a forward fails midway.
- `PipelineGenerator` (greedy) and `SpeculativeGenerator` (draft model on the client,
  verification in one pass, `rewind` of rejected tokens) — identical output to greedy.
- Quantized stages: bitsandbytes nf4 / fp4 / int8 (`quantize=`, `--quantize`), extra
  `distllm[quantization]`.
- `dtype="auto"` keeps bf16 for bf16 checkpoints on GPUs that support it.

### Relay and transport
- Long-poll for results (`/get_result?wait=`); no fixed sleeps in the worker loop.
- Routes: `/submit?route=` chains stages relay-side (one client request per token).
- Bearer-token auth with separate client and worker tokens (`DISTLLM_TOKEN`,
  `DISTLLM_WORKER_TOKEN`); `RelayAuthError`.
- First result wins; results from a worker the task is not assigned to are rejected.
- Chain roots survive until the chain ends; queue items are never lost on poll timeout;
  failure messages travel in the body; hidden states are never decoded by the relay.
- Tensors on the wire as raw bytes in their own dtype (bf16/fp16 no longer doubled); the old
  format is still decoded.
- Simulated network latency for experiments (`--simulate-worker-rtt-ms/--simulate-client-rtt-ms`).

### Client and worker
- `Cluster.call`, `Cluster.call_route`; `TaskLostError` on unknown tasks instead of spinning
  until the timeout; back-off on relay errors.
- Workers retry result uploads with exponential back-off; `worker_node(...).run_async()` for
  notebooks.

### Breaking changes
- `transformers >= 4.56` is required; `accelerate` and `bitsandbytes` are no longer hard
  dependencies.
- `examples/distributed_hf/worker_hf.py` no longer loads the model in 4-bit by default
  (use `--quantize nf4`); `--token` is the Hugging Face token, the relay token is
  `--relay-token`.
- Failure messages of workers are `TypeName: message`.

## 0.1.0

Initial relay/worker framework: decorator-based workers, relay with task re-queueing,
msgpack transport, HF pipeline example.
