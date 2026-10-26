# Architecture

distllm has three kinds of processes:

- **relay** — the only process that must be reachable by everyone. An HTTP server (FastAPI)
  with in-memory task queues, one per *role*. It never runs model code.
- **workers** — connect *out* to the relay (no inbound ports, works behind NAT and in Colab),
  long-poll it for tasks of their role, run a function, post the result.
- **clients** — submit tasks and wait for results. For LLM inference the client holds only
  the tokenizer (and optionally a small draft model).

```
src/distllm/
├── relay.py      relay server: queues, task lifecycle, routes, auth, latency simulation
├── worker.py     Worker / worker_node (worker side), Cluster (client side)
├── transport.py  msgpack payloads with tensors as raw bytes
├── hf.py         Hugging Face pipeline: load_stage, PipelineStage, PipelineGenerator,
│                 SpeculativeGenerator
└── cli.py        `distllm relay`, `distllm worker` (echo worker for health checks)
```

## Relay

### Endpoints

| endpoint | side | purpose |
|---|---|---|
| `POST /register?node_id&role` | worker | announce a node of a role |
| `POST /heartbeat?node_id` | worker | every 10 s; 404 means "relay restarted, register again" |
| `POST /deregister?node_id` | worker | graceful exit; its assigned tasks are re-queued at once |
| `GET /poll/{role}?worker_id` | worker | long-poll (≤ 5 s) for a task: 200 with `{task_id, payload}` or 204 |
| `POST /result/{task_id}?worker_id&stop_chain` | worker | body = msgpack result |
| `POST /fail/{task_id}?worker_id` | worker | body = error message (text, ≤ 8000 chars kept) |
| `POST /submit?target_role&sender_id&route` | client | body = msgpack payload; returns `{task_id}` |
| `GET /get_result/{task_id}?wait` | client | long-poll (≤ 60 s): msgpack result, `{"status": "FAILED", "error"}`, or `{"status": ...}` while pending |

### Task lifecycle

```
PENDING ──poll──► ASSIGNED ──result──► COMPLETED
   ▲                 │  └────fail────► FAILED
   └── re-queue ─────┘  (worker missed heartbeats for 30 s, deregistered,
                         or did not answer within TASK_TIMEOUT = 30 s)
```

- **First result wins.** A re-queued task may be computed twice; any later result or
  failure for a finished task is ignored, and `/result`/`/fail` from a worker other than
  the one the task is currently assigned to are rejected with 409.
- **Cleanup.** Finished tasks are kept 5 minutes after completion; unfinished ones 30
  minutes after creation.
- `/poll` never loses a dequeued task id on timeout or cancellation (a known
  `asyncio.wait_for(queue.get())` race on Python < 3.12).

### Routes (chained stages)

`/submit?route=stage-1,stage-2` makes the relay forward each step's result as the payload
of the next role. The client waits on the first (root) task and receives the last step's
result — one client round trip per token instead of one per stage.

- An intermediate step becomes `FORWARDED`; the root stays alive until the chain ends,
  however long a later stage takes (e.g. a worker still downloading its weights).
- A failure anywhere fails the root.
- A worker result flagged with `?stop_chain=true` (a stage's `cache_miss`) ends the chain
  early and goes straight to the client. The relay never decodes hidden-state payloads;
  only bodies ≤ 4 KiB are inspected for a `stop_chain` key, for compatibility.

### Authentication

`DISTLLM_TOKEN` protects client endpoints, `DISTLLM_WORKER_TOKEN` (defaults to the
client token) worker endpoints: `Authorization: Bearer <token>`, compared in constant
time. Separate tokens mean a worker notebook cannot submit tasks and a client cannot
register as a stage and read hidden states. Without tokens the relay is open and warns
when bound to a public interface. Use an HTTPS tunnel — the token is a bearer secret.

### Latency simulation

`start_relay(worker_rtt_ms=..., client_rtt_ms=...)` / `distllm relay
--simulate-worker-rtt-ms --simulate-client-rtt-ms` add one artificial round trip per HTTP
request, half before and half after handling, separately for worker and client
endpoints. For experiments only.

## Client and worker (`worker.py`)

- `Cluster.call(role, payload)` = submit + long-poll wait; `Cluster.call_route(roles,
  payload)` uses a route. `TaskLostError` when the relay no longer knows the task (404),
  `RelayAuthError` on 401; other relay errors are retried with a 1 s back-off until the
  timeout.
- `Worker` registers with retries, heartbeats every 10 s, re-registers after a relay
  restart, uploads results with exponential back-off (5 attempts), and reports processing
  errors as `TypeName: message` in the body of `/fail`. Sync functions run in a thread so
  heartbeats keep flowing during long computations.

## Transport (`transport.py`)

Payloads are msgpack dicts. A tensor is encoded as
`{"__tensor__": true, "dtype": "bfloat16", "shape": [1, 37, 896], "data": <raw bytes>}` —
fp16/bf16 stay two bytes per value (the previous format went through float32 numpy and
doubled the traffic; it is still decoded). Only a whitelist of dtypes is accepted.

## Pipeline stages (`hf.py`)

### Loading: `load_stage(model_id, layer_start, layer_end, *, device, dtype, quantize, ...)`

1. Copy the config with `num_hidden_layers = layer_end - layer_start + 1` (and the matching
   slice of `layer_types`, for models that mix sliding and full attention) and build the
   model on the `meta` device. The stage *is* a small model of the same architecture, so
   its own `forward` computes rotary embeddings, the causal mask and cache indexing —
   nothing is re-implemented per architecture.
2. Map stage parameter names to checkpoint names (`layers.{i}` → `layers.{layer_start+i}`);
   embeddings for the first stage, final norm and `lm_head` for the last (with tied
   embeddings `lm_head` reuses the embedding matrix).
3. Download only the shards that contain those tensors (via the safetensors/bin index) and
   read them lazily, **one transformer layer at a time**: convert to the dtype, optionally
   quantize the block's `nn.Linear` layers with bitsandbytes (nf4/fp4/int8) while moving
   them to the device. Peak memory: the (quantized) stage plus one full-precision layer.
4. Recreate non-persistent buffers (rotary `inv_freq`) on the device; drop modules the
   stage does not need; fail loudly if anything is left on `meta`.

`dtype="auto"`: CPU → float32; CUDA → bfloat16 for bf16 checkpoints on GPUs that support
it, float16 otherwise.

### Stage protocol: `PipelineStage.handle(payload)`

| field | meaning |
|---|---|
| `session_id` | one generation; the stage keeps a `DynamicCache` per session (LRU, 8 sessions, 10 min TTL) |
| `start_pos` | position of the first new token. `0` starts a fresh cache; otherwise it must equal the cached length |
| `input_ids` / `hidden_states` | new tokens (first stage) or their hidden states (other stages) |
| `rewind` | allow `start_pos` below the cached length: crop the cache first (rejected draft tokens) |
| `output` | last stage only: `"logits"` (last position, default) or `"argmax"` (greedy id at every new position) |
| `close` | drop the session |

Answers: intermediate stages return a ready payload for the next stage (control fields +
`hidden_states`), the last stage `{"logits"}` or `{"argmax"}`. A stage that does not hold
the expected prefix answers `{"cache_miss": true, "cached": n, "stop_chain": true}`. If a
forward fails midway (e.g. CUDA OOM), the session is dropped, because earlier layers have
already appended to the cache.

### Generation: `PipelineGenerator`

```
step 0:  stage-0..N  ←  prompt (start_pos = 0)          → logits → token
step t:  stage-0..N  ←  last token (start_pos = length)  → logits → token
on cache_miss or TaskLostError: replay sequence from start_pos = 0 (≤ 2 times per step)
```

With `route_call=cluster.call_route` a step is one client request; otherwise one per stage.
The next token is picked on the client from the last stage's logits: greedy by default,
or `generate(..., sampling=SamplingParams(temperature, top_k, top_p, seed))` — seeded
sampling is reproducible and equal to sampling from the full model with the same seed.
Sessions are closed on all stages at the end (best effort; TTL is the safety net).

### Speculative decoding: `SpeculativeGenerator`

Each round:

1. The draft model (on the client, in a worker thread) proposes `k` greedy tokens
   `d1..dk`, reusing its own KV cache.
2. The pipeline runs `[pending, d1..dk]` in one pass with `rewind=True, output="argmax"`
   and returns the target's greedy choice `a0..ak` after every prefix.
3. Accept the longest prefix with `d(i+1) == a(i)`, plus the target's own next token
   `a(m)`: `m + 1` new tokens. The stage caches keep `pending + d1..dm`; the next round's
   `rewind` crops the rejected tail, the draft cache is cropped the same way.

The result equals greedy decoding of the target exactly; only the number of round trips
changes. It pays off when round trips dominate and the draft is much cheaper than the
target (see [benchmarks](benchmarks.md)).

## Failure handling at a glance

| failure | what happens |
|---|---|
| worker dies mid-task | missed heartbeats → task re-queued after ≤ 30 s → next worker of the role; a stage without the cache answers `cache_miss` → client replays the prefix |
| worker restarts | same; its sessions are gone, replay rebuilds them |
| result upload fails | worker retries with back-off; if it gives up, the relay re-queues on timeout |
| task computed twice | first result wins; KV caches stay consistent (`start_pos` check / `rewind`) |
| relay restarts | workers re-register on the next heartbeat; clients get `TaskLostError` → replay |
| forward raises | session dropped, error returned to the client as `RuntimeError("Task failed on worker: ...")` |
| wrong token | `RelayAuthError` immediately on both sides |
