# Your own machines, with friends

distllm is not tied to Colab. A worker is any machine with Python ≥ 3.10, PyTorch and an
**outbound** connection to the relay — no public IP, no open ports. Mix whatever you
have: a desktop with an RTX card, a friend's gaming PC, a home server, a rented GPU, a
laptop on CPU.

```
              ┌───────────── relay (VPS / your PC behind a tunnel / Tailscale) ─────────────┐
you: client ──┤   stage-0 (you)  →  stage-1 (Alice)  →  stage-2 (Bob)  →  stage-3 (you)      │
              └──────────────────────────────────────────────────────────────────────────────┘
```

> Status: everything here is covered by tests with separate processes over HTTP, run in CI on
> Linux, macOS and Windows (CPU) and locally on Apple GPUs. CUDA and real internet links
> have not been benchmarked yet. Expect rough edges and please report them.

## 1. Where to put the relay

The relay must be reachable by every participant. It runs no model code, so the
smallest VPS or any always-on machine is enough — especially with the Rust relay
(`relay-rs/`: one binary, a few MB of RAM, no Python needed on the server; same protocol and
flags as `distllm relay`).

| option | when | notes |
|---|---|---|
| **VPS with a public IP** | the simplest for a group | put it behind HTTPS (e.g. Caddy: `caddy reverse-proxy --from relay.example.com --to localhost:8000`) |
| **Tailscale / ZeroTier** | everyone can install a client | private network: no public endpoint at all, usually lower latency; run the relay with `--host 0.0.0.0` and use the tailnet IP |
| **Your PC + tunnel** | quick tests | `cloudflared tunnel --url http://localhost:8000` gives an HTTPS URL; long-polls stay under its ~100 s limit |

Always use tokens, also inside a private network:

```bash
export DISTLLM_TOKEN=$(openssl rand -hex 24)          # keep for yourself (client)
export DISTLLM_WORKER_TOKEN=$(openssl rand -hex 24)   # give to friends (workers)
uv run distllm relay --port 8000
```

With separate tokens a friend's worker can only take tasks of a role and return results
for tasks assigned to it; it cannot submit tasks or read other sessions' results.

## 2. Hardware

| device | status | notes |
|---|---|---|
| NVIDIA GPU (CUDA) | main target | `--device auto` picks it; bf16 for bf16 checkpoints on Ampere+ (RTX 30xx/40xx, A100…), fp16 otherwise; `--quantize nf4/int8` works |
| CPU (x86, ARM) | tested | slow, but fine for small models and for thin first/last stages; float32 by default |
| Apple Silicon (MPS) | tested | chosen by `auto` when there is no CUDA, float16 (matched float32 output exactly in our checks; bf16 drifted); bitsandbytes quantization does not run on MPS |
| AMD (ROCm) / Intel GPUs | untested | PyTorch ROCm builds expose the GPU as `cuda` and may just work; no guarantees |

Operating systems: Linux, macOS and Windows — CI runs all tests (Python 3.10/3.12, CPU,
bitsandbytes included) and builds and tests the Rust relay on each of them.

Stages on different machines can be mixed freely — Windows, macOS and Linux, GPU and CPU,
fp16 and bf16 in one pipeline: each stage converts incoming hidden states to its own dtype,
and the tiny numeric differences stay within the audit tolerance. Tensors travel as raw
little-endian bytes, which covers every mainstream CPU (x86, ARM, Apple); big-endian
machines (e.g. IBM s390x) are not supported. Two things must match on all machines: the
**model revision** (below) and the **distllm version**.

## 3. Split the model

`examples/plan_split.py` estimates memory from the model's architecture (only
`config.json` is downloaded) and splits layers in proportion to each machine's free
memory. List machines in pipeline order, `name:GB` (VRAM for a GPU, RAM for CPU);
`name:GB:N` fixes a stage to N layers; repeat a name to give one machine several stages:

```bash
uv run python examples/plan_split.py --model-id Qwen/Qwen2.5-14B-Instruct \
    --nodes me:6:2,alice:10,bob:10,me:6:2 --quantize nf4 --context 4096 \
    --relay-url https://relay.example.com
```

```
machine    role      layers      est. GB / machine GB
me         stage-0     0..1          1.9 /        6.0
alice      stage-1     2..23         3.6 /       10.0
bob        stage-2    24..45         3.6 /       10.0
me         stage-3    46..47         1.9 /        6.0
```

…followed by the exact command for every machine and for the client. The estimate covers
weights plus a KV cache for one `--context`-token session and keeps 15% headroom.

**Everybody must load the same model version.** Each machine downloads its own slice of the
weights; if the Hub repository is updated between two downloads, the slices come from
different models and the output is garbage. `plan_split.py` therefore resolves the current
commit of the model (or `--revision` you give it) and puts `--revision <commit>` into every
command it prints — keep it. In Python: `load_stage(..., revision=resolve_revision(model_id))`.
Use the same distllm version on all machines too.

Rules of thumb:

- **As few stages as memory allows.** Every stage adds a network hop per token.
- **Fast machines in the middle, big memory where the layers are.** The first stage also
  holds the embeddings, the last one `lm_head` (each 1.6 GB in fp16 for Qwen2.5-14B; they
  are never quantized).
- The KV cache grows with the context and with every concurrent session.

## 4. Privacy and trust — read before inviting people

What each participant can see:

| stage | receives | sees |
|---|---|---|
| first | `input_ids` | **your prompt in plain text** |
| middle | hidden states | no tokens directly; after only a few layers the states are close to token embeddings and can be mapped back to the text — not encryption |
| last | hidden states | **the generated tokens** (it computes them) |

So:

- **Keep the first and the last stage on your own machine** (`me:GB:N` at both ends, as
  above). They are light on compute; give them a few layers so that what leaves your
  machine is several layers deep, not raw embeddings.
- **Results are not verified unless you ask for it.** A worker can return garbage. Run a
  second, trusted copy of the stages (e.g. on your machine: `worker_hf.py` with roles
  `audit-0`, `audit-1`, …, the same layer ranges and revision) and pass `--audit-roles audit-0,audit-1 --audit-rate 0.05`: on ~5% of
  the tokens the client recomputes the prefix through the auditors and compares the logits;
  a mismatch stops generation with `VerificationError` naming the stage that computes
  differently. It catches a broken or cheating stage at the cost of occasional full
  prefills; it does not make a pipeline of strangers trustworthy.
- **Hide traffic from the relay:** `export DISTLLM_PAYLOAD_KEY=$(distllm keygen)` on your
  machine and give the same value to every worker (`pip install distllm[encryption]`).
  Prompts, hidden states and tokens then cross the relay and the tunnel encrypted; the relay
  host learns only who talks to whom, when and how much. Workers still see what they compute.
- Tokens are bearer secrets: share them over a private channel, rotate them when someone
  leaves, use HTTPS or a private network.

## 5. Run it

Every friend, once:

```bash
git clone https://github.com/Lionl1/distllm.git && cd distllm
uv sync --extra quantization
export DISTLLM_WORKER_TOKEN=...   # from you
```

then the command `plan_split.py` printed for them, e.g.:

```bash
uv run python examples/distributed_hf/worker_hf.py --model-id Qwen/Qwen2.5-14B-Instruct \
    --layer-start 2 --layer-end 23 --role stage-1 --relay-url https://relay.example.com \
    --dtype float16 --quantize nf4 --revision <commit printed by plan_split>
```

Each machine downloads only the shards with its layers. You run your stages (two
terminals for `stage-0` and `stage-3` in the example) and the client:

```bash
export DISTLLM_TOKEN=...
uv run python examples/distributed_hf/client_hf.py --model-id Qwen/Qwen2.5-14B-Instruct \
    --workers stage-0,stage-1,stage-2,stage-3 --relay-url https://relay.example.com --route \
    --revision <commit> --prompt "..." --max-tokens 200
```

Add `--draft-model-id Qwen/Qwen2.5-0.5B-Instruct --speculative-k 4` to verify several
tokens per network round trip (the draft runs on your machine and must share the
tokenizer), or `--temperature 0.7 --top-p 0.9` to sample. Speculative decoding is greedy
only and cannot be combined with sampling or `--audit-roles`.

## 6. Day-to-day

- **Someone leaves or reboots**: with a replica of their role online, the session moves
  there right away; otherwise generation waits until the role is served again (same command
  on any machine). Either way the client replays the prefix and continues with the same
  output. Nothing to restart on your side.
- **Several people generating at once** works — each session has its own KV cache on
  every stage (8 sessions per stage by default), and workers batch the decode steps of
  concurrent sessions into one forward (`--max-batch`, default 8): on an Apple GPU 8
  sessions gave 2.8× the throughput of running them one by one. Prompts (prefill) still run
  one at a time.
- **Several friends may serve the same role** (replicas): the relay keeps each session on
  the replica that holds its KV cache, so different sessions spread over replicas without
  replays. If that friend leaves, the session moves to another replica (one replay).
- **Speed** is dominated by round trips: with ~60 ms to the relay expect a few tokens per
  second for a 2-stage pipeline, up to ~2× more with speculative decoding
  ([benchmarks](benchmarks.md), simulated latency).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `RelayAuthError` on a friend's machine | they need `DISTLLM_WORKER_TOKEN`, not the client token |
| a stage never gets tasks | the role name differs from the client's `--workers` list |
| `CUDA out of memory` | fewer layers on that machine, `--quantize nf4`, smaller `--context` / shorter prompts |
| everything stalls ~30 s, then continues | a worker missed heartbeats; its task was re-queued |
| `TaskLostError` | the relay restarted; the generator resends the step automatically |
| garbage output only with one participant | that worker computes wrongly: a different `--revision` (or none), wrong model id, broken GPU, or bad intent — `--audit-roles` names it |

See also: [architecture](architecture.md), [Colab guide](colab.md).
