# distllm

Run a language model that does not fit on one machine across several unreliable ones —
free Google Colab GPUs, laptops, spot instances — connected through a single relay.
Nodes need no public IP, may come and go, and the output is still exactly what the full
model would produce.

| Qwen2.5-0.5B, 2 stages, CPU | tokens/s | same tokens as the full model |
|---|---:|:---:|
| old behaviour: 1 s polling, no KV cache¹ | 0.49 | yes |
| distllm 0.2 | **10.86** | yes |
| one process, no network (upper bound) | 13.38 | — |

¹ Measured with the old settings on the new code: the original pipeline example did not run
at all on transformers ≥ 4.45 (decoder layers were called without rotary embeddings).

With 60 ms of simulated network latency, speculative decoding through the pipeline gives
another **2.2×** (3.80 → 8.46 tokens/s on SmolLM2-360M). Details: [docs/benchmarks.md](docs/benchmarks.md).

## How it works

```
                       ┌──────────────┐  task (input ids)   ┌───────────────────────────────┐
client: tokenizer ───► │              │ ──────────────────► │ stage-0: embed + layers 0..11 │
  + (optional) draft   │    relay     │  hidden states      └───────────────────────────────┘
  model for            │  (HTTP, no   │ ◄── forwarded ───────────────┘
  speculative decoding │  inbound     │ ──────────────────► ┌───────────────────────────────┐
         ▲             │  ports on    │                     │ stage-1: layers 12..23 + head │
         └── logits ── │  workers)    │ ◄────────────────── └───────────────────────────────┘
                       └──────────────┘
```

- **Stages load only their layers.** A stage is built as a smaller model of the same
  architecture and reads only its tensors (and shards) from the checkpoint — optionally
  quantized to nf4/int8 on the fly.
- **Per-session KV cache on every stage**; after the prompt one token per step travels the
  pipeline. A stage that restarted answers `cache_miss`, the client replays the prefix, and
  generation continues with identical output.
- **The relay chains stages** (one client request per token) and long-polls in both
  directions, so a hop costs network latency, not polling intervals.
- **Speculative decoding**: a small draft model on the client proposes *k* tokens, the
  pipeline verifies them in one round trip — up to *k + 1* tokens per network round trip,
  still exactly greedy output of the big model.
- **Several users at once**: replicas of a stage keep each session on the node holding its
  cache, and workers batch the decode steps of concurrent sessions into one forward.
- **Sampling** (temperature, top-k, top-p, seed) besides greedy decoding.
- **Spot checks** against a trusted copy of the stages name a stage that computes wrongly.
- **One model version everywhere**: stages pin the Hub commit (`revision`), so every machine
  loads the same weights; pinned models load offline from the cache.
- **Bearer-token auth** for clients and workers with rotation without downtime, tensors on the wire in their own dtype.
- **Two relays, one protocol**: Python, or a single Rust binary with a few MB of RAM.

The full design: [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
git clone https://github.com/Lionl1/distllm.git && cd distllm
uv sync                     # add --extra quantization for nf4/int8 stages

export DISTLLM_TOKEN=$(openssl rand -hex 24)
uv run distllm relay --port 8000
# or the Rust relay — same protocol, ~2× lower per-hop latency, 4 MB of RAM, one binary:
# (cd relay-rs && cargo build --release) && relay-rs/target/release/distllm-relay --port 8000
```

On any machines that can reach the relay — your PCs, friends' computers, a VPS, or Colab
([self-hosted guide](docs/self-hosted.md), [Colab guide](docs/colab.md)). With several
machines, let `examples/plan_split.py` split the layers: it prints these commands for every
machine with the model commit pinned (`--revision`), so all of them load the same weights.

```bash
uv run python examples/distributed_hf/worker_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --layer-start 0  --layer-end 11 --role stage-0 --relay-url http://RELAY:8000
uv run python examples/distributed_hf/worker_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --layer-start 12 --layer-end 23 --role stage-1 --relay-url http://RELAY:8000
```

Generate:

```bash
uv run python examples/distributed_hf/client_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --workers stage-0,stage-1 --relay-url http://RELAY:8000 --route \
    --prompt "The future of AI is"
# Speculative decoding: add a draft model that shares the tokenizer and is much smaller
# than the target, e.g. target Qwen/Qwen2.5-7B with --draft-model-id Qwen/Qwen2.5-0.5B
```

Workers pick CUDA when available (bf16 for bf16 checkpoints on Ampere+, fp16 otherwise),
then Apple Silicon GPUs (MPS, fp16), then CPU (float32); `--device`, `--dtype` and
`--quantize nf4|fp4|int8` override it (quantization needs CUDA or CPU).

### From Python

```python
from distllm import Cluster
from distllm.hf import PipelineGenerator, SpeculativeGenerator

async with Cluster("https://relay.example", token="...") as cluster:
    generator = PipelineGenerator(cluster.call, ["stage-0", "stage-1"], route_call=cluster.call_route)
    result = await generator.generate(input_ids, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
    print(result.token_ids, result.tokens_per_second, result.cache_recoveries)

    speculative = SpeculativeGenerator(cluster.call, ["stage-0", "stage-1"], draft_model, k=4,
                                       route_call=cluster.call_route)
    result = await speculative.generate(input_ids, 50)
    print(result.acceptance_rate, result.tokens_per_round)
```

A stage in your own worker:

```python
from distllm import worker_node
from distllm.hf import load_stage

# Same commit on every machine (plan_split.py prints it; or distllm.hf.resolve_revision()).
stage = load_stage("Qwen/Qwen2.5-7B", 0, 13, quantize="nf4", revision="<commit>")

@worker_node(role="stage-0", relay_url="https://relay.example", max_batch=8, batched=True)
def process(payloads):                   # decode steps of concurrent sessions in one forward
    return stage.handle_batch(payloads)

process.start()                          # in a notebook: await process.run_async()
```

### Low-level API

The relay is not tied to transformers: any function can be a worker.

```python
import torch
from distllm import Cluster, worker_node

@worker_node(role="gpu-worker", relay_url="http://relay:8000")
def double(payload):
    return {"y": payload["x"] * 2}          # tensors, numbers, strings, nested dicts

double.start()

# elsewhere
async with Cluster("http://relay:8000") as cluster:
    print(await cluster.call("gpu-worker", {"x": torch.ones(3)}))
```

Sampling and spot checks:

```python
from distllm.hf import AuditConfig, SamplingParams

generator = PipelineGenerator(cluster.call, ["stage-0", "stage-1"],
                              audit=AuditConfig(roles=["audit-0", "audit-1"], rate=0.05))
result = await generator.generate(input_ids, 50, sampling=SamplingParams(temperature=0.7, top_p=0.9, seed=1))
# a stage that computes differently from your trusted audit-* copy raises VerificationError
```

## Documentation

- [Architecture](docs/architecture.md) — relay API and task lifecycle, stage protocol,
  KV cache and recovery, routes, speculative decoding, transport, security.
- [Your own machines, with friends](docs/self-hosted.md) — relay on a VPS / Tailscale, mixed hardware,
  splitting layers (`examples/plan_split.py`), privacy and trust.
- [Running on Google Colab](docs/colab.md) — relay behind a tunnel, workers in notebooks.
- [Benchmarks](docs/benchmarks.md) — methodology, all results, how to reproduce.
- [Rust relay](relay-rs/README.md) — drop-in replacement for the Python relay.
- [Changelog](CHANGELOG.md)

## Limitations

- One sequence per session; concurrent sessions are batched per stage for decode steps
  (prefill runs one by one). Sampling (temperature, top-k, top-p, seed) for
  `PipelineGenerator`; speculative decoding is greedy only.
- Decoder-only `*ForCausalLM` checkpoints whose tensor names match the model (Llama,
  Qwen2/2.5/3, Mistral, SmolLM, ...); multimodal checkpoints are rejected with a clear error.
- Quantization: bitsandbytes nf4/fp4/int8 (no GPTQ/AWQ checkpoints).
- Replicas of a stage role are supported: the relay keeps each session on the replica that
  holds its KV cache and moves it (with a replay) only when that worker is gone. Replicas do
  not share load within one session.
- End-to-end encryption (`DISTLLM_PAYLOAD_KEY`) hides payloads from the relay and the
  network, not from the workers that compute on them.
- The relay keeps its state in memory and needs none on disk: after a restart workers
  re-register with the sessions they hold and the client resends the lost step, so a
  generation continues without recomputation. Plain `Cluster.call` users get
  `TaskLostError` for a task lost in the restart and resubmit it themselves.
- CI runs the full test suite and the Rust relay on Linux, macOS and Windows (CPU; Apple MPS
  checked locally). CUDA paths (incl. bitsandbytes on CUDA) are not covered: CI has no GPU.

## Development

```bash
uv sync --extra test --extra quantization
uv run pytest                    # downloads a few tiny models from the Hub on first run
uv run python benchmarks/bench_pipeline.py --help
```

## License

MIT — see [LICENSE](LICENSE).
