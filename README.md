# distllm 🚀

A fault-tolerant distributed LLM inference framework designed for unstable compute nodes (like free Google Colab instances).

## Features
- **Magic API**: Simple decorator-based worker definition.
- **Relay Architecture**: Works behind NAT/Firewalls without public IPs.
- **Binary Transport**: High-speed tensor serialization via msgpack.
- **Self-Healing**: Automatic task recovery if workers crash mid-inference.
- **Pipeline-parallel HF inference**: split a Llama/Qwen/Mistral model across nodes by layers, with per-session KV cache that rebuilds itself after a node restart.

## Installation
Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/user/distllm.git
cd distllm

# Sync the environment and install dependencies
uv sync
```

## Quick Start

### 1. Start Relay (Central Server)
```bash
uv run distllm relay --port 8000
```

### 2. Define a Worker Node
Create a file named `worker.py`:

```python
from distllm import worker_node
import torch

@worker_node(role="gpu-worker", relay_url="http://your-relay-ip:8000")
def process_layers(payload):
    x = payload["hidden_states"]
    # ... your model logic ...
    return {"hidden_states": x * 2}

if __name__ == "__main__":
    process_layers.start()
```

Run the worker:
```bash
uv run python worker.py
```

### 3. Submit Tasks (Entry Node)
```python
import asyncio
import torch
from distllm import Cluster

async def main():
    cluster = Cluster("http://your-relay-ip:8000")
    task_id = await cluster.submit("gpu-worker", {"hidden_states": torch.randn(1, 10)})
    result = await cluster.wait_for(task_id)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Pipeline Inference for Hugging Face Models

`distllm.hf` splits a decoder-only model (Llama, Qwen2/2.5/3, Mistral, ...) into
**stages** — contiguous ranges of transformer layers — and runs each stage on its
own node. The client keeps only the tokenizer.

```
client (tokenizer) ──► relay ──► stage-0: embed + layers 0..11 ──► relay ──► stage-1: layers 12..23 + norm + lm_head
        ▲                                                                                  │
        └───────────────────────────────── logits of the last token ◄──────────────────────┘
```

- **Each node loads only its layers.** The stage is built as a smaller model of the
  same architecture (`num_hidden_layers = layers in the stage`) on the `meta` device,
  then only its tensors are read from the checkpoint (only the shards that contain
  them are downloaded). Rotary embeddings, the causal mask and the cache come from
  the model's own `forward`, so the split is exact: tests check that it generates the
  same tokens as the full model.
- **KV cache per session on every stage.** After the prompt, only one new token
  (or its hidden state) travels through the pipeline per step.
- **Self-healing cache.** If a stage restarted or a replacement node took its role,
  it answers `cache_miss`; the client replays the prefix once and generation
  continues with the same output.
- **Long-poll everywhere.** Workers wait on the relay for tasks, the client waits for
  results; a hop costs network latency, not a polling interval.
- **Compact transport.** Tensors are sent as raw bytes in their own dtype, so
  fp16/bf16 hidden states are not inflated to float32.

### Run it

```bash
uv run distllm relay --port 8000

# one terminal / Colab per stage (Qwen2.5-0.5B has 24 layers)
uv run python examples/distributed_hf/worker_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --layer-start 0 --layer-end 11 --role stage-0 --relay-url http://RELAY:8000
uv run python examples/distributed_hf/worker_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --layer-start 12 --layer-end 23 --role stage-1 --relay-url http://RELAY:8000

uv run python examples/distributed_hf/client_hf.py --model-id Qwen/Qwen2.5-0.5B \
    --workers stage-0,stage-1 --relay-url http://RELAY:8000 --prompt "The future of AI is"
```

Workers pick `cuda` with float16 when a GPU is available, otherwise `cpu` with
float32 (`--device`, `--dtype` override it).

From Python:

```python
from distllm import Cluster
from distllm.hf import PipelineGenerator

async with Cluster("http://RELAY:8000") as cluster:
    result = await PipelineGenerator(cluster.call, ["stage-0", "stage-1"]).generate(input_ids, 50)
print(result.token_ids, result.tokens_per_second, result.cache_recoveries)
```

### Benchmark

- model: `Qwen/Qwen2.5-0.5B` (24 layers), float32 on cpu
- stages: 2 (0-11, 12-23), relay and workers are separate processes on one machine (arm, Darwin)
- prompt: 37 tokens, generated: 32 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model |
|---|---:|---:|---:|:---:|
| single process (no network) | 659 | 69 | 13.38 | yes |
| distllm: long-poll + KV cache | 578 | 79 | 10.86 | yes |
| long-poll, no KV cache | 518 | 209 | 4.57 | yes |
| legacy: 1s polling, 0.5s worker sleep, no KV cache | 2041 | 2026 | 0.49 | yes |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 896): legacy format 130 KiB, new format 65 KiB (2.0x smaller).

On a real network every hop adds round-trip latency; the speed-up comes from removing the polling waits (≈1.5 s per hop) and recomputation (O(n²) → O(n)), which grow with distance and length.

Reproduce: `uv run python benchmarks/bench_pipeline.py --model-id Qwen/Qwen2.5-0.5B`.

### Beating network latency

On loopback a hop costs ~1 ms. Over a real network (free Colab GPUs behind a tunnel)
every request to the relay costs a round trip, and that, not GPU time, bounds the
speed: a token through 2 stages with 60 ms RTT is ~250 ms of pure networking.
Two techniques attack it:

- **Relay-side routes** (`PipelineGenerator(..., route_call=cluster.call_route)`): the
  relay forwards hidden states from stage to stage itself; the client sends one
  request per token instead of one per stage. Saves the client legs of every hop —
  the farther the client is from the relay, the bigger the win.
- **Speculative decoding** (`SpeculativeGenerator(call, roles, draft_model, k=4)`): a small
  draft model with the same tokenizer runs on the client and proposes `k` tokens; the
  pipeline verifies all of them in one pass (`output="argmax"`) and rejected positions
  are cropped from the stage caches (`rewind`). One round trip yields up to `k+1`
  tokens, and the output is exactly greedy decoding of the big model.

Simulate a network locally: `distllm relay --simulate-worker-rtt-ms 60 --simulate-client-rtt-ms 20`
(or `--worker-rtt-ms/--client-rtt-ms` in the benchmark).

**With simulated network latency:**

- model: `HuggingFaceTB/SmolLM2-360M` (32 layers), float32 on cpu; draft: `HuggingFaceTB/SmolLM2-135M`
- stages: 2 (0-15, 16-31), relay and workers are separate processes on one machine (arm, Darwin)
- network: simulated RTT to relay: workers 60 ms, client 20 ms
- prompt: 37 tokens, generated: 48 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model | notes |
|---|---:|---:|---:|:---:|---|
| single process (no network) | 481 | 51 | 18.65 | yes |  |
| long-poll + KV cache, request per stage | 591 | 259 | 3.80 | yes |  |
| + relay-side route, request per token | 537 | 229 | 4.35 | yes |  |
| + speculative decoding, k=2 | 698 | 136 | 6.06 | yes | acceptance 74%, 2.4 tokens/round |
| + speculative decoding, k=4 | 663 | 90 | 7.85 | yes | acceptance 58%, 3.2 tokens/round |
| + speculative decoding, k=6 | 691 | 70 | 8.46 | yes | acceptance 45%, 3.7 tokens/round |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 960): legacy format 139 KiB, new format 69 KiB (2.0x smaller).

Over the network the pipeline is ~5x slower than one process — pure round trips.
Routes win ~14% here (the client is close to the relay); speculative decoding wins
**2.2x** (3.80 → 8.46 tokens/s at k=6) with identical output: 3.7 tokens per round trip
even at 45% acceptance.

**Same models, no latency (loopback):**

- model: `HuggingFaceTB/SmolLM2-360M` (32 layers), float32 on cpu; draft: `HuggingFaceTB/SmolLM2-135M`
- stages: 2 (0-15, 16-31), relay and workers are separate processes on one machine (arm, Darwin)
- network: no simulated latency (loopback)
- prompt: 37 tokens, generated: 48 tokens, greedy

| mode | prefill, ms | per token, ms (median) | tokens/s | same tokens as full model | notes |
|---|---:|---:|---:|:---:|---|
| single process (no network) | 190 | 50 | 19.78 | yes |  |
| long-poll + KV cache, request per stage | 323 | 60 | 16.04 | yes |  |
| + relay-side route, request per token | 323 | 58 | 16.57 | yes |  |
| + speculative decoding, k=2 | 551 | 83 | 10.12 | yes | acceptance 74%, 2.4 tokens/round |
| + speculative decoding, k=4 | 516 | 59 | 12.68 | yes | acceptance 58%, 3.2 tokens/round |
| + speculative decoding, k=6 | 550 | 56 | 12.67 | yes | acceptance 45%, 3.7 tokens/round |

Hidden-state transfer for the 37-token prompt (bf16, hidden size 960): legacy format 139 KiB, new format 69 KiB (2.0x smaller).

Without latency speculative decoding *loses* (12.7 vs 16.6 tokens/s): the draft is only
~2.7x smaller than the target and runs on the same CPU, so drafting costs as much as it
saves. It pays off when round trips dominate — exactly the distributed case — and with a
much smaller draft relative to the target (e.g. 0.5B draft for a 7B+ target).

Reproduce:

```bash
uv run python benchmarks/bench_pipeline.py --model-id HuggingFaceTB/SmolLM2-360M \
    --draft-model-id HuggingFaceTB/SmolLM2-135M --new-tokens 48 \
    --worker-rtt-ms 60 --client-rtt-ms 20 --modes distllm,route,spec-2,spec-4,spec-6
```

### Quantized stages

`load_stage(..., quantize="nf4" | "fp4" | "int8")` (worker flag `--quantize`) quantizes the
linear layers of every transformer block with bitsandbytes while loading — layer by layer,
so peak memory is the quantized stage plus one full-precision layer. Embeddings, the final
norm and `lm_head` stay in the stage dtype. Install with `uv sync --extra quantization`
(bitsandbytes >= 0.46 also runs on CPU).

This is what makes free Colab useful: a 7B model in float16 is ~14 GB, in nf4 ~4 GB of
transformer blocks — split over two T4s it leaves plenty of room for the KV cache.

- model: `Qwen/Qwen2.5-0.5B` (24 layers), 2 stages, CPU, compute dtype float32
- 32 greedy tokens; agreement = share of positions equal to the float32 pipeline (greedy decoding diverges for good after the first different token)
- embeddings / lm_head are not quantized; they dominate the first/last stage of small models

| method | transformer blocks, MiB | largest stage, MiB | token agreement with fp32 | first differing token | per token, ms |
|---|---:|---:|---:|---:|---:|
| none | 1365 | 1202 | 100% | — | 62 |
| int8 | 342 | 690 | 19% | 2 | 176 |
| nf4 | 171 | 605 | 3% | 0 | 234 |
| fp4 | 171 | 605 | 19% | 6 | 226 |

The quantized output is identical to quantizing the same model with transformers'
`BitsAndBytesConfig` (checked for nf4 and int8), i.e. the split itself adds no error. The
quality loss is the quantization's: a 0.5B model is very sensitive — greedy decoding
diverges from float32 within a few tokens. Larger models tolerate nf4 much better; measure
on yours: `uv run --extra quantization python benchmarks/bench_quantization.py --model-id ...`.

### Security

A relay exposed through a tunnel is reachable by anyone who knows its URL: without
a token they can submit tasks to your GPUs or register as a stage and read hidden
states. Protect it with bearer tokens — one for clients, optionally another for
workers, so a worker notebook cannot submit tasks and a client cannot pose as a stage:

```bash
export DISTLLM_TOKEN=$(openssl rand -hex 24)          # clients
export DISTLLM_WORKER_TOKEN=$(openssl rand -hex 24)   # workers (defaults to DISTLLM_TOKEN)
uv run distllm relay --port 8000                       # reads both from the environment
```

`Cluster`, `Worker`, `worker_node` and the examples pick the tokens up from the same
variables (or `token=` / `--relay-token`). A rejected token raises `RelayAuthError`
immediately instead of retrying forever. The relay prints a warning when it listens on
a public interface without a token. Tokens travel in an `Authorization` header — use
an HTTPS tunnel (cloudflared, ngrok) between Colab and the relay.

### Limitations

- Batch size 1 and greedy decoding.
- Quantization is bitsandbytes nf4/fp4/int8 only (no GPTQ/AWQ checkpoints).
- Checkpoints must be decoder-only `*ForCausalLM` with tensor names matching the model
  (multimodal checkpoints such as Gemma 3 are rejected with a clear error).
- Run one worker per stage role: with several replicas of one role the KV cache is
  not pinned to a replica, so requests that land on another replica trigger a replay.

## Google Colab Workflow
To quickly set up `distllm` in a free Google Colab notebook:

```python
# 1. Install uv
!pip install uv

# 2. Install distllm and its dependencies using uv
!uv pip install distllm@git+https://github.com/user/distllm.git

# 3. Alternatively, if working on the source:
# !git clone https://github.com/user/distllm.git
# %cd distllm
# !uv pip install -e .
```

## Development
This project uses `uv` for dependency management.
- To add a new dependency: `uv add <package>`
- To run tests: `uv run pytest`
