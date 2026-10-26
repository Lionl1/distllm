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

### Limitations

- Batch size 1 and greedy decoding.
- float16 / bfloat16 / float32 only — 4-bit/8-bit quantized stages are not supported yet.
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
