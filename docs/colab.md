# Running on Google Colab

The goal: run a model that does not fit on one free GPU — for example **Qwen2.5-14B in nf4**
(~7 GB of transformer blocks plus 1.6 GB each for the embeddings and `lm_head` in fp16) — on two free Colab T4s,
driven from your laptop.

```
laptop: relay + cloudflared tunnel + client ◄──HTTPS──► Colab #1: stage-0 (layers 0..23)
                                             ◄──HTTPS──► Colab #2: stage-1 (layers 24..47)
```

Colab notebooks can open outbound connections but accept no inbound ones; the relay is the
only public endpoint, workers connect out to it.

> Status: the code paths used here are covered by tests in CI on Linux, macOS and Windows
> (including nf4/int8 with bitsandbytes on CPU); a full run on Colab GPUs (CUDA) has not been
> benchmarked yet.

## 1. Relay on your machine

```bash
git clone https://github.com/Lionl1/distllm.git && cd distllm
uv sync

export DISTLLM_TOKEN=$(openssl rand -hex 24)          # for the client
export DISTLLM_WORKER_TOKEN=$(openssl rand -hex 24)   # for the notebooks
uv run distllm relay --port 8000 --host 127.0.0.1
```

Expose it over HTTPS (any tunnel works; with Cloudflare no account is needed):

```bash
cloudflared tunnel --url http://localhost:8000
# → https://<random-words>.trycloudflare.com
```

The relay long-polls for at most 60 s, below the ~100 s request limit of Cloudflare's free
tunnels.

## 2. One notebook per stage

Pin the model version first, so that both notebooks load the same weights — on your
machine:

```bash
uv run python -c "from distllm.hf import resolve_revision; print(resolve_revision('Qwen/Qwen2.5-14B-Instruct'))"
```

Runtime → Change runtime type → **T4 GPU**. Put `DISTLLM_WORKER_TOKEN` (and `HF_TOKEN` for
gated models) into Colab *Secrets* (the key icon), enable notebook access.

```python
!pip -q install "distllm[quantization] @ git+https://github.com/Lionl1/distllm.git"
```

```python
import os
from google.colab import userdata

os.environ["DISTLLM_WORKER_TOKEN"] = userdata.get("DISTLLM_WORKER_TOKEN")
RELAY_URL = "https://<random-words>.trycloudflare.com"

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"   # 48 layers
REVISION = "<commit hash>"                # same in every notebook and in the client, see below
ROLE, FIRST, LAST = "stage-0", 0, 23      # second notebook: "stage-1", 24, 47
```

```python
from distllm import worker_node
from distllm.hf import load_stage

# Downloads only the shards with layers FIRST..LAST and quantizes layer by layer:
# the 12 GB of Colab RAM are enough although this fp16 slice alone is ~13 GB.
stage = load_stage(MODEL_ID, FIRST, LAST, quantize="nf4", revision=REVISION)
print(stage.name, stage.device, stage.dtype)

@worker_node(role=ROLE, relay_url=RELAY_URL, max_batch=8, batched=True)
def process(payloads):   # decode steps of concurrent sessions share one forward
    return stage.handle_batch(payloads)

await process.run_async()   # notebooks already run an event loop: not process.start()
```

T4 has no fast bfloat16, so `dtype="auto"` picks float16 there.

## 3. Generate from your machine

```bash
uv run python examples/distributed_hf/client_hf.py \
    --model-id Qwen/Qwen2.5-14B-Instruct \
    --relay-url https://<random-words>.trycloudflare.com \
    --workers stage-0,stage-1 --route --revision <commit hash> \
    --prompt "Explain pipeline parallelism in two sentences." --max-tokens 64
```

Add speculative decoding with a small model of the same family running on the laptop:

```bash
    ... --draft-model-id Qwen/Qwen2.5-0.5B-Instruct --speculative-k 4
```

The client prints prefill time, tokens/s, cache recoveries and (with a draft) the
acceptance rate.

## Choosing the split

- Split layers so that every stage fits its GPU with room for the KV cache; the first stage
  also holds the embeddings, the last one the final norm and `lm_head` (for Qwen2.5-14B each
  of them is 1.6 GB in fp16; they are not quantized).
- Fewer stages = fewer round trips per token. Use as many as memory requires, not more.
- Over the internet a round trip, not the GPU, dominates per-token time — that is where
  `--route` and speculative decoding help (see [benchmarks](benchmarks.md)).

## When things go wrong

| symptom | cause / fix |
|---|---|
| `RelayAuthError` | token mismatch: notebooks need `DISTLLM_WORKER_TOKEN`, the client `DISTLLM_TOKEN` |
| `RuntimeError: An event loop is already running` | use `await process.run_async()` in notebooks |
| Colab disconnected / runtime restarted | re-run the notebook with the same `ROLE`; the client replays the prefix on `cache_miss` and the output stays identical |
| `TaskLostError` | the relay restarted: tasks in flight are gone; the generator replays automatically, otherwise re-run |
| generation stalls ~30 s, then continues | a worker missed heartbeats; the relay re-queued its task |
| `CUDA out of memory` | smaller stages, `--quantize nf4`, or a shorter prompt (the KV cache grows with length) |
| first request is slow | workers download and load their shards on start; tasks wait in the relay queue meanwhile |
