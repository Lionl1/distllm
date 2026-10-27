# distllm vs llama.cpp RPC vs Petals

- model: `HuggingFaceTB/SmolLM2-360M` (32 layers), float32 on CPU, one machine (Apple M1, 8 GB)
- 2 nodes, prompt 37 tokens, 48 tokens generated
- llama.cpp: commit caa8191, CPU-only build with `-DGGML_RPC=ON`, f32 GGUF, `llama-bench`
- distllm: `benchmarks/bench_pipeline.py` (results_loopback.md, results_network.md)
- latency: llama.cpp through `benchmarks/delay_proxy.py` (60 ms RTT between the client and
  each RPC server); distllm through the relay's simulation (workers 60 ms, client 20 ms RTT)

## Generation speed, tokens/s

| setup | llama.cpp RPC | distllm |
|---|---:|---:|
| one process | 30.6 | 19.8 |
| 2 nodes, loopback | 35.3 | 16.0 (route 16.6) |
| 2 nodes, 60 ms RTT | 6.3 | 3.8 (route 4.4) |
| 2 nodes, 60 ms RTT, speculative k=6 | — | **8.5** |

Prompt processing (37 tokens): llama.cpp 375 t/s single, 336 t/s over RPC, 148 t/s at 60 ms.

## A node dies mid-generation

| | result |
|---|---|
| llama.cpp RPC | client aborts: `Remote RPC server crashed or returned malformed response`, SIGABRT (exit 134); the generation is lost |
| distllm | generation continues on a replacement/replica worker, tokens identical to the full model (`tests/test_chaos.py`: killed, frozen, partitioned worker, relay restart) |

## Petals

Petals 2.2.0.post1 (last release, 2023) needs Python ≤ 3.11 and torch 2.1 (it fails to import
with current torch). On macOS arm64 a private two-server swarm started and a plain forward pass
worked, but every inference session (`rpc_inference`, which `generate` uses) was reset by the
server before the handler ran — with the default `spawn` start method and with `fork` forced.
No speed numbers; needs a Linux/CUDA rerun.
