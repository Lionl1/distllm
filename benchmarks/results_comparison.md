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

Petals 2.2.0.post1 (last release, 2023) on one GitHub Actions Linux runner (x86, 4 vCPU),
next to distllm on the same runner (`.github/workflows/bench-petals.yml`,
`benchmarks/petals/`): CPU, float32, blocks 0:16 on one server and two replicas of 16:32,
same prompt (37 tokens), 48 tokens, loopback.

| tokens/s | Petals | distllm |
|---|---:|---:|
| one process | — | 12.2 |
| 2 nodes, loopback | 8.6–8.8 | 9.9 (route 10.3) |
| 2 nodes, 60 ms RTT (distllm: relay simulation) | not measured | 3.8 (route 4.1, speculative k=6 6.4) |

Both produce the same tokens as transformers.

**A server dies** (SIGKILL 3 s into a 96-token generation): killing the idle replica changes
nothing; killing the replica that serves the session, Petals keeps retrying the dead server
(its DHT record is still valid), finds the live replica after ~4 minutes and does not finish
the generation within the 5-minute limit. distllm detects a dead worker after its node
timeout (30 s by default, 3 s in the chaos tests) and finishes with the same tokens.

**Installing it today:** `pip install petals` resolves current dependencies (protobuf 7):
servers start and plain forward passes work, but every inference session is reset before
the server handler runs — also with Petals' own CI model `Maykeye/TinyLLama-v0`, on Linux
and macOS. It works with everything resolved as of early 2024
(`uv pip install --exclude-newer 2024-02-01`, torch 2.1.2, hivemind built with
`grpcio-tools<1.61`, `protobuf<4.26`).
