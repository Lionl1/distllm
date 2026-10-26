# distllm benchmark

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
