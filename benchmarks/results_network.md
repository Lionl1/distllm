# distllm benchmark

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
