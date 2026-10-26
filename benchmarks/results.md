# distllm benchmark

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
