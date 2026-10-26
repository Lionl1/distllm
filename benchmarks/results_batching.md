# Batched decode on one stage

- model: `HuggingFaceTB/SmolLM2-360M` (all 32 layers in one stage), torch.float32 on cpu
- 16 decode steps per session after a 32-token prompt; tokens/s summed over sessions

| sessions | one by one, tok/s | batched, tok/s | speed-up |
|---:|---:|---:|---:|
| 1 | 14.9 | 14.8 | 0.99x |
| 2 | 13.6 | 7.0 | 0.52x |
| 4 | 13.4 | 14.3 | 1.06x |
| 8 | 16.0 | 27.9 | 1.75x |
