# Batched decode on one stage

- model: `HuggingFaceTB/SmolLM2-360M` (all 32 layers in one stage), torch.float16 on mps
- 16 decode steps per session after a 32-token prompt; tokens/s summed over sessions

| sessions | one by one, tok/s | batched, tok/s | speed-up |
|---:|---:|---:|---:|
| 1 | 18.8 | 18.9 | 1.00x |
| 2 | 19.6 | 27.5 | 1.40x |
| 4 | 20.2 | 38.4 | 1.90x |
| 8 | 20.0 | 56.6 | 2.84x |
