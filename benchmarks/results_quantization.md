# Quantized stages

- model: `Qwen/Qwen2.5-0.5B` (24 layers), 2 stages, CPU, compute dtype float32
- 32 greedy tokens; agreement = share of positions equal to the float32 pipeline (greedy decoding diverges for good after the first different token)
- embeddings / lm_head are not quantized; they dominate the first/last stage of small models

| method | transformer blocks, MiB | largest stage, MiB | token agreement with fp32 | first differing token | per token, ms |
|---|---:|---:|---:|---:|---:|
| none | 1365 | 1202 | 100% | — | 62 |
| int8 | 342 | 690 | 19% | 2 | 176 |
| nf4 | 171 | 605 | 3% | 0 | 234 |
| fp4 | 171 | 605 | 19% | 6 | 226 |
