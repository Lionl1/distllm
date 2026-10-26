# Python relay vs Rust relay

Same machine (Apple Silicon laptop, CPU), echo workers in Python, client `Cluster` in Python,
`benchmarks/bench_relay.py`, median of 200 calls (decode) / 50 calls (prefill), two runs each:

| request | Python relay | Rust relay |
|---|---:|---:|
| decode step (1 token, 1.7 KiB), one hop | 4.03–4.05 ms | **1.79–1.81 ms** |
| decode step, 2-stage route | 6.79–6.84 ms | **3.17–3.22 ms** |
| prefill 512 tokens (0.9 MiB), one hop | 9.43–10.48 ms | **6.02–6.16 ms** |
| prefill, 2-stage route | 14.78–15.54 ms | **9.68–9.82 ms** |

Memory after 300 registrations: Python relay 242 MB RSS (it imports torch through the transport
module), Rust relay 3.6 MB.

End to end (SmolLM2-360M, 2 stages, route, CPU, 48 tokens, three interleaved runs) the median
time per token was 79–83 ms with the Python relay and 73–89 ms with the Rust one, against 71 ms
for the model in one process: the relay's 2–3 ms per token are below the noise of a CPU-bound
run on a busy laptop. The Rust relay pays off in per-hop latency under load, memory, and
deployment (one static binary on a small VPS, no Python).
