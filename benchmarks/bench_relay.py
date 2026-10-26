"""Round-trip cost of a task through the relay with echo workers (no model).

    uv run python benchmarks/bench_relay.py python
    uv run python benchmarks/bench_relay.py rust      # needs: cd relay-rs && cargo build --release
"""
import asyncio, multiprocessing as mp, socket, statistics, sys, time
from pathlib import Path
import torch

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

def relay(port):
    import logging; logging.getLogger("uvicorn.access").disabled = True
    from distllm.relay import start_relay
    start_relay(port=port, host="127.0.0.1")

def worker(port, role):
    import builtins; builtins.print = lambda *a, **k: None
    from distllm import worker_node
    @worker_node(role=role, relay_url=f"http://127.0.0.1:{port}")
    def echo(p): return p
    echo.start(node_id=role)

async def measure(url, size, n, route):
    from distllm import Cluster
    payload = {"session_id": "s", "start_pos": 0, "hidden_states": torch.zeros(1, size, 896, dtype=torch.bfloat16)}
    async with Cluster(url) as c:
        for _ in range(5):
            await c.call("echo-0", payload)
        times = []
        for _ in range(n):
            t = time.perf_counter()
            if route: await c.call_route(["echo-0", "echo-1"], payload)
            else: await c.call("echo-0", payload)
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)

if __name__ == "__main__":
    ctx = mp.get_context("spawn"); port = free_port(); url = f"http://127.0.0.1:{port}"
    import subprocess
    impl = sys.argv[1] if len(sys.argv) > 1 else "python"
    relay_proc = None
    if impl == "rust":
        relay_proc = subprocess.Popen([str(Path(__file__).parents[1] / "relay-rs/target/release/distllm-relay"), "--port", str(port), "--host", "127.0.0.1"], stderr=subprocess.DEVNULL)
        procs = []
    else:
        procs = [ctx.Process(target=relay, args=(port,), daemon=True)]
    procs += [ctx.Process(target=worker, args=(port, f"echo-{i}"), daemon=True) for i in range(2)]
    for p in procs: p.start()
    time.sleep(4)
    for size, label in [(1, "decode step (1 token, 1.7 KiB)"), (512, "prefill 512 tokens (0.9 MiB)")]:
        one = asyncio.run(measure(url, size, 200 if size == 1 else 50, False))
        two = asyncio.run(measure(url, size, 200 if size == 1 else 50, True))
        print(f"[{impl}] {label}: 1 hop {one:.2f} ms, 2-stage route {two:.2f} ms (median)")
    for p in procs: p.terminate()
    if relay_proc: relay_proc.terminate()
