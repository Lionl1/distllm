"""Chaos tests: break things in the middle of a generation, demand identical output.

Real processes, real HTTP, a tiny model. Every scenario starts a generation, breaks the
system after a few tokens and checks that the generated tokens are exactly those of the
full model in one process. Runs against the Python relay and, when built, the Rust one.
Short relay/worker timeouts (3-4 s) keep failure detection fast.
"""

import asyncio
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.chaos  # slow (~5 min): run with `pytest -m chaos`

from distllm import Cluster  # noqa: E402
from distllm.hf import PipelineGenerator  # noqa: E402

SPAWN = multiprocessing.get_context("spawn")
MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])
NEW_TOKENS = 14
CHAOS_AT = 4            # break things after this many tokens
STEP_DELAY = 0.15       # seconds per stage call: stretches generation so chaos lands mid-way
TASK_TIMEOUT, NODE_TIMEOUT, HEARTBEAT = 4.0, 3.0, 0.5

ROOT = Path(__file__).parents[1]
EXE = ".exe" if os.name == "nt" else ""
RUST_BIN = ROOT / f"relay-rs/target/release/distllm-relay{EXE}"
IMPLS = ["python"] + (["rust"] if RUST_BIN.exists() else [])


# --------------------------------------------------------------------------- #
# Processes
# --------------------------------------------------------------------------- #

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"port {port} did not open")


class Relay:
    def __init__(self, impl):
        self.impl, self.port, self.proc = impl, _free_port(), None
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self):
        args = ["--port", str(self.port), "--host", "127.0.0.1",
                "--task-timeout", str(TASK_TIMEOUT), "--node-timeout", str(NODE_TIMEOUT)]
        if self.impl == "rust":
            cmd = [str(RUST_BIN), *args]
        else:
            cmd = [sys.executable, "-m", "distllm.cli", "relay", *args]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_port(self.port)

    def kill(self):
        self.proc.kill()
        self.proc.wait(timeout=10)


def _run_stage(layer, role, relay_url, node_id):
    """A pipeline stage worker; every call is slowed down by STEP_DELAY."""
    os.environ["DISTLLM_HEARTBEAT_INTERVAL"] = str(HEARTBEAT)
    import builtins

    builtins.print = lambda *a, **k: None
    from distllm import worker_node
    from distllm.hf import load_stage

    stage = load_stage(MODEL, layer, layer, device="cpu", dtype="float32")

    @worker_node(role=role, relay_url=relay_url)
    def process(payload):
        time.sleep(STEP_DELAY)
        return stage.handle(payload)

    process.start(node_id=node_id)


def _start_stage(layer, role, relay_url, node_id):
    p = SPAWN.Process(target=_run_stage, args=(layer, role, relay_url, node_id), daemon=True)
    p.start()
    return p


class LinkProxy:
    """TCP proxy between a worker and the relay whose link can be cut and restored."""

    def __init__(self, target_port):
        self.target_port, self.port = target_port, _free_port()
        self.up = True
        self._conns = []
        self._lock = threading.Lock()
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", self.port))
        self._server.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _accept(self):
        while True:
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            if not self.up:
                client.close()  # link down: connection refused/reset
                continue
            try:
                upstream = socket.create_connection(("127.0.0.1", self.target_port))
            except OSError:
                client.close()
                continue
            with self._lock:
                self._conns += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    @staticmethod
    def _pipe(src, dst):
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass

    def cut(self):
        self.up = False
        with self._lock:
            for s in self._conns:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                    s.close()
                except OSError:
                    pass
            self._conns.clear()

    def restore(self):
        self.up = True

    def close(self):
        self.cut()
        self._server.close()


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def reference():
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    seq, out = PROMPT, []
    with torch.inference_mode():
        for _ in range(NEW_TOKENS):
            token = int(model(seq).logits[0, -1].argmax())
            out.append(token)
            seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
    return out


@pytest.fixture(params=IMPLS)
def env(request):
    relay = Relay(request.param)
    relay.start()
    procs, proxies = [], []
    yield relay, procs, proxies
    for p in procs:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGCONT) if hasattr(signal, "SIGCONT") else None
            except OSError:
                pass
            p.kill()
    for proxy in proxies:
        proxy.close()
    if relay.proc and relay.proc.poll() is None:
        relay.kill()


async def _generate_with_chaos(relay_url, chaos, route=False):
    """Generate; when CHAOS_AT tokens are out, run ``chaos()`` (a coroutine) concurrently."""
    reached = asyncio.Event()
    count = {"n": 0}

    def on_token(_):
        count["n"] += 1
        if count["n"] == CHAOS_AT:
            reached.set()

    async def run_chaos():
        await reached.wait()
        started = time.monotonic()
        await chaos()
        return started

    async with Cluster(relay_url) as cluster:
        generator = PipelineGenerator(
            cluster.call, ["c-0", "c-1"], max_recoveries_per_step=5,
            route_call=cluster.call_route if route else None,
        )
        chaos_task = asyncio.create_task(run_chaos())
        result = await asyncio.wait_for(generator.generate(PROMPT, NEW_TOKENS, on_token=on_token), timeout=150)
        chaos_started = await chaos_task
    return result, time.monotonic() - chaos_started


def _wait_ready(relay_url):
    """Both stages answer (a close-session probe round-trips)."""

    async def probe():
        async with Cluster(relay_url) as cluster:
            for role in ("c-0", "c-1"):
                await cluster.call(role, {"session_id": "probe", "close": True}, timeout=60)

    asyncio.run(probe())


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("route", [False, True], ids=["per-stage", "route"])
def test_killed_worker_replaced_by_a_new_one(env, reference, route):
    relay, procs, _ = env
    procs.append(_start_stage(0, "c-0", relay.url, "s0"))
    victim = _start_stage(1, "c-1", relay.url, "s1-old")
    procs.append(victim)
    _wait_ready(relay.url)

    async def chaos():
        victim.kill()  # SIGKILL: no deregistration, no goodbye
        await asyncio.sleep(1.0)
        procs.append(_start_stage(1, "c-1", relay.url, "s1-new"))

    result, _ = asyncio.run(_generate_with_chaos(relay.url, chaos, route=route))
    assert result.token_ids == reference
    assert result.cache_recoveries >= 1


def test_bound_replica_killed_other_replica_takes_over(env, reference):
    relay, procs, _ = env
    procs.append(_start_stage(0, "c-0", relay.url, "s0"))
    replicas = [_start_stage(1, "c-1", relay.url, f"s1-{i}") for i in range(2)]
    procs.extend(replicas)
    _wait_ready(relay.url)

    async def chaos():
        # We do not know which replica holds the session: kill one; if the survivor held
        # it, the run simply continues — both cases must give the same tokens.
        replicas[0].kill()

    result, _ = asyncio.run(_generate_with_chaos(relay.url, chaos))
    assert result.token_ids == reference


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="needs POSIX signals")
def test_frozen_worker_requeued_and_late_result_ignored(env, reference):
    relay, procs, _ = env
    procs.append(_start_stage(0, "c-0", relay.url, "s0"))
    frozen = _start_stage(1, "c-1", relay.url, "s1-a")
    procs.append(frozen)
    procs.append(_start_stage(1, "c-1", relay.url, "s1-b"))
    _wait_ready(relay.url)

    async def chaos():
        os.kill(frozen.pid, signal.SIGSTOP)  # hangs mid-task: no results, no heartbeats
        await asyncio.sleep(NODE_TIMEOUT + 2)
        os.kill(frozen.pid, signal.SIGCONT)  # wakes up and posts a stale result

    result, _ = asyncio.run(_generate_with_chaos(relay.url, chaos))
    assert result.token_ids == reference


def test_network_partition_of_a_worker(env, reference):
    relay, procs, proxies = env
    proxy = LinkProxy(relay.port)
    proxies.append(proxy)
    procs.append(_start_stage(0, "c-0", proxy.url, "s0"))  # stage-0 talks through the proxy
    procs.append(_start_stage(1, "c-1", relay.url, "s1"))
    _wait_ready(relay.url)

    async def chaos():
        proxy.cut()
        await asyncio.sleep(NODE_TIMEOUT + 1.5)  # long enough to be declared dead
        proxy.restore()

    result, _ = asyncio.run(_generate_with_chaos(relay.url, chaos))
    assert result.token_ids == reference


@pytest.mark.parametrize("route", [False, True], ids=["per-stage", "route"])
def test_relay_restart_mid_generation(env, reference, route):
    relay, procs, _ = env
    procs.append(_start_stage(0, "c-0", relay.url, "s0"))
    procs.append(_start_stage(1, "c-1", relay.url, "s1"))
    _wait_ready(relay.url)

    async def chaos():
        await asyncio.sleep(STEP_DELAY / 2)  # a task is in flight on a stage right now
        relay.kill()  # all relay state is gone, including that task
        await asyncio.sleep(1.0)
        relay.start()  # same port, empty state

    result, _ = asyncio.run(_generate_with_chaos(relay.url, chaos, route=route))
    assert result.token_ids == reference
    assert result.cache_recoveries >= 1
