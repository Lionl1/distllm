"""Relay conformance over real HTTP — runs against the Python relay and, when built,
the Rust one (relay-rs/target/release/distllm-relay or $DISTLLM_RELAY_BIN)."""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from distllm.transport import pack_payload, unpack_payload

ROOT = Path(__file__).parents[1]
EXE = ".exe" if os.name == "nt" else ""
RUST_BIN = Path(os.environ.get("DISTLLM_RELAY_BIN", ROOT / f"relay-rs/target/release/distllm-relay{EXE}"))
IMPLS = ["python"] + (["rust"] if RUST_BIN.exists() else [])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(impl, token=None, worker_token=None, worker_rtt_ms=0, client_rtt_ms=0):
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k not in ("DISTLLM_TOKEN", "DISTLLM_WORKER_TOKEN")}
    if impl == "rust":
        cmd = [str(RUST_BIN), "--port", str(port), "--host", "127.0.0.1",
               "--simulate-worker-rtt-ms", str(worker_rtt_ms), "--simulate-client-rtt-ms", str(client_rtt_ms)]
        if token:
            cmd += ["--token", token]
        if worker_token:
            cmd += ["--worker-token", worker_token]
    else:
        code = (
            "import logging; logging.getLogger('uvicorn.access').disabled = True\n"
            "from distllm.relay import start_relay\n"
            f"start_relay(port={port}, host='127.0.0.1', token={token!r}, worker_token={worker_token!r}, "
            f"worker_rtt_ms={worker_rtt_ms}, client_rtt_ms={client_rtt_ms})"
        )
        cmd = [sys.executable, "-c", code]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc, f"http://127.0.0.1:{port}"
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"{impl} relay did not start")


@pytest.fixture(params=IMPLS, scope="module")
def relay(request):
    proc, url = _start(request.param)
    with httpx.Client(base_url=url, timeout=30) as client:
        client.impl = request.param
        yield client
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(params=IMPLS)
def relay_factory(request):
    procs = []

    def make(**opts):
        proc, url = _start(request.param, **opts)
        procs.append(proc)
        return httpx.Client(base_url=url, timeout=30)

    yield make
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=10)


def submit(c, role, payload=None, **params):
    resp = c.post("/submit", content=pack_payload(payload or {"x": 1}),
                  params={"target_role": role, "sender_id": "c", **params})
    assert resp.status_code == 200, resp.text
    return resp.json()["task_id"]


def poll(c, role, worker, **params):
    resp = c.get(f"/poll/{role}", params={"worker_id": worker, **params})
    assert resp.status_code == 200, (resp.status_code, resp.text)
    assert resp.headers["content-type"] == "application/octet-stream"
    return unpack_payload(resp.content)


def test_lifecycle(relay):
    relay.post("/register", params={"node_id": "w", "role": "life"})
    tid = submit(relay, "life", {"x": [1, 2, 3]})
    task = poll(relay, "life", "w")
    assert task["task_id"] == tid and unpack_payload(task["payload"]) == {"x": [1, 2, 3]}
    assert relay.get(f"/get_result/{tid}").json() == {"status": "ASSIGNED"}
    assert relay.post(f"/result/{tid}", content=pack_payload({"y": 2}), params={"worker_id": "w"}).json() == {"status": "ok"}
    got = relay.get(f"/get_result/{tid}")
    assert got.headers["content-type"] == "application/octet-stream"
    assert unpack_payload(got.content) == {"y": 2}


def test_long_poll_result_and_empty_poll(relay):
    tid = submit(relay, "lp")
    threading.Timer(0.3, lambda: relay.post(f"/result/{tid}", content=pack_payload({"y": 1}))).start()
    started = time.perf_counter()
    resp = relay.get(f"/get_result/{tid}", params={"wait": 10})
    assert 0.25 < time.perf_counter() - started < 2.5
    assert unpack_payload(resp.content) == {"y": 1}
    started = time.perf_counter()
    assert relay.get("/poll/nobody", params={"worker_id": "w"}).status_code == 204
    assert 4.5 < time.perf_counter() - started < 7


def test_poll_waits_for_a_task(relay):
    threading.Timer(0.3, lambda: submit(relay, "late")).start()
    started = time.perf_counter()
    poll(relay, "late", "w")
    assert time.perf_counter() - started < 2.5


def test_route_stop_chain_and_failures(relay):
    root = submit(relay, "ra", route="rb,rc")
    poll(relay, "ra", "w")
    assert relay.post(f"/result/{root}", content=pack_payload({"h": 1}), params={"worker_id": "w"}).json() == {"status": "forwarded"}
    assert relay.get(f"/get_result/{root}").json() == {"status": "FORWARDED"}
    b = poll(relay, "rb", "w")
    assert unpack_payload(b["payload"]) == {"h": 1}
    relay.post(f"/result/{b['task_id']}", content=pack_payload({"h": 2}), params={"worker_id": "w"})
    c = poll(relay, "rc", "w")
    relay.post(f"/result/{c['task_id']}", content=pack_payload({"final": True}), params={"worker_id": "w"})
    assert unpack_payload(relay.get(f"/get_result/{root}").content) == {"final": True}

    # stop_chain: flagged, and sniffed in a small body
    for params, body in (({"stop_chain": "true"}, {"miss": 1}), ({}, {"cache_miss": True, "stop_chain": True})):
        root = submit(relay, "sa", route="sb")
        poll(relay, "sa", "w")
        relay.post(f"/result/{root}", content=pack_payload(body), params={"worker_id": "w", **params})
        assert unpack_payload(relay.get(f"/get_result/{root}").content) == body

    # failure in a chain fails the root; message from the body, capped
    root = submit(relay, "fa", route="fb")
    poll(relay, "fa", "w")
    relay.post(f"/result/{root}", content=pack_payload({"h": 1}), params={"worker_id": "w"})
    b = poll(relay, "fb", "w")
    relay.post(f"/fail/{b['task_id']}", content=("E" * 20000).encode(), params={"worker_id": "w"})
    failed = relay.get(f"/get_result/{root}").json()
    assert failed["status"] == "FAILED" and failed["error"] == "E" * 8000


def test_first_result_wins_and_worker_checks(relay):
    tid = submit(relay, "own")
    poll(relay, "own", "owner")
    assert relay.post(f"/result/{tid}", content=pack_payload({}), params={"worker_id": "intruder"}).status_code == 409
    assert relay.post(f"/result/{tid}", content=pack_payload({})).status_code == 409
    assert relay.post(f"/fail/{tid}", params={"error_msg": "x"}).status_code == 409
    assert relay.post(f"/result/{tid}", content=pack_payload({"y": 1}), params={"worker_id": "owner"}).json() == {"status": "ok"}
    assert relay.post(f"/result/{tid}", content=pack_payload({"y": 2}), params={"worker_id": "owner"}).json() == {"status": "ignored"}
    assert relay.post(f"/fail/{tid}", params={"error_msg": "late", "worker_id": "owner"}).json() == {"status": "ignored"}
    assert unpack_payload(relay.get(f"/get_result/{tid}").content) == {"y": 1}
    # an unassigned task accepts a result from anyone (first wins)
    free = submit(relay, "free")
    assert relay.post(f"/result/{free}", content=pack_payload({"z": 1})).json() == {"status": "ok"}


def test_not_found_and_validation(relay):
    assert relay.get("/get_result/nope").status_code == 404
    assert relay.post("/result/nope", content=b"").status_code == 404
    assert relay.post("/fail/nope").status_code == 404
    assert relay.post("/heartbeat", params={"node_id": "ghost"}).status_code == 404
    relay.post("/register", params={"node_id": "alive", "role": "r"})
    assert relay.post("/heartbeat", params={"node_id": "alive"}).json() == {"status": "ok"}
    assert relay.post("/submit", content=b"").status_code == 422
    assert relay.get("/poll/r").status_code == 422


def test_session_affinity(relay):
    for w in ("a1", "a2"):
        relay.post("/register", params={"node_id": w, "role": "aff"})
    first = submit(relay, "aff", session="s1")
    assert poll(relay, "aff", "a1")["task_id"] == first
    relay.post(f"/result/{first}", content=pack_payload({}), params={"worker_id": "a1"})
    second = submit(relay, "aff", session="s1")
    assert poll(relay, "aff", "a1")["task_id"] == second  # stays on a1
    relay.post(f"/result/{second}", content=pack_payload({}), params={"worker_id": "a1"})
    third = submit(relay, "aff", session="s1")
    relay.post("/deregister", params={"node_id": "a1"})
    assert poll(relay, "aff", "a2")["task_id"] == third  # moved to the other replica


def test_reregistered_worker_reclaims_its_sessions(relay):
    """After a relay restart the affinity map is empty: a worker claims its sessions back."""
    relay.post("/register", params={"node_id": "r1", "role": "rec"})
    relay.post("/register", params={"node_id": "r2", "role": "rec", "sessions": "s7,s8"})
    bound = submit(relay, "rec", session="s7")
    free = submit(relay, "rec")
    assert poll(relay, "rec", "r1")["task_id"] == free  # s7 is not in the role queue...
    assert poll(relay, "rec", "r2")["task_id"] == bound  # ...it waits for r2
    # A live replica keeps its session: a later claim by another worker is ignored.
    relay.post("/register", params={"node_id": "r1", "role": "rec", "sessions": "s8"})
    again = submit(relay, "rec", session="s8")
    assert poll(relay, "rec", "r2")["task_id"] == again


def test_batched_poll(relay):
    ids = [submit(relay, "bat") for _ in range(5)]
    first = poll(relay, "bat", "bw", max_tasks=3)
    assert [t["task_id"] for t in first["tasks"]] == ids[:3]
    rest = poll(relay, "bat", "bw", max_tasks=8)
    assert [t["task_id"] for t in rest["tasks"]] == ids[3:]


def test_large_payload(relay):
    blob = os.urandom(8 * 1024 * 1024)
    tid = submit(relay, "big", {"blob": blob})
    assert unpack_payload(poll(relay, "big", "w")["payload"])["blob"] == blob


def test_auth(relay_factory):
    c = relay_factory(token="client-t", worker_token="worker-t")
    bearer = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731
    assert c.post("/register", params={"node_id": "n", "role": "r"}).status_code == 401
    assert c.post("/register", params={"node_id": "n", "role": "r"}, headers=bearer("client-t")).status_code == 401
    assert c.post("/register", params={"node_id": "n", "role": "r"}, headers=bearer("worker-t")).status_code == 200
    body = pack_payload({})
    assert c.post("/submit", content=body, params={"target_role": "r", "sender_id": "c"}, headers=bearer("worker-t")).status_code == 401
    ok = c.post("/submit", content=body, params={"target_role": "r", "sender_id": "c"}, headers=bearer("client-t"))
    assert ok.status_code == 200
    assert c.get("/get_result/" + ok.json()["task_id"], headers=bearer("worker-t")).status_code == 401
    assert c.post("/submit", content=body, params={"target_role": "r", "sender_id": "c"},
                  headers={"Authorization": "client-t"}).status_code == 401


def test_simulated_latency(relay_factory):
    c = relay_factory(worker_rtt_ms=200)
    started = time.perf_counter()
    c.post("/register", params={"node_id": "n", "role": "r"})
    # 200 ms simulated RTT = two 100 ms sleeps. On Windows asyncio timers run on a ~15.6 ms
    # clock and may fire up to one tick early, so allow 50 ms (without latency it is ~1 ms).
    assert time.perf_counter() - started >= 0.15
    started = time.perf_counter()
    submit(c, "r")
    assert time.perf_counter() - started < 0.15


def test_token_rotation(relay_factory, tmp_path):
    bearer = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731

    def status(c, token):
        return c.post("/register", params={"node_id": "n", "role": "r"}, headers=bearer(token)).status_code

    # Several tokens at once: both the new and the old one work during the switch.
    c = relay_factory(token="new-t,old-t")
    assert (status(c, "new-t"), status(c, "old-t"), status(c, "other")) == (200, 200, 401)

    # A token file is re-read when it changes: rotation without restarting the relay.
    tokens = tmp_path / "tokens"
    tokens.write_text("# clients and workers\nfirst-t\n")
    c = relay_factory(token=f"@{tokens}")
    assert (status(c, "first-t"), status(c, "second-t")) == (200, 401)
    tokens.write_text("second-t\n")
    os.utime(tokens, (time.time() + 5, time.time() + 5))  # a visibly newer mtime
    time.sleep(1.3)  # the relay checks the file at most once per second
    assert (status(c, "first-t"), status(c, "second-t")) == (401, 200)
