import asyncio
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .auth import TokenSet
from .transport import pack_payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(relay_janitor())
    yield


app = FastAPI(title="distllm Relay Server", lifespan=lifespan)

# --- Network latency simulation (for local experiments) ---
# Every HTTP request to the relay costs one round trip on a real network. These
# settings add that round trip artificially: half before handling (request in
# flight), half after (response in flight). Workers and clients usually sit at
# different distances from the relay (e.g. Colab GPUs vs. a laptop), hence two knobs.
WORKER_PATHS = ("/register", "/deregister", "/heartbeat", "/poll", "/result", "/fail")
simulated_rtt = {"worker": 0.0, "client": 0.0}  # seconds


def set_simulated_latency(worker_rtt_ms: float = 0.0, client_rtt_ms: float = 0.0) -> None:
    simulated_rtt["worker"] = max(0.0, worker_rtt_ms) / 1000.0
    simulated_rtt["client"] = max(0.0, client_rtt_ms) / 1000.0


def _side(path: str) -> str:
    return "worker" if path.startswith(WORKER_PATHS) else "client"


@app.middleware("http")
async def simulate_network_latency(request: Request, call_next):
    rtt = simulated_rtt[_side(request.url.path)]
    if rtt <= 0:
        return await call_next(request)
    await asyncio.sleep(rtt / 2)
    response = await call_next(request)
    await asyncio.sleep(rtt / 2)
    return response


# --- Authentication ---
# Shared-secret bearer tokens. Workers and clients may get different tokens: a Colab
# notebook running a worker then cannot submit tasks, and a client cannot pose as a
# stage and read other users' hidden states. No token configured = open relay.
# Each side takes a token spec (see auth.py): one token, "new,old" or "@file".
auth_tokens: Dict[str, TokenSet] = {"client": TokenSet(None), "worker": TokenSet(None)}


def set_auth_tokens(token: Optional[str] = None, worker_token: Optional[str] = None) -> None:
    """``token`` protects client endpoints; ``worker_token`` (default: ``token``) worker ones."""
    auth_tokens["client"] = TokenSet(token)
    auth_tokens["worker"] = TokenSet(worker_token or token)


@app.middleware("http")
async def require_token(request: Request, call_next):
    expected = auth_tokens[_side(request.url.path)]
    if expected.enabled:
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not expected.accepts(value.strip()):
            return JSONResponse({"detail": "Invalid or missing token"}, status_code=401)
    return await call_next(request)


# --- State ---
@dataclass(eq=False)
class Task:
    task_id: str
    sender_id: str
    target_role: str
    payload: bytes
    # Roles that receive this task's result next (chained pipeline stages).
    route: List[str] = field(default_factory=list)
    # The task the sender waits on; the final result of the chain lands there.
    root_id: Optional[str] = None
    # Tasks of one session go to the same worker of a role (it holds the KV cache).
    session: Optional[str] = None
    status: str = "PENDING"  # PENDING, ASSIGNED, FORWARDED, COMPLETED, FAILED
    worker_id: Optional[str] = None
    assigned_at: Optional[float] = None
    result: Optional[bytes] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    # Set when the task needs no more work: COMPLETED/FAILED, or FORWARDED for an
    # intermediate step of a chain. A chain's root is set only when the chain ends.
    # Long-polling clients wait on it; cleanup uses it to tell live tasks from dead ones.
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self):
        self.root_id = self.root_id or self.task_id
        self.session = self.session or None


@dataclass
class Node:
    node_id: str
    role: str
    last_heartbeat: float = field(default_factory=time.time)


nodes: Dict[str, Node] = {}
tasks: Dict[str, Task] = {}
pending_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)  # role -> task ids
worker_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)  # worker_id -> task ids
# Session affinity: several workers may serve one role (replicas), but a session's KV
# cache lives on one of them. (role, session) -> (worker_id, last used); tasks of a bound
# session go to that worker's own queue while it is alive.
affinity: Dict[tuple, tuple] = {}
SESSION_AFFINITY_TTL = 900.0
MAX_REPORTED_SESSIONS = 256  # sessions a worker may claim when it (re-)registers

TASK_TIMEOUT = 30.0  # seconds before an assigned task without a result is re-queued
NODE_TIMEOUT = 30.0  # seconds before a silent node is considered dead (3x heartbeat interval)
JANITOR_INTERVAL = 5.0  # seconds between janitor sweeps
FINISHED_TTL = 300.0  # finished tasks are kept this long for their senders
DANGLING_TTL = 1800.0  # unfinished tasks older than this are dropped


def set_timeouts(task_timeout: float = 30.0, node_timeout: float = 30.0) -> None:
    """How fast the relay gives up on a silent task / node (lower = faster failover)."""
    global TASK_TIMEOUT, NODE_TIMEOUT, JANITOR_INTERVAL
    TASK_TIMEOUT = max(0.1, task_timeout)
    NODE_TIMEOUT = max(0.1, node_timeout)
    JANITOR_INTERVAL = min(5.0, TASK_TIMEOUT / 3, NODE_TIMEOUT / 3)


MAX_RESULT_WAIT = 60.0  # upper bound for long-poll in /get_result
FINISHED = ("COMPLETED", "FAILED", "FORWARDED")
# A worker result containing this key ends a chain early (e.g. a stage cache miss).
# Workers flag it with ?stop_chain=1; small bodies are also inspected for compatibility.
STOP_CHAIN_KEY = "stop_chain"
STOP_CHAIN_SNIFF_LIMIT = 4096  # bytes; hidden-state payloads are never decoded by the relay
MAX_ERROR_LENGTH = 8000


def _bound_worker(task: Task) -> Optional[str]:
    """The live worker holding this task's session for its role, if any."""
    key = (task.target_role, task.session)
    worker_id = affinity.get(key, (None,))[0] if task.session else None
    node = nodes.get(worker_id) if worker_id else None
    if worker_id and (node is None or node.role != task.target_role):
        del affinity[key]
        return None
    return worker_id


async def _enqueue(task: Task) -> None:
    tasks[task.task_id] = task
    worker_id = _bound_worker(task)
    await (worker_queues[worker_id] if worker_id else pending_queues[task.target_role]).put(task.task_id)


async def _requeue(task: Task, reason: str) -> None:
    """Return an assigned task to its role queue (any replica may take it)."""
    print(f"Task {task.task_id} {reason}. Re-queuing...")
    key = (task.target_role, task.session)
    if task.session and affinity.get(key, (None,))[0] == task.worker_id:
        del affinity[key]  # the bound worker is dead or too slow: let another replica take it
    task.status, task.worker_id, task.assigned_at = "PENDING", None, None
    await pending_queues[task.target_role].put(task.task_id)


async def _forget_worker(worker_id: str) -> None:
    """Re-queue everything a gone worker held or had queued, drop its sessions."""
    for task in list(tasks.values()):
        if task.status == "ASSIGNED" and task.worker_id == worker_id:
            await _requeue(task, f"was assigned to gone node {worker_id}")
    queue = worker_queues.pop(worker_id, None)
    while queue is not None and not queue.empty():
        task = tasks.get(queue.get_nowait())
        if task is not None and task.status == "PENDING":
            await pending_queues[task.target_role].put(task.task_id)
    for key in [k for k, (w, _) in affinity.items() if w == worker_id]:
        del affinity[key]


async def cleanup_stuck_tasks():
    """Re-queue assigned tasks whose worker did not answer within TASK_TIMEOUT."""
    now = time.time()
    for task in list(tasks.values()):
        if task.status == "ASSIGNED" and task.assigned_at and now - task.assigned_at > TASK_TIMEOUT:
            await _requeue(task, f"timed out on worker {task.worker_id}")


def cleanup_old_tasks():
    """Forget finished and dangling tasks and stale session affinity (no memory leaks).

    A chain root stays FORWARDED (not done) while later stages run — it must survive
    however long they take, up to DANGLING_TTL.
    """
    now = time.time()
    for task_id, task in list(tasks.items()):
        finished = task.done.is_set() and task.completed_at and now - task.completed_at > FINISHED_TTL
        if finished or (not task.done.is_set() and now - task.created_at > DANGLING_TTL):
            del tasks[task_id]
    for key in [k for k, (_, used) in affinity.items() if now - used > SESSION_AFFINITY_TTL]:
        del affinity[key]


async def _drop_node(node_id: str, why: str) -> None:
    node = nodes.pop(node_id, None)
    if node is not None:
        print(f"Node {node_id} ({node.role}) {why}.")
        await _forget_worker(node_id)


async def relay_janitor():
    """Drops dead nodes, recovers stuck tasks, forgets old ones."""
    while True:
        now = time.time()
        for node_id in [n for n, node in nodes.items() if now - node.last_heartbeat > NODE_TIMEOUT]:
            await _drop_node(node_id, "timed out")
        await cleanup_stuck_tasks()
        cleanup_old_tasks()
        await asyncio.sleep(JANITOR_INTERVAL)


def _task(task_id: str) -> Task:
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    return tasks[task_id]


# --- API Endpoints ---

@app.post("/register")
async def register(node_id: str, role: str, sessions: str = ""):
    """``sessions``: comma-separated sessions whose KV cache the worker holds. After a relay
    restart the affinity map is empty; workers re-register with their sessions so that the
    next steps go back to them instead of being replayed on another replica."""
    nodes[node_id] = Node(node_id, role)
    now = time.time()
    for session in [s for s in sessions.split(",") if s][:MAX_REPORTED_SESSIONS]:
        entry = affinity.get((role, session))
        if entry is None or entry[0] not in nodes:  # never steal a session from a live replica
            affinity[(role, session)] = (node_id, now)
    return {"status": "ok"}


@app.post("/deregister")
async def deregister(node_id: str):
    await _drop_node(node_id, "deregistered gracefully")
    return {"status": "ok"}


@app.post("/heartbeat")
async def heartbeat(node_id: str):
    if node_id not in nodes:
        raise HTTPException(status_code=404)
    nodes[node_id].last_heartbeat = time.time()
    return {"status": "ok"}


@app.post("/submit")
async def submit_task(request: Request, target_role: str, sender_id: str, route: str = "", session: str = ""):
    """Submits a task to be processed by a specific role.

    ``route`` (comma-separated roles) chains the task: the result of each step is
    sent by the relay straight to the next role, and the sender gets only the final
    result — one client round trip per chain instead of one per stage.
    ``session`` pins the tasks of one session to the worker of each role that first
    took it (the one holding its KV cache), as long as that worker is alive.
    """
    task_id = str(uuid.uuid4())
    next_roles = [r for r in (x.strip() for x in route.split(",")) if r]
    await _enqueue(Task(task_id, sender_id, target_role, await request.body(), route=next_roles, session=session))
    return {"task_id": task_id}


async def _queue_get_first(queues: List[asyncio.Queue], timeout: float) -> Optional[str]:
    """Get an item from the first of ``queues`` that has one, waiting up to ``timeout``.

    Never loses an item: asyncio.wait_for(queue.get(), t) may discard one dequeued right
    as the timeout fires (Python < 3.12); items taken by getters that lost the race (or
    when the request itself is cancelled) are put back.
    """
    for queue in queues:
        if not queue.empty():
            return queue.get_nowait()
    getters = [asyncio.ensure_future(queue.get()) for queue in queues]
    cancelled = False
    try:
        await asyncio.wait(getters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        cancelled = True
    for getter in getters:
        getter.cancel()
    await asyncio.gather(*getters, return_exceptions=True)
    got = [
        (queue, getter.result())
        for queue, getter in zip(queues, getters)
        if getter.done() and not getter.cancelled() and getter.exception() is None
    ]
    for queue, item in got[0 if cancelled else 1:]:
        queue.put_nowait(item)
    if cancelled:
        raise asyncio.CancelledError
    return got[0][1] if got else None


def _take(task_id: Optional[str], worker_id: str) -> Optional[Task]:
    """Assign a dequeued task to the worker (None if it is gone or already taken)."""
    task = tasks.get(task_id) if task_id else None
    if task is None or task.status != "PENDING":
        return None
    task.status, task.worker_id, task.assigned_at = "ASSIGNED", worker_id, time.time()
    if task.session:
        affinity[(task.target_role, task.session)] = (worker_id, task.assigned_at)
    return task


@app.get("/poll/{role}")
async def poll_task(role: str, worker_id: str, max_tasks: int = 1):
    """Workers call this to fetch pending tasks.

    Waits up to 5 s for the first task (this worker's sessions first, then the role
    queue). With ``max_tasks > 1`` it also takes whatever else is already queued, up to
    that many, so a batching worker can run them in one forward; the response is then
    ``{"tasks": [{"task_id", "payload"}, ...]}`` instead of a single task.
    """
    own, shared = worker_queues[worker_id], pending_queues[role]
    first = _take(await _queue_get_first([own, shared], timeout=5.0), worker_id)
    if first is None:
        return Response(status_code=204)
    taken = [first]
    while len(taken) < max_tasks and not (own.empty() and shared.empty()):
        task = _take((own if not own.empty() else shared).get_nowait(), worker_id)
        if task is not None:
            taken.append(task)
    # Payloads are already msgpack bytes from submit: wrap them without re-encoding.
    items = [{"task_id": t.task_id, "payload": t.payload} for t in taken]
    body = items[0] if max_tasks <= 1 else {"tasks": items}
    return Response(content=pack_payload(body), media_type="application/octet-stream")


def _stops_chain(body: bytes, flagged: bool) -> bool:
    if flagged:
        return True
    # Workers flag it explicitly; sniff only tiny bodies (a cache-miss answer is a few
    # bytes) so that hidden states are never decoded on the relay's event loop.
    if len(body) > STOP_CHAIN_SNIFF_LIMIT:
        return False
    try:
        result = msgpack.unpackb(body, raw=False)
    except Exception:  # noqa: BLE001 — not a msgpack dict: just forward it
        return False
    return isinstance(result, dict) and bool(result.get(STOP_CHAIN_KEY))


def _check_worker(task: Task, worker_id: Optional[str]) -> None:
    """Only the worker a task is assigned to may complete it.

    An assigned task requires ?worker_id= to match — omitting it would otherwise be a
    way around the check. A task that is not assigned (never polled, or re-queued after
    a timeout) accepts any result: first result wins.
    """
    if task.worker_id is not None and worker_id != task.worker_id:
        raise HTTPException(status_code=409, detail="task is assigned to another worker")


def _finish(task: Task, status: str, result: bytes) -> None:
    """Complete a task and, for the last step of a chain, its root task too."""
    now = time.time()
    for target in {task.task_id: task, task.root_id: tasks.get(task.root_id)}.values():
        if target is not None and not target.done.is_set():
            target.status, target.result, target.completed_at = status, result, now
            target.done.set()


@app.post("/result/{task_id}")
async def post_result(task_id: str, request: Request, worker_id: Optional[str] = None, stop_chain: bool = False):
    """Workers submit the result of a task."""
    task = _task(task_id)
    body = await request.body()
    if task.status in FINISHED:  # re-assigned after a timeout: the first result wins
        return {"status": "ignored"}
    _check_worker(task, worker_id)
    if task.route and not _stops_chain(body, stop_chain):
        task.status, task.completed_at = "FORWARDED", time.time()
        if task.task_id != task.root_id:
            task.done.set()  # nobody waits on an intermediate step
        await _enqueue(Task(
            str(uuid.uuid4()), task.sender_id, task.route[0], body,
            route=task.route[1:], root_id=task.root_id, session=task.session,
        ))
        return {"status": "forwarded"}
    _finish(task, "COMPLETED", body)
    return {"status": "ok"}


@app.post("/fail/{task_id}")
async def post_failure(task_id: str, request: Request, error_msg: Optional[str] = None, worker_id: Optional[str] = None):
    """Workers report a failed task execution.

    The message goes in the request body (plain text); ``?error_msg=`` is still
    accepted, but long messages (tracebacks) do not fit into a URL.
    """
    task = _task(task_id)
    if task.status in FINISHED:
        return {"status": "ignored"}
    _check_worker(task, worker_id)
    body = (await request.body()).decode("utf-8", errors="replace")
    _finish(task, "FAILED", (body or error_msg or "unknown error")[:MAX_ERROR_LENGTH].encode())
    return {"status": "ok"}


@app.get("/get_result/{task_id}")
async def get_result(task_id: str, wait: float = 0.0):
    """Senders call this to fetch the result of a task.

    With ``wait > 0`` the request is held (long-poll) until the task finishes or
    ``wait`` seconds pass, so the result is returned as soon as it is ready.
    """
    task = _task(task_id)
    if wait > 0 and not task.done.is_set():
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task.done.wait(), timeout=min(wait, MAX_RESULT_WAIT))
        task = _task(task_id)  # it may have been cleaned up meanwhile
    if task.status == "COMPLETED":
        return Response(content=task.result, media_type="application/octet-stream")
    if task.status == "FAILED":
        return {"status": "FAILED", "error": task.result.decode("utf-8")}
    return {"status": task.status}


def start_relay(
    port: int = 8000,
    worker_rtt_ms: float = 0.0,
    client_rtt_ms: float = 0.0,
    host: str = "0.0.0.0",
    token: Optional[str] = None,
    worker_token: Optional[str] = None,
    task_timeout: float = 30.0,
    node_timeout: float = 30.0,
):
    """Run the relay. Tokens default to $DISTLLM_TOKEN / $DISTLLM_WORKER_TOKEN.

    ``node_timeout``: a worker silent for that long (no heartbeat) is considered dead and
    its tasks are re-queued; ``task_timeout``: an assigned task without a result for that
    long is re-queued. Keep node_timeout above ~3 worker heartbeat intervals.
    """
    set_timeouts(task_timeout, node_timeout)
    token = token or os.environ.get("DISTLLM_TOKEN")
    worker_token = worker_token or os.environ.get("DISTLLM_WORKER_TOKEN")
    set_auth_tokens(token, worker_token)
    if not (token or worker_token) and host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "[!] WARNING: relay is reachable from the network without a token: anyone who "
            "knows the URL can submit tasks or register as a worker. Set DISTLLM_TOKEN."
        )
    set_simulated_latency(worker_rtt_ms, client_rtt_ms)
    uvicorn.run(app, host=host, port=port)
