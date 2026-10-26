import time
import asyncio
import os
import secrets
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import msgpack
from .transport import pack_payload, unpack_payload

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the janitor
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
    side = _side(request.url.path)
    rtt = simulated_rtt[side]
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
auth_tokens: Dict[str, Optional[str]] = {"client": None, "worker": None}


def set_auth_tokens(token: Optional[str] = None, worker_token: Optional[str] = None) -> None:
    """``token`` protects client endpoints; ``worker_token`` (default: ``token``) worker ones."""
    auth_tokens["client"] = token or None
    auth_tokens["worker"] = worker_token or token or None


@app.middleware("http")
async def require_token(request: Request, call_next):
    expected = auth_tokens[_side(request.url.path)]
    if expected is None:
        return await call_next(request)
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(value.strip().encode(), expected.encode()):
        return JSONResponse({"detail": "Invalid or missing token"}, status_code=401)
    return await call_next(request)

# --- Task State Management ---
class Task:
    def __init__(
        self,
        task_id: str,
        sender_id: str,
        target_role: str,
        payload: bytes,
        route: Optional[List[str]] = None,
        root_id: Optional[str] = None,
        session: Optional[str] = None,
    ):
        self.task_id = task_id
        self.sender_id = sender_id
        self.target_role = target_role
        self.payload = payload
        # Roles that receive this task's result next (chained pipeline stages).
        self.route = list(route or [])
        # The task the sender waits on; the final result of the chain lands there.
        self.root_id = root_id or task_id
        # Tasks of one session go to the same worker of a role (it holds the KV cache).
        self.session = session or None
        self.status = "PENDING" # PENDING, ASSIGNED, FORWARDED, COMPLETED, FAILED
        self.worker_id = None
        self.assigned_at = None
        self.result = None
        self.created_at = time.time()
        self.completed_at = None
        # Set when the task needs no more work: COMPLETED/FAILED, or FORWARDED for an
        # intermediate step of a chain. A chain's root is set only when the chain ends.
        # Long-polling clients wait on it; cleanup uses it to tell live tasks from dead ones.
        self.done = asyncio.Event()

# --- Registry State ---
class Node:
    def __init__(self, node_id: str, role: str):
        self.node_id = node_id
        self.role = role
        self.last_heartbeat = time.time()

nodes: Dict[str, Node] = {}
tasks: Dict[str, Task] = {}
pending_queues: Dict[str, asyncio.Queue] = {} # role -> Queue
# Session affinity: several workers may serve one role (replicas), but a session's KV
# cache lives on one of them. (role, session) -> (worker_id, last used); tasks of a bound
# session go to that worker's own queue while it is alive.
worker_queues: Dict[str, asyncio.Queue] = {} # worker_id -> Queue
affinity: Dict[tuple, tuple] = {}
SESSION_AFFINITY_TTL = 900.0

TASK_TIMEOUT = 30.0 # Seconds before a task is returned to queue
NODE_TIMEOUT = 30.0 # Seconds before a node is considered dead (3x heartbeat interval)
JANITOR_INTERVAL = 5.0 # Seconds between janitor sweeps


def set_timeouts(task_timeout: float = 30.0, node_timeout: float = 30.0) -> None:
    """How fast the relay gives up on a silent task / node (lower = faster failover)."""
    global TASK_TIMEOUT, NODE_TIMEOUT, JANITOR_INTERVAL
    TASK_TIMEOUT = max(0.1, task_timeout)
    NODE_TIMEOUT = max(0.1, node_timeout)
    JANITOR_INTERVAL = min(5.0, TASK_TIMEOUT / 3, NODE_TIMEOUT / 3)
MAX_RESULT_WAIT = 60.0 # Upper bound for long-poll in /get_result
FINISHED = ("COMPLETED", "FAILED", "FORWARDED")
# A worker result containing this key ends a chain early (e.g. a stage cache miss).
# Workers flag it with ?stop_chain=1; small bodies are also inspected for compatibility.
STOP_CHAIN_KEY = "stop_chain"
STOP_CHAIN_SNIFF_LIMIT = 4096  # bytes; hidden-state payloads are never decoded by the relay
MAX_ERROR_LENGTH = 8000

# --- Background Tasks ---
def _role_queue(role: str) -> asyncio.Queue:
    if role not in pending_queues:
        pending_queues[role] = asyncio.Queue()
    return pending_queues[role]


def _worker_queue(worker_id: str) -> asyncio.Queue:
    if worker_id not in worker_queues:
        worker_queues[worker_id] = asyncio.Queue()
    return worker_queues[worker_id]


def _bound_worker(task: Task) -> Optional[str]:
    """The live worker holding this task's session for its role, if any."""
    if not task.session:
        return None
    entry = affinity.get((task.target_role, task.session))
    if entry is None:
        return None
    worker_id = entry[0]
    node = nodes.get(worker_id)
    if node is None or node.role != task.target_role:
        affinity.pop((task.target_role, task.session), None)
        return None
    return worker_id


async def _enqueue(task: Task) -> None:
    tasks[task.task_id] = task
    worker_id = _bound_worker(task)
    queue = _worker_queue(worker_id) if worker_id else _role_queue(task.target_role)
    await queue.put(task.task_id)


async def _requeue(task: Task, reason: str) -> None:
    """Return an assigned task to its role queue (any replica may take it)."""
    print(f"Task {task.task_id} {reason}. Re-queuing...")
    if task.session:
        # The bound worker is dead or too slow: let another replica take the session.
        entry = affinity.get((task.target_role, task.session))
        if entry and entry[0] == task.worker_id:
            del affinity[(task.target_role, task.session)]
    task.status = "PENDING"
    task.worker_id = None
    task.assigned_at = None
    await _role_queue(task.target_role).put(task.task_id)


async def _forget_worker(worker_id: str) -> None:
    """Re-queue everything a gone worker held or had queued, drop its sessions."""
    for task in list(tasks.values()):
        if task.status == "ASSIGNED" and task.worker_id == worker_id:
            await _requeue(task, f"was assigned to gone node {worker_id}")
    queue = worker_queues.pop(worker_id, None)
    while queue is not None and not queue.empty():
        task = tasks.get(queue.get_nowait())
        if task is not None and task.status == "PENDING":
            await _role_queue(task.target_role).put(task.task_id)
    for key in [k for k, (w, _) in affinity.items() if w == worker_id]:
        del affinity[key]


async def cleanup_stuck_tasks():
    """Recover stuck tasks."""
    now = time.time()
    for tid, task in list(tasks.items()):
        if task.status == "ASSIGNED" and task.assigned_at and (now - task.assigned_at > TASK_TIMEOUT):
            await _requeue(task, f"timed out on worker {task.worker_id}")

def cleanup_old_tasks():
    """Remove old tasks to prevent memory leaks."""
    now = time.time()
    to_delete = []
    for tid, task in list(tasks.items()):
        # Clean up finished tasks after 5 minutes. A chain root stays FORWARDED (not done)
        # while later stages run — it must survive however long they take.
        if task.done.is_set() and task.completed_at and (now - task.completed_at > 300.0):
            to_delete.append(tid)
        # Clean up stuck/dangling tasks older than 30 minutes
        elif not task.done.is_set() and (now - task.created_at > 1800.0):
            to_delete.append(tid)
            
    for tid in to_delete:
        try:
            del tasks[tid]
        except KeyError:
            pass
    for key in [k for k, (_, used) in affinity.items() if now - used > SESSION_AFFINITY_TTL]:
        del affinity[key]

async def relay_janitor():
    """Cleans up dead nodes, recovers stuck tasks, and prevents memory leaks."""
    while True:
        now = time.time()
        
        # 1. Cleanup Dead Nodes
        dead_node_ids = [nid for nid, node in list(nodes.items()) if now - node.last_heartbeat > NODE_TIMEOUT]
        for nid in dead_node_ids:
            if nid in nodes:
                print(f"Node {nid} ({nodes[nid].role}) timed out.")
                del nodes[nid]
                await _forget_worker(nid)

        # 2. Recover Stuck Tasks
        await cleanup_stuck_tasks()
        
        # 3. Clean up old tasks (prevent memory leak)
        cleanup_old_tasks()
        
        await asyncio.sleep(JANITOR_INTERVAL)

# --- API Endpoints ---

@app.post("/register")
async def register(node_id: str, role: str):
    nodes[node_id] = Node(node_id, role)
    _role_queue(role)
    _worker_queue(node_id)
    return {"status": "ok"}

@app.post("/deregister")
async def deregister(node_id: str):
    if node_id in nodes:
        print(f"Node {node_id} ({nodes[node_id].role}) deregistered gracefully.")
        del nodes[node_id]
        await _forget_worker(node_id)
    return {"status": "ok"}

@app.post("/heartbeat")
async def heartbeat(node_id: str):
    if node_id in nodes:
        nodes[node_id].last_heartbeat = time.time()
        return {"status": "ok"}
    raise HTTPException(status_code=404)


@app.post("/submit")
async def submit_task(request: Request, target_role: str, sender_id: str, route: str = "", session: str = ""):
    """Submits a task to be processed by a specific role.

    ``route`` (comma-separated roles) chains the task: the result of each step is
    sent by the relay straight to the next role, and the sender gets only the final
    result — one client round trip per chain instead of one per stage.
    ``session`` pins the tasks of one session to the worker of each role that first
    took it (the one holding its KV cache), as long as that worker is alive.
    """
    body = await request.body()
    task_id = str(uuid.uuid4())
    next_roles = [r for r in (x.strip() for x in route.split(",")) if r]
    await _enqueue(Task(task_id, sender_id, target_role, body, route=next_roles, session=session))
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
    if cancelled:
        for queue, item in got:
            queue.put_nowait(item)
        raise asyncio.CancelledError
    if not got:
        return None
    for queue, item in got[1:]:
        queue.put_nowait(item)
    return got[0][1]


async def _queue_get(queue: asyncio.Queue, timeout: float) -> Optional[str]:
    return await _queue_get_first([queue], timeout)


def _take(task_id: Optional[str], worker_id: str) -> Optional[Task]:
    """Assign a dequeued task to the worker (None if it is gone or already taken)."""
    task = tasks.get(task_id) if task_id else None
    if task is None or task.status != "PENDING":
        return None
    task.status = "ASSIGNED"
    task.worker_id = worker_id
    task.assigned_at = time.time()
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
    own_queue = _worker_queue(worker_id)
    role_queue = _role_queue(role)

    # Wait up to 5s for a task (long polling): this worker's sessions first.
    first = _take(await _queue_get_first([own_queue, role_queue], timeout=5.0), worker_id)
    if first is None:
        return Response(status_code=204)
    taken = [first]
    while len(taken) < max(1, max_tasks):
        queue = own_queue if not own_queue.empty() else role_queue
        if queue.empty():
            break
        task = _take(queue.get_nowait(), worker_id)
        if task is not None:
            taken.append(task)

    # Payloads are already msgpack bytes from submit: wrap them without re-encoding.
    if max_tasks <= 1:
        body = {"task_id": first.task_id, "payload": first.payload}
    else:
        body = {"tasks": [{"task_id": t.task_id, "payload": t.payload} for t in taken]}
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
        if target is None or target.done.is_set():
            continue
        target.status = status
        target.result = result
        target.completed_at = now
        target.done.set()


@app.post("/result/{task_id}")
async def post_result(
    task_id: str, request: Request, worker_id: Optional[str] = None, stop_chain: bool = False
):
    """Workers submit the result of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    body = await request.body()
    # Задача могла быть переотдана другому воркеру после таймаута: первый результат побеждает.
    if task.status in FINISHED:
        return {"status": "ignored"}
    _check_worker(task, worker_id)

    if task.route and not _stops_chain(body, stop_chain):
        task.status = "FORWARDED"
        task.completed_at = time.time()
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
async def post_failure(
    task_id: str, request: Request, error_msg: Optional[str] = None, worker_id: Optional[str] = None
):
    """Workers report a failed task execution.

    The message goes in the request body (plain text); ``?error_msg=`` is still
    accepted, but long messages (tracebacks) do not fit into a URL.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    if task.status in FINISHED:
        return {"status": "ignored"}
    _check_worker(task, worker_id)
    body = (await request.body()).decode("utf-8", errors="replace")
    message = (body or error_msg or "unknown error")[:MAX_ERROR_LENGTH]
    _finish(task, "FAILED", message.encode("utf-8"))
    return {"status": "ok"}

@app.get("/get_result/{task_id}")
async def get_result(task_id: str, wait: float = 0.0):
    """Senders call this to fetch the result of a task.

    With ``wait > 0`` the request is held (long-poll) until the task finishes or
    ``wait`` seconds pass, so the result is returned as soon as it is ready.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404)

    task = tasks[task_id]
    if wait > 0 and not task.done.is_set():
        try:
            await asyncio.wait_for(task.done.wait(), timeout=min(wait, MAX_RESULT_WAIT))
        except asyncio.TimeoutError:
            pass
        if task_id not in tasks:
            raise HTTPException(status_code=404)
    if task.status == "COMPLETED":
        return Response(content=task.result, media_type="application/octet-stream")
    if task.status == "FAILED":
        error_msg = task.result.decode("utf-8") if isinstance(task.result, bytes) else str(task.result)
        return {"status": "FAILED", "error": error_msg}
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
