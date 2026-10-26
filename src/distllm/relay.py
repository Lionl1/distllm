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
    ):
        self.task_id = task_id
        self.sender_id = sender_id
        self.target_role = target_role
        self.payload = payload
        # Roles that receive this task's result next (chained pipeline stages).
        self.route = list(route or [])
        # The task the sender waits on; the final result of the chain lands there.
        self.root_id = root_id or task_id
        self.status = "PENDING" # PENDING, ASSIGNED, FORWARDED, COMPLETED, FAILED
        self.worker_id = None
        self.assigned_at = None
        self.result = None
        self.created_at = time.time()
        self.completed_at = None
        # Set when the task reaches COMPLETED/FAILED; long-polling clients wait on it.
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

TASK_TIMEOUT = 30.0 # Seconds before a task is returned to queue
NODE_TIMEOUT = 30.0 # Seconds before a node is considered dead (3x heartbeat interval)
MAX_RESULT_WAIT = 60.0 # Upper bound for long-poll in /get_result
FINISHED = ("COMPLETED", "FAILED", "FORWARDED")
# A worker result containing this key ends a chain early (e.g. a stage cache miss).
STOP_CHAIN_KEY = "stop_chain"

# --- Background Tasks ---
async def cleanup_stuck_tasks():
    """Recover stuck tasks."""
    now = time.time()
    for tid, task in list(tasks.items()):
        if task.status == "ASSIGNED" and task.assigned_at and (now - task.assigned_at > TASK_TIMEOUT):
            print(f"Task {tid} timed out on worker {task.worker_id}. Re-queuing...")
            task.status = "PENDING"
            task.worker_id = None
            task.assigned_at = None
            if task.target_role not in pending_queues:
                pending_queues[task.target_role] = asyncio.Queue()
            await pending_queues[task.target_role].put(tid)

def cleanup_old_tasks():
    """Remove old tasks to prevent memory leaks."""
    now = time.time()
    to_delete = []
    for tid, task in list(tasks.items()):
        # Clean up completed or failed tasks after 5 minutes
        if task.status in FINISHED and task.completed_at and (now - task.completed_at > 300.0):
            to_delete.append(tid)
        # Clean up stuck/dangling tasks older than 30 minutes
        elif task.status not in FINISHED and (now - task.created_at > 1800.0):
            to_delete.append(tid)
            
    for tid in to_delete:
        try:
            del tasks[tid]
        except KeyError:
            pass

async def relay_janitor():
    """Cleans up dead nodes, recovers stuck tasks, and prevents memory leaks."""
    while True:
        now = time.time()
        
        # 1. Cleanup Dead Nodes
        dead_node_ids = [nid for nid, node in list(nodes.items()) if now - node.last_heartbeat > NODE_TIMEOUT]
        for nid in dead_node_ids:
            if nid in nodes:
                print(f"Node {nid} ({nodes[nid].role}) timed out.")
                role = nodes[nid].role
                del nodes[nid]
                # Immediately re-queue tasks assigned to the dead worker
                for tid, task in list(tasks.items()):
                    if task.status == "ASSIGNED" and task.worker_id == nid:
                        print(f"Task {tid} was assigned to dead node {nid}. Re-queuing immediately...")
                        task.status = "PENDING"
                        task.worker_id = None
                        task.assigned_at = None
                        if task.target_role not in pending_queues:
                            pending_queues[task.target_role] = asyncio.Queue()
                        await pending_queues[task.target_role].put(tid)

        # 2. Recover Stuck Tasks
        await cleanup_stuck_tasks()
        
        # 3. Clean up old tasks (prevent memory leak)
        cleanup_old_tasks()
        
        await asyncio.sleep(5)

# --- API Endpoints ---

@app.post("/register")
async def register(node_id: str, role: str):
    nodes[node_id] = Node(node_id, role)
    if role not in pending_queues:
        pending_queues[role] = asyncio.Queue()
    return {"status": "ok"}

@app.post("/deregister")
async def deregister(node_id: str):
    if node_id in nodes:
        print(f"Node {node_id} ({nodes[node_id].role}) deregistered gracefully.")
        # Immediately re-queue tasks assigned to this worker
        for tid, task in list(tasks.items()):
            if task.status == "ASSIGNED" and task.worker_id == node_id:
                print(f"Task {tid} was assigned to deregistered node {node_id}. Re-queuing immediately...")
                task.status = "PENDING"
                task.worker_id = None
                task.assigned_at = None
                if task.target_role not in pending_queues:
                    pending_queues[task.target_role] = asyncio.Queue()
                await pending_queues[task.target_role].put(tid)
        del nodes[node_id]
    return {"status": "ok"}

@app.post("/heartbeat")
async def heartbeat(node_id: str):
    if node_id in nodes:
        nodes[node_id].last_heartbeat = time.time()
        return {"status": "ok"}
    raise HTTPException(status_code=404)

async def _enqueue(task: Task) -> None:
    tasks[task.task_id] = task
    if task.target_role not in pending_queues:
        pending_queues[task.target_role] = asyncio.Queue()
    await pending_queues[task.target_role].put(task.task_id)


@app.post("/submit")
async def submit_task(request: Request, target_role: str, sender_id: str, route: str = ""):
    """Submits a task to be processed by a specific role.

    ``route`` (comma-separated roles) chains the task: the result of each step is
    sent by the relay straight to the next role, and the sender gets only the final
    result — one client round trip per chain instead of one per stage.
    """
    body = await request.body()
    task_id = str(uuid.uuid4())
    next_roles = [r for r in (x.strip() for x in route.split(",")) if r]
    await _enqueue(Task(task_id, sender_id, target_role, body, route=next_roles))
    return {"task_id": task_id}

@app.get("/poll/{role}")
async def poll_task(role: str, worker_id: str):
    """Workers call this to fetch a pending task."""
    # Создаём очередь, если её нет (например, relay перезапустился), и ждём в ней:
    # мгновенный 204 заставлял бы воркер крутиться в цикле запросов.
    if role not in pending_queues:
        pending_queues[role] = asyncio.Queue()

    try:
        # Wait up to 5s for a task (long polling)
        task_id = await asyncio.wait_for(pending_queues[role].get(), timeout=5.0)
        task = tasks.get(task_id)
        if not task:
            return Response(status_code=204)
        
        # Double check task hasn't been completed/assigned already
        if task.status != "PENDING":
             return Response(status_code=204)

        task.status = "ASSIGNED"
        task.worker_id = worker_id
        task.assigned_at = time.time()
        
        # We need to return the task_id AND the payload.
        # Since we want to avoid JSON encoding binary data, 
        # let's pack the whole response into msgpack and return it as bytes.
        response_data = pack_payload({
            "task_id": task.task_id,
            "payload": task.payload # This is already bytes from submit
        })
        return Response(content=response_data, media_type="application/octet-stream")
    except asyncio.TimeoutError:
        return Response(status_code=204)

def _stops_chain(body: bytes) -> bool:
    try:
        result = msgpack.unpackb(body, raw=False)
    except Exception:  # noqa: BLE001 — not a msgpack dict: just forward it
        return False
    return isinstance(result, dict) and bool(result.get(STOP_CHAIN_KEY))


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
async def post_result(task_id: str, request: Request):
    """Workers submit the result of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    body = await request.body()
    # Задача могла быть переотдана другому воркеру после таймаута: первый результат побеждает.
    if task.status in FINISHED:
        return {"status": "ignored"}

    if task.route and not _stops_chain(body):
        task.status = "FORWARDED"
        task.completed_at = time.time()
        await _enqueue(Task(
            str(uuid.uuid4()), task.sender_id, task.route[0], body,
            route=task.route[1:], root_id=task.root_id,
        ))
        return {"status": "forwarded"}

    _finish(task, "COMPLETED", body)
    return {"status": "ok"}

@app.post("/fail/{task_id}")
async def post_failure(task_id: str, error_msg: str):
    """Workers report a failed task execution."""
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    if task.status in FINISHED:
        return {"status": "ignored"}
    _finish(task, "FAILED", error_msg.encode("utf-8"))
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
):
    """Run the relay. Tokens default to $DISTLLM_TOKEN / $DISTLLM_WORKER_TOKEN."""
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
