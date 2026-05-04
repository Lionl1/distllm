import time
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import uvicorn
from .transport import pack_payload, unpack_payload

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the janitor
    asyncio.create_task(relay_janitor())
    yield

app = FastAPI(title="distllm Relay Server", lifespan=lifespan)

# --- Task State Management ---
class Task:
    def __init__(self, task_id: str, sender_id: str, target_role: str, payload: bytes):
        self.task_id = task_id
        self.sender_id = sender_id
        self.target_role = target_role
        self.payload = payload
        self.status = "PENDING" # PENDING, ASSIGNED, COMPLETED
        self.worker_id = None
        self.assigned_at = None
        self.result = None
        self.created_at = time.time()

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
NODE_TIMEOUT = 15.0 # Seconds before a node is considered dead

# --- Background Tasks ---
async def cleanup_stuck_tasks():
    """Recover stuck tasks."""
    now = time.time()
    for tid, task in tasks.items():
        if task.status == "ASSIGNED" and task.assigned_at and (now - task.assigned_at > TASK_TIMEOUT):
            print(f"Task {tid} timed out on worker {task.worker_id}. Re-queuing...")
            task.status = "PENDING"
            task.worker_id = None
            task.assigned_at = None
            await pending_queues[task.target_role].put(tid)

async def relay_janitor():
    """Cleans up dead nodes and recovers stuck tasks."""
    while True:
        now = time.time()
        
        # 1. Cleanup Dead Nodes
        dead_node_ids = [nid for nid, node in nodes.items() if now - node.last_heartbeat > NODE_TIMEOUT]
        for nid in dead_node_ids:
            print(f"Node {nid} ({nodes[nid].role}) timed out.")
            del nodes[nid]

        # 2. Recover Stuck Tasks
        await cleanup_stuck_tasks()
        
        await asyncio.sleep(5)

# --- API Endpoints ---

@app.post("/register")
async def register(node_id: str, role: str):
    nodes[node_id] = Node(node_id, role)
    if role not in pending_queues:
        pending_queues[role] = asyncio.Queue()
    return {"status": "ok"}

@app.post("/heartbeat")
async def heartbeat(node_id: str):
    if node_id in nodes:
        nodes[node_id].last_heartbeat = time.time()
        return {"status": "ok"}
    raise HTTPException(status_code=404)

@app.post("/submit")
async def submit_task(request: Request, target_role: str, sender_id: str):
    """Submits a task to be processed by a specific role."""
    body = await request.body()
    task_id = str(uuid.uuid4())
    task = Task(task_id, sender_id, target_role, body)
    tasks[task_id] = task
    
    if target_role not in pending_queues:
        pending_queues[target_role] = asyncio.Queue()
    
    await pending_queues[target_role].put(task_id)
    return {"task_id": task_id}

@app.get("/poll/{role}")
async def poll_task(role: str, worker_id: str):
    """Workers call this to fetch a pending task."""
    if role not in pending_queues:
        return Response(status_code=204)
    
    try:
        # Wait up to 5s for a task (long polling)
        task_id = await asyncio.wait_for(pending_queues[role].get(), timeout=5.0)
        task = tasks[task_id]
        
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

@app.post("/result/{task_id}")
async def post_result(task_id: str, request: Request):
    """Workers submit the result of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    task.result = await request.body()
    task.status = "COMPLETED"
    return {"status": "ok"}

@app.get("/get_result/{task_id}")
async def get_result(task_id: str):
    """Senders call this to fetch the result of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404)
    
    task = tasks[task_id]
    if task.status == "COMPLETED":
        return Response(content=task.result, media_type="application/octet-stream")
    return {"status": task.status}

def start_relay(port: int = 8000):
    uvicorn.run(app, host="0.0.0.0", port=port)
