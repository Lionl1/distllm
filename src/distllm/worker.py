import asyncio
import uuid
import time
import httpx
import functools
from typing import Callable, Optional
from .transport import pack_payload, unpack_payload

class Worker:
    def __init__(self, role: str, relay_url: str, node_id: str = None):
        self.role = role
        self.relay_url = relay_url.rstrip("/")
        self.node_id = node_id or str(uuid.uuid4())[:8]
        # High timeout for unstable tunnels
        self.client = httpx.AsyncClient(timeout=120.0)

    async def register(self):
        """Registers the worker node with the relay, retrying if necessary."""
        while True:
            try:
                print(f"[*] Registering node {self.node_id}...")
                resp = await self.client.post(
                    f"{self.relay_url}/register", 
                    params={"node_id": self.node_id, "role": self.role}
                )
                if resp.status_code == 200:
                    print(f"[v] Registration successful.")
                    return
                else:
                    print(f"[!] Registration failed with status {resp.status_code}. Retrying in 5s...")
            except Exception as e:
                print(f"[!] Registration failed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

    async def run(self, func: Callable):
        """Main loop for the worker node."""
        print(f"[*] Worker {self.node_id} (Role: {self.role})")
        print(f"[*] Connecting to relay: {self.relay_url}")
        
        # 1. Register with Retries
        await self.register()
        
        # 2. Heartbeat task
        async def heartbeat():
            while True:
                try:
                    resp = await self.client.post(f"{self.relay_url}/heartbeat", params={"node_id": self.node_id})
                    if resp.status_code == 404:
                        print(f"[!] Node not registered on relay (maybe relay restarted). Re-registering...")
                        await self.register()
                except Exception as e:
                    print(f"[!] Heartbeat error: {e}")
                await asyncio.sleep(10) # Increased heartbeat interval
        
        heartbeat_task = asyncio.create_task(heartbeat())

        # 3. Polling loop
        try:
            while True:
                try:
                    resp = await self.client.get(f"{self.relay_url}/poll/{self.role}", params={"worker_id": self.node_id})
                    if resp.status_code == 200:
                        task_data = unpack_payload(resp.content)
                        if task_data:
                            task_id = task_data["task_id"]
                            input_payload = unpack_payload(task_data["payload"])
                            
                            print(f"[+] Processing task {task_id}...")
                            
                            try:
                                if asyncio.iscoroutinefunction(func):
                                    result = await func(input_payload)
                                else:
                                    result = await asyncio.to_thread(func, input_payload)
                                
                                result_bytes = pack_payload(result)
                                await self.client.post(f"{self.relay_url}/result/{task_id}", content=result_bytes)
                                print(f"[v] Task {task_id} completed.")
                            except Exception as e:
                                print(f"[!] Task {task_id} failed with error: {e}")
                                try:
                                    await self.client.post(
                                        f"{self.relay_url}/fail/{task_id}", 
                                        params={"error_msg": str(e)}
                                    )
                                except Exception as post_err:
                                    print(f"[!] Failed to report task failure to relay: {post_err}")
                    elif resp.status_code == 204:
                        pass # No tasks
                    else:
                        print(f"[!] Relay returned status {resp.status_code}")
                        await asyncio.sleep(2.0) # Back off on relay errors
                except Exception as e:
                    print(f"[!] Worker loop error: {e}")
                    await asyncio.sleep(2) # Prevent rapid-fire errors
                
                await asyncio.sleep(0.5)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            
            # Graceful deregistration
            try:
                await self.client.post(
                    f"{self.relay_url}/deregister", 
                    params={"node_id": self.node_id}
                )
                print(f"[*] Deregistered node {self.node_id} gracefully.")
            except Exception as e:
                print(f"[!] Failed to deregister node: {e}")
                
            await self.client.aclose()

def worker_node(role: str, relay_url: str = "http://localhost:8000"):
    """Decorator to turn a function into a distributed worker."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        def start(node_id: str = None):
            worker = Worker(role=role, relay_url=relay_url, node_id=node_id)
            asyncio.run(worker.run(func))
        
        wrapper.start = start
        return wrapper
    return decorator

class Cluster:
    """Entry point for submitting tasks and awaiting results."""
    def __init__(self, relay_url: str):
        self.relay_url = relay_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
        self.node_id = "client-" + str(uuid.uuid4())[:8]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()

    async def submit(self, role: str, payload: dict) -> str:
        """Submits a task and returns the task_id."""
        data = pack_payload(payload)
        resp = await self.client.post(
            f"{self.relay_url}/submit", 
            content=data, 
            params={"target_role": role, "sender_id": self.node_id}
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    async def wait_for(self, task_id: str, timeout: float = 60.0) -> dict:
        """Polls for the result of a task."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = await self.client.get(f"{self.relay_url}/get_result/{task_id}")
                if resp.status_code == 200:
                    if resp.headers.get("content-type") == "application/octet-stream":
                        return unpack_payload(resp.content)
                    else:
                        try:
                            data = resp.json()
                            if data.get("status") == "FAILED":
                                raise RuntimeError(f"Task failed on worker: {data.get('error')}")
                        except (ValueError, KeyError):
                            pass
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
        raise TimeoutError(f"Task {task_id} timed out.")
