import asyncio
import uuid
import time
import httpx
import functools
from typing import Callable, Optional
from .transport import pack_payload, unpack_payload

class Worker:
    def __init__(self, role: str, relay_url: str, node_id: str = None, idle_sleep: float = 0.0):
        self.role = role
        self.relay_url = relay_url.rstrip("/")
        self.node_id = node_id or str(uuid.uuid4())[:8]
        # Pause between polls. Not needed: /poll already long-polls on the relay.
        # Kept as a knob to reproduce the old (0.5s) behaviour in benchmarks.
        self.idle_sleep = idle_sleep
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

                if self.idle_sleep > 0:
                    await asyncio.sleep(self.idle_sleep)
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

        def start(node_id: str = None, idle_sleep: float = 0.0):
            worker = Worker(role=role, relay_url=relay_url, node_id=node_id, idle_sleep=idle_sleep)
            asyncio.run(worker.run(func))
        
        wrapper.start = start
        return wrapper
    return decorator

class Cluster:
    """Entry point for submitting tasks and awaiting results.

    By default results are fetched with long-polling: the relay answers as soon
    as the task finishes. ``poll_interval`` switches to the old short-polling
    (check every N seconds) and exists for comparison in benchmarks.
    """
    LONG_POLL_WAIT = 30.0

    def __init__(self, relay_url: str, poll_interval: Optional[float] = None):
        self.relay_url = relay_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
        self.node_id = "client-" + str(uuid.uuid4())[:8]
        self.poll_interval = poll_interval

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
        """Waits for the result of a task (long-poll by default)."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            long_poll = self.poll_interval is None
            params = {"wait": min(remaining, self.LONG_POLL_WAIT)} if long_poll else None
            try:
                resp = await self.client.get(f"{self.relay_url}/get_result/{task_id}", params=params)
                if resp.status_code == 200:
                    if resp.headers.get("content-type") == "application/octet-stream":
                        return unpack_payload(resp.content)
                    try:
                        data = resp.json()
                    except ValueError:
                        data = {}
                    if data.get("status") == "FAILED":
                        raise RuntimeError(f"Task failed on worker: {data.get('error')}")
            except httpx.HTTPError:
                # Relay unreachable or long-poll interrupted: back off briefly and retry.
                await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
                continue
            if not long_poll:
                await asyncio.sleep(self.poll_interval)
        raise TimeoutError(f"Task {task_id} timed out.")

    async def call(self, role: str, payload: dict, timeout: float = 120.0) -> dict:
        """Submits a task and waits for its result."""
        task_id = await self.submit(role, payload)
        return await self.wait_for(task_id, timeout=timeout)
