import asyncio
import functools
import os
import time
import uuid
from collections import OrderedDict
from typing import Callable, Optional

import httpx

from .auth import first_token
from .crypto import RID_KEY, PayloadDecryptionError, cipher_from, is_sealed
from .transport import pack_payload, unpack_payload


def _http_client(spec: Optional[str]) -> httpx.AsyncClient:
    token = first_token(spec)  # a spec may list several tokens or name a file
    # High timeout for unstable tunnels.
    return httpx.AsyncClient(timeout=120.0, headers={"Authorization": f"Bearer {token}"} if token else {})


def _open(cipher, data: bytes) -> bytes:
    if cipher:
        return cipher.open(data)
    if is_sealed(data):
        raise PayloadDecryptionError("Payload is encrypted: set DISTLLM_PAYLOAD_KEY")
    return data


def _session_of(payload) -> Optional[str]:
    session = payload.get("session_id") if isinstance(payload, dict) else None
    return session if isinstance(session, str) and session else None


class RelayAuthError(PermissionError):
    """The relay rejected the token (HTTP 401)."""


class TaskLostError(RuntimeError):
    """The relay does not know the task (it restarted, or the result expired)."""


class Worker:
    RESULT_RETRIES = 5
    MAX_CLAIMED_SESSIONS = 64
    SESSION_CLAIM_TTL = 900.0  # seconds, as the relay's session affinity

    def __init__(
        self,
        role: str,
        relay_url: str,
        node_id: str = None,
        idle_sleep: float = 0.0,
        token: Optional[str] = None,
        max_batch: int = 1,
        batched: bool = False,
        heartbeat_interval: Optional[float] = None,
        payload_key: Optional[str] = None,
    ):
        """``max_batch > 1`` takes up to that many queued tasks per poll. With
        ``batched=True`` the function receives a list of payloads and returns a list of
        results (an Exception instance marks a failed item); otherwise the tasks run one
        by one. ``payload_key`` (else ``$DISTLLM_PAYLOAD_KEY``) turns on end-to-end
        encryption of payloads; clients and workers must share it."""
        self.role = role
        self.max_batch = max(1, max_batch)
        self.batched = batched
        # Seconds between heartbeats ($DISTLLM_HEARTBEAT_INTERVAL); the relay declares a
        # worker dead after its node timeout, so keep ~3 heartbeats within it.
        self.heartbeat_interval = heartbeat_interval or float(os.environ.get("DISTLLM_HEARTBEAT_INTERVAL", "10"))
        self.relay_url = relay_url.rstrip("/")
        self.node_id = node_id or uuid.uuid4().hex[:8]
        # Pause between polls. Not needed: /poll already long-polls on the relay.
        # Kept as a knob to reproduce the old (0.5s) behaviour in benchmarks.
        self.idle_sleep = idle_sleep
        # Worker token: explicit, else $DISTLLM_WORKER_TOKEN, else $DISTLLM_TOKEN.
        self.client = _http_client(token or os.environ.get("DISTLLM_WORKER_TOKEN") or os.environ.get("DISTLLM_TOKEN"))
        self.cipher = cipher_from(payload_key)
        # Sessions this worker served recently (session_id -> last use): claimed again when
        # it re-registers after a relay restart, so their KV caches are not replayed elsewhere.
        self._sessions: "OrderedDict[str, float]" = OrderedDict()

    def _track_sessions(self, tasks: list) -> None:
        now = time.monotonic()
        for _, payload in tasks:
            if session := _session_of(payload):
                self._sessions.pop(session, None)
                if not payload.get("close"):
                    self._sessions[session] = now
        while len(self._sessions) > self.MAX_CLAIMED_SESSIONS or (
            self._sessions and now - next(iter(self._sessions.values())) > self.SESSION_CLAIM_TTL
        ):
            self._sessions.popitem(last=False)

    def _decode(self, data: bytes):
        """A task payload, or the exception that makes it unusable (reported as the task's failure)."""
        try:
            return unpack_payload(_open(self.cipher, data))
        except Exception as e:
            return e

    async def register(self):
        """Registers the worker node with the relay, retrying if necessary."""
        params = {"node_id": self.node_id, "role": self.role, "sessions": ",".join(self._sessions)}
        while True:
            print(f"[*] Registering node {self.node_id}...")
            try:
                resp = await self.client.post(f"{self.relay_url}/register", params=params)
                if resp.status_code == 200:
                    print("[v] Registration successful.")
                    return
                if resp.status_code == 401:
                    raise RelayAuthError("Relay rejected the worker token (set DISTLLM_WORKER_TOKEN or DISTLLM_TOKEN)")
                print(f"[!] Registration failed with status {resp.status_code}. Retrying in 5s...")
            except httpx.HTTPError as e:
                print(f"[!] Registration failed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

    async def _call(self, func: Callable, arg):
        if asyncio.iscoroutinefunction(func):
            return await func(arg)
        # Sync functions run in a thread so heartbeats keep flowing.
        return await asyncio.to_thread(func, arg)

    async def _run_func(self, func: Callable, payloads: list) -> list:
        """One result (or exception) per payload."""
        if self.batched:
            try:
                results = await self._call(func, payloads)
                if len(results) != len(payloads):
                    raise RuntimeError(f"batch function returned {len(results)} results for {len(payloads)} tasks")
                return results
            except Exception as e:  # the whole batch failed
                return [e] * len(payloads)
        results = []
        for payload in payloads:
            try:
                results.append(await self._call(func, payload))
            except Exception as e:
                results.append(e)
        return results

    async def _process(self, func: Callable, tasks: list) -> None:
        """Run tasks (one, or a batch with ``batched=True``) and upload every outcome."""
        # Payloads that could not be decoded fail on their own; the rest run.
        good = [(task_id, payload) for task_id, payload in tasks if not isinstance(payload, Exception)]
        outcomes = [(task_id, payload, None) for task_id, payload in tasks if isinstance(payload, Exception)]
        if good:
            self._track_sessions(good)
            # Encrypted requests carry an id that the result must echo (see Cluster.wait_for).
            rids = [payload.pop(RID_KEY, None) if isinstance(payload, dict) else None for _, payload in good]
            print(f"[+] Processing {', '.join(task_id for task_id, _ in good)}...")
            results = await self._run_func(func, [payload for _, payload in good])
            outcomes += [(task_id, result, rid) for (task_id, _), result, rid in zip(good, results, rids)]
        await asyncio.gather(*(self._report(*outcome) for outcome in outcomes))

    async def _report(self, task_id: str, result, rid: Optional[str] = None) -> None:
        if rid is not None and isinstance(result, dict):
            result = {**result, RID_KEY: rid}
        if not isinstance(result, Exception):
            try:
                body = pack_payload(result)
                if self.cipher:
                    body = self.cipher.seal(body)
            except Exception as e:  # unserializable result
                result = e
        if isinstance(result, Exception):
            print(f"[!] Task {task_id} failed with error: {result}")
            try:
                # In the body: tracebacks do not fit into a URL.
                await self.client.post(
                    f"{self.relay_url}/fail/{task_id}",
                    params={"worker_id": self.node_id},
                    content=f"{type(result).__name__}: {result}".encode(),
                )
            except httpx.HTTPError as e:
                print(f"[!] Failed to report task failure to relay: {e}")
            return
        params = {"worker_id": self.node_id}
        if isinstance(result, dict) and result.get("stop_chain"):
            params["stop_chain"] = "true"
        if await self._post_result(task_id, body, params):
            print(f"[v] Task {task_id} completed.")

    async def _post_result(self, task_id: str, body: bytes, params: dict) -> bool:
        """Upload a result, retrying transient failures with exponential backoff.

        The work is already done (and a stage has advanced its KV cache): losing the
        upload would stall the client until the relay re-queues the task.
        """
        for attempt in range(self.RESULT_RETRIES):
            try:
                resp = await self.client.post(f"{self.relay_url}/result/{task_id}", content=body, params=params)
                if resp.status_code == 401:
                    raise RelayAuthError("Relay rejected the worker token")
                if resp.status_code < 500:
                    # 200 ok/ignored/forwarded; 404 task gone; 409 re-assigned: nothing to retry.
                    return resp.status_code == 200
                print(f"[!] Relay returned {resp.status_code} for result of {task_id} (attempt {attempt + 1})")
            except httpx.HTTPError as e:
                print(f"[!] Failed to upload result of {task_id} (attempt {attempt + 1}): {e}")
            if attempt + 1 < self.RESULT_RETRIES:
                await asyncio.sleep(0.5 * 2**attempt)
        print(f"[!] Giving up on result of {task_id}; the relay will re-queue the task")
        return False

    async def _heartbeat(self):
        while True:
            try:
                resp = await self.client.post(f"{self.relay_url}/heartbeat", params={"node_id": self.node_id})
                if resp.status_code == 404:
                    print("[!] Node not registered on relay (maybe relay restarted). Re-registering...")
                    await self.register()
            except Exception as e:  # keep beating whatever happens
                print(f"[!] Heartbeat error: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    async def _poll_once(self, func: Callable) -> None:
        params = {"worker_id": self.node_id}
        if self.max_batch > 1:
            params["max_tasks"] = self.max_batch
        resp = await self.client.get(f"{self.relay_url}/poll/{self.role}", params=params)
        if resp.status_code == 200:
            data = unpack_payload(resp.content)
            items = data.get("tasks", [data])
            await self._process(func, [(item["task_id"], self._decode(item["payload"])) for item in items])
        elif resp.status_code == 401:
            raise RelayAuthError("Relay rejected the worker token")
        elif resp.status_code != 204:  # 204: no tasks
            print(f"[!] Relay returned status {resp.status_code}")
            await asyncio.sleep(2.0)  # back off on relay errors

    async def run(self, func: Callable):
        """Main loop for the worker node."""
        print(f"[*] Worker {self.node_id} (Role: {self.role})")
        print(f"[*] Connecting to relay: {self.relay_url}")
        await self.register()
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            while True:
                try:
                    await self._poll_once(func)
                except RelayAuthError:
                    raise
                except Exception as e:
                    print(f"[!] Worker loop error: {e}")
                    await asyncio.sleep(2)  # prevent rapid-fire errors
                if self.idle_sleep > 0:
                    await asyncio.sleep(self.idle_sleep)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            try:
                await self.client.post(f"{self.relay_url}/deregister", params={"node_id": self.node_id})
                print(f"[*] Deregistered node {self.node_id} gracefully.")
            except httpx.HTTPError as e:
                print(f"[!] Failed to deregister node: {e}")
            await self.client.aclose()


def worker_node(
    role: str,
    relay_url: str = "http://localhost:8000",
    token: Optional[str] = None,
    max_batch: int = 1,
    batched: bool = False,
    payload_key: Optional[str] = None,
):
    """Decorator to turn a function into a distributed worker."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        def _worker(node_id: str = None, idle_sleep: float = 0.0) -> Worker:
            return Worker(role, relay_url, node_id, idle_sleep, token, max_batch, batched, payload_key=payload_key)

        def start(node_id: str = None, idle_sleep: float = 0.0):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_worker(node_id, idle_sleep).run(func))
            raise RuntimeError(
                "An event loop is already running (Jupyter/Colab): use `await process.run_async()` instead of start()"
            )

        async def run_async(node_id: str = None, idle_sleep: float = 0.0):
            """Run the worker inside an existing event loop (e.g. `await` in a notebook cell)."""
            await _worker(node_id, idle_sleep).run(func)

        wrapper.start, wrapper.run_async = start, run_async
        return wrapper

    return decorator


class Cluster:
    """Entry point for submitting tasks and awaiting results.

    By default results are fetched with long-polling: the relay answers as soon
    as the task finishes. ``poll_interval`` switches to the old short-polling
    (check every N seconds) and exists for comparison in benchmarks.
    """

    LONG_POLL_WAIT = 30.0
    SUBMIT_RETRY_FOR = 60.0

    def __init__(
        self,
        relay_url: str,
        poll_interval: Optional[float] = None,
        token: Optional[str] = None,
        payload_key: Optional[str] = None,
    ):
        """``payload_key`` (else ``$DISTLLM_PAYLOAD_KEY``): encrypt payloads end to end."""
        self.relay_url = relay_url.rstrip("/")
        self.client = _http_client(token or os.environ.get("DISTLLM_TOKEN"))
        self.cipher = cipher_from(payload_key)
        self.node_id = "client-" + uuid.uuid4().hex[:8]
        self.poll_interval = poll_interval
        self._rids = {}  # task_id -> request id expected back in the sealed result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()

    @staticmethod
    def _check_auth(resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise RelayAuthError("Relay rejected the client token (set DISTLLM_TOKEN)")

    async def submit(self, role: str, payload: dict, route: Optional[list] = None) -> str:
        """Submits a task and returns the task_id.

        ``route`` lists roles that receive the result next: the relay forwards it
        through them and the returned task_id resolves to the last role's result.
        Connection errors (relay restarting, tunnel hiccup) are retried with back-off for
        up to SUBMIT_RETRY_FOR seconds; a duplicate submission is harmless because stages
        check start_pos.
        """
        params = {"target_role": role, "sender_id": self.node_id}
        if route:
            params["route"] = ",".join(route)
        if session := _session_of(payload):
            params["session"] = session  # keeps the session on the replica that holds its KV cache
        rid = None
        if self.cipher and isinstance(payload, dict):
            # Sealed with the request and echoed sealed by every stage of a route: the relay
            # cannot answer this request with a (validly encrypted) result of another one.
            rid = uuid.uuid4().hex
            payload = {**payload, RID_KEY: rid}
        data = pack_payload(payload)
        if self.cipher:
            data = self.cipher.seal(data)
        deadline = time.monotonic() + self.SUBMIT_RETRY_FOR
        delay = 0.2
        while True:
            try:
                resp = await self.client.post(f"{self.relay_url}/submit", content=data, params=params)
                if resp.status_code < 500 or time.monotonic() >= deadline:
                    break  # a lasting 5xx is reported by raise_for_status below
            except httpx.TransportError:
                if time.monotonic() >= deadline:
                    raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)
        self._check_auth(resp)
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        if rid:
            self._rids[task_id] = rid
        return task_id

    def _result(self, task_id: str, content: bytes):
        result = unpack_payload(_open(self.cipher, content))
        expected = self._rids.pop(task_id, None)
        if expected is not None and (not isinstance(result, dict) or result.pop(RID_KEY, None) != expected):
            raise PayloadDecryptionError(f"Result of task {task_id} belongs to another request (replayed?)")
        return result

    async def wait_for(self, task_id: str, timeout: float = 60.0) -> dict:
        """Waits for the result of a task (long-poll by default)."""
        deadline = time.monotonic() + timeout
        long_poll = self.poll_interval is None
        while (remaining := deadline - time.monotonic()) > 0:
            params = {"wait": min(remaining, self.LONG_POLL_WAIT)} if long_poll else None
            try:
                resp = await self.client.get(f"{self.relay_url}/get_result/{task_id}", params=params)
            except httpx.HTTPError:
                resp = None  # relay unreachable or long-poll interrupted
            if resp is not None:
                self._check_auth(resp)
                if resp.status_code == 404:
                    raise TaskLostError(f"Task {task_id} is unknown to the relay (it restarted or the result expired)")
                if resp.status_code == 200:
                    if resp.headers.get("content-type") == "application/octet-stream":
                        return self._result(task_id, resp.content)
                    data = resp.json()
                    if data.get("status") == "FAILED":
                        raise RuntimeError(f"Task failed on worker: {data.get('error')}")
                    if not long_poll:
                        await asyncio.sleep(self.poll_interval)
                    continue
            # Relay error or unreachable: back off instead of hammering it.
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"Task {task_id} timed out.")

    async def call(self, role: str, payload: dict, timeout: float = 120.0) -> dict:
        """Submits a task and waits for its result."""
        return await self.wait_for(await self.submit(role, payload), timeout=timeout)

    async def call_route(self, roles: list, payload: dict, timeout: float = 120.0) -> dict:
        """Sends a payload through several roles in order, relay-side, and waits for the end result."""
        return await self.wait_for(await self.submit(roles[0], payload, route=list(roles[1:])), timeout=timeout)
