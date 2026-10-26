import pytest
from fastapi.testclient import TestClient
from distllm.relay import app, tasks, cleanup_stuck_tasks
from distllm.transport import pack_payload, unpack_payload
import torch
import time
import asyncio

client = TestClient(app)

def test_registration():
    response = client.post("/register", params={"node_id": "test-node", "role": "tester"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_heartbeat():
    client.post("/register", params={"node_id": "test-node", "role": "tester"})
    response = client.post("/heartbeat", params={"node_id": "test-node"})
    assert response.status_code == 200

def test_task_lifecycle():
    # 1. Register a worker
    client.post("/register", params={"node_id": "worker-1", "role": "compute"})
    
    # 2. Submit a task
    payload = {"x": torch.tensor([1, 2, 3])}
    packed_payload = pack_payload(payload)
    
    submit_resp = client.post(
        "/submit", 
        content=packed_payload, 
        params={"target_role": "compute", "sender_id": "client-1"}
    )
    assert submit_resp.status_code == 200
    task_id = submit_resp.json()["task_id"]
    
    # 3. Poll for the task
    poll_resp = client.get("/poll/compute", params={"worker_id": "worker-1"})
    assert poll_resp.status_code == 200
    task_data = unpack_payload(poll_resp.content)
    assert task_data["task_id"] == task_id
    
    # 4. Post result
    result_payload = {"y": torch.tensor([2, 4, 6])}
    packed_result = pack_payload(result_payload)
    res_resp = client.post(f"/result/{task_id}", content=packed_result)
    assert res_resp.status_code == 200
    
    # 5. Get result
    get_res_resp = client.get(f"/get_result/{task_id}")
    assert get_res_resp.status_code == 200
    final_result = unpack_payload(get_res_resp.content)
    assert torch.equal(final_result["y"], result_payload["y"])

@pytest.mark.asyncio
async def test_task_timeout_and_recovery():
    # 1. Register worker
    client.post("/register", params={"node_id": "worker-fail", "role": "heavy"})
    
    # 2. Submit task
    payload = {"data": "stuff"}
    submit_resp = client.post(
        "/submit", 
        content=pack_payload(payload), 
        params={"target_role": "heavy", "sender_id": "client-2"}
    )
    task_id = submit_resp.json()["task_id"]
    
    # 3. Assign to worker
    client.get("/poll/heavy", params={"worker_id": "worker-fail"})
    
    # Verify it is ASSIGNED
    resp = client.get(f"/get_result/{task_id}")
    assert resp.json()["status"] == "ASSIGNED"
    
    # 4. Mock the timeout
    tasks[task_id].assigned_at = time.time() - 100 
    
    # 5. Call cleanup manually
    await cleanup_stuck_tasks()
    
    # Verify it is PENDING again
    resp = client.get(f"/get_result/{task_id}")
    assert resp.json()["status"] == "PENDING"

def test_task_failure():
    # 1. Register worker
    client.post("/register", params={"node_id": "worker-err", "role": "buggy"})
    
    # 2. Submit task
    payload = {"data": "broken"}
    submit_resp = client.post(
        "/submit",
        content=pack_payload(payload),
        params={"target_role": "buggy", "sender_id": "client-err"}
    )
    task_id = submit_resp.json()["task_id"]
    
    # 3. Poll task
    client.get("/poll/buggy", params={"worker_id": "worker-err"})
    
    # 4. Report failure
    fail_resp = client.post(f"/fail/{task_id}", params={"error_msg": "something went wrong"})
    assert fail_resp.status_code == 200
    
    # 5. Check result status and error message
    get_res_resp = client.get(f"/get_result/{task_id}")
    assert get_res_resp.status_code == 200
    res_json = get_res_resp.json()
    assert res_json["status"] == "FAILED"
    assert res_json["error"] == "something went wrong"

def test_deregistration():
    # 1. Register worker
    client.post("/register", params={"node_id": "worker-dereg", "role": "cleanup"})
    
    # 2. Submit task
    payload = {"data": "cleanup-test"}
    submit_resp = client.post(
        "/submit",
        content=pack_payload(payload),
        params={"target_role": "cleanup", "sender_id": "client-dereg"}
    )
    task_id = submit_resp.json()["task_id"]
    
    # 3. Assign task to worker
    client.get("/poll/cleanup", params={"worker_id": "worker-dereg"})
    
    # Verify it is ASSIGNED
    resp = client.get(f"/get_result/{task_id}")
    assert resp.json()["status"] == "ASSIGNED"
    
    # 4. Deregister worker
    dereg_resp = client.post("/deregister", params={"node_id": "worker-dereg"})
    assert dereg_resp.status_code == 200
    
    # Verify the task is PENDING again immediately
    resp = client.get(f"/get_result/{task_id}")
    assert resp.json()["status"] == "PENDING"


@pytest.mark.asyncio
async def test_get_result_long_poll_returns_as_soon_as_done():
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay") as ac:
        submit = await ac.post(
            "/submit", content=pack_payload({"x": 1}),
            params={"target_role": "lp", "sender_id": "c"},
        )
        task_id = submit.json()["task_id"]

        async def finish_later():
            await asyncio.sleep(0.3)
            await ac.post(f"/result/{task_id}", content=pack_payload({"y": 2}))

        started = time.monotonic()
        finisher = asyncio.create_task(finish_later())
        resp = await ac.get(f"/get_result/{task_id}", params={"wait": 10})
        elapsed = time.monotonic() - started
        await finisher

        assert resp.headers["content-type"] == "application/octet-stream"
        assert unpack_payload(resp.content) == {"y": 2}
        # Ответ пришёл сразу после результата, а не по истечении wait.
        assert 0.25 < elapsed < 2.0


@pytest.mark.asyncio
async def test_get_result_long_poll_times_out_with_status():
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://relay") as ac:
        submit = await ac.post(
            "/submit", content=pack_payload({"x": 1}),
            params={"target_role": "lp-timeout", "sender_id": "c"},
        )
        task_id = submit.json()["task_id"]
        resp = await ac.get(f"/get_result/{task_id}", params={"wait": 0.2})
        assert resp.json() == {"status": "PENDING"}


def test_first_result_wins():
    submit = client.post(
        "/submit", content=pack_payload({"x": 1}),
        params={"target_role": "dup", "sender_id": "c"},
    )
    task_id = submit.json()["task_id"]

    assert client.post(f"/result/{task_id}", content=pack_payload({"y": "first"})).json() == {"status": "ok"}
    # Дубликат после переотдачи задачи не перезаписывает результат.
    assert client.post(f"/result/{task_id}", content=pack_payload({"y": "late"})).json() == {"status": "ignored"}
    assert client.post(f"/fail/{task_id}", params={"error_msg": "late"}).json() == {"status": "ignored"}
    assert unpack_payload(client.get(f"/get_result/{task_id}").content) == {"y": "first"}
