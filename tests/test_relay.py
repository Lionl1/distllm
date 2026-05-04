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
