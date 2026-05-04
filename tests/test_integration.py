import asyncio
import torch
import multiprocessing
import time
import pytest
from distllm import worker_node, Cluster, start_relay

# Define a test worker at module level for pickling
@worker_node(role="test-worker", relay_url="http://localhost:8002")
def my_worker(payload):
    x = payload["x"]
    return {"y": x + 1}

def run_relay():
    import uvicorn
    from distllm.relay import app
    uvicorn.run(app, host="0.0.0.0", port=8002)

def run_worker():
    my_worker.start(node_id="W-INT")

@pytest.fixture(scope="module")
def relay_server():
    p = multiprocessing.Process(target=run_relay)
    p.start()
    time.sleep(2) # Wait for startup
    yield "http://localhost:8002"
    p.terminate()

@pytest.mark.asyncio
async def test_full_integration(relay_server):
    # Start worker in a separate process
    p_worker = multiprocessing.Process(target=run_worker)
    p_worker.start()
    
    try:
        await asyncio.sleep(3) # Wait for worker registration
        
        cluster = Cluster(relay_server)
        
        # Submit task
        task_id = await cluster.submit("test-worker", {"x": torch.tensor([10, 20])})
        
        # Wait for result
        result = await cluster.wait_for(task_id, timeout=10)
        
        assert torch.equal(result["y"], torch.tensor([11, 21]))
        
    finally:
        p_worker.terminate()
