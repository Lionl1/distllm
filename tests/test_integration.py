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

@worker_node(role="failing-worker", relay_url="http://localhost:8002")
def my_failing_worker(payload):
    raise ValueError("simulated processing error")

def run_relay():
    import uvicorn
    from distllm.relay import app
    uvicorn.run(app, host="0.0.0.0", port=8002)

def run_worker():
    my_worker.start(node_id="W-INT")

def run_failing_worker():
    my_failing_worker.start(node_id="W-FAIL")

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

@pytest.mark.asyncio
async def test_failing_worker_integration(relay_server):
    # Start failing worker in a separate process
    p_worker = multiprocessing.Process(target=run_failing_worker)
    p_worker.start()
    
    try:
        await asyncio.sleep(3) # Wait for worker registration
        
        async with Cluster(relay_server) as cluster:
            # Submit task
            task_id = await cluster.submit("failing-worker", {"x": 10})
            
            # Wait for result, expect it to raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await cluster.wait_for(task_id, timeout=10)
            
            assert "Task failed on worker: simulated processing error" in str(exc_info.value)
            
    finally:
        p_worker.terminate()


TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def run_stage_worker(layer_start, layer_end, role):
    from distllm.hf import load_stage

    stage = load_stage(TINY_LLAMA, layer_start, layer_end, device="cpu", dtype="float32")

    @worker_node(role=role, relay_url="http://localhost:8002")
    def process(payload):
        return stage.handle(payload)

    process.start(node_id=role)


@pytest.mark.asyncio
async def test_hf_pipeline_over_relay(relay_server):
    """Two stage processes behind the relay generate exactly what the full model does."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    from distllm.hf import PipelineGenerator

    prompt = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA, dtype=torch.float32).eval()
    seq, expected = prompt, []
    with torch.inference_mode():
        for _ in range(6):
            nxt = int(model(seq).logits[0, -1].argmax())
            expected.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)

    workers = [
        multiprocessing.Process(target=run_stage_worker, args=(0, 0, "hf-stage-0")),
        multiprocessing.Process(target=run_stage_worker, args=(1, 1, "hf-stage-1")),
    ]
    for p in workers:
        p.start()
    try:
        async with Cluster(relay_server) as cluster:
            roles = ["hf-stage-0", "hf-stage-1"]
            generator = PipelineGenerator(cluster.call, roles)
            # Workers load the model on start; the first tasks wait in the relay queue.
            result = await asyncio.wait_for(generator.generate(prompt, 6), timeout=120)
            routed = PipelineGenerator(cluster.call, roles, route_call=cluster.call_route)
            routed_result = await asyncio.wait_for(routed.generate(prompt, 6), timeout=60)
        assert result.token_ids == expected
        assert routed_result.token_ids == expected
    finally:
        for p in workers:
            p.terminate()
