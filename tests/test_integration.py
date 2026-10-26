import asyncio
import torch
import multiprocessing

# Never fork: the test process has already run torch (its OpenMP thread pool is up), and
# a forked child can inherit a held lock and hang on its first forward. "fork" is the
# default on Linux (it made test_replicas_keep_sessions_without_replays hang in CI).
SPAWN = multiprocessing.get_context("spawn")
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
    import os
    import socket
    import subprocess
    from pathlib import Path

    if os.environ.get("DISTLLM_TEST_RELAY") == "rust":
        # Same suite against the Rust relay (cargo build --release in relay-rs/).
        exe = ".exe" if os.name == "nt" else ""
        binary = os.environ.get(
            "DISTLLM_RELAY_BIN", str(Path(__file__).parents[1] / f"relay-rs/target/release/distllm-relay{exe}")
        )
        p = subprocess.Popen([binary, "--port", "8002", "--host", "0.0.0.0"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p = SPAWN.Process(target=run_relay)
        p.start()
    deadline = time.time() + 30
    while time.time() < deadline:  # wait until the relay accepts connections
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8002)) == 0:
                break
        time.sleep(0.2)
    yield "http://localhost:8002"
    p.terminate()

@pytest.mark.asyncio
async def test_full_integration(relay_server):
    # Start worker in a separate process
    p_worker = SPAWN.Process(target=run_worker)
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
    p_worker = SPAWN.Process(target=run_failing_worker)
    p_worker.start()
    
    try:
        await asyncio.sleep(3) # Wait for worker registration
        
        async with Cluster(relay_server) as cluster:
            # Submit task
            task_id = await cluster.submit("failing-worker", {"x": 10})
            
            # Wait for result, expect it to raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await cluster.wait_for(task_id, timeout=10)
            
            assert "Task failed on worker: ValueError: simulated processing error" in str(exc_info.value)
            
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
    from distllm.hf import PipelineGenerator, SpeculativeGenerator

    prompt = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA, dtype=torch.float32).eval()
    seq, expected = prompt, []
    with torch.inference_mode():
        for _ in range(6):
            nxt = int(model(seq).logits[0, -1].argmax())
            expected.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)

    workers = [
        SPAWN.Process(target=run_stage_worker, args=(0, 0, "hf-stage-0")),
        SPAWN.Process(target=run_stage_worker, args=(1, 1, "hf-stage-1")),
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
            speculative = SpeculativeGenerator(
                cluster.call, roles, model, k=3, route_call=cluster.call_route
            )
            speculative_result = await asyncio.wait_for(speculative.generate(prompt, 6), timeout=60)
        assert result.token_ids == expected
        assert routed_result.token_ids == expected
        assert speculative_result.token_ids == expected
        assert speculative_result.rounds < 6
    finally:
        for p in workers:
            p.terminate()


def run_stage_replica(layer_start, layer_end, role, node_id):
    from distllm.hf import load_stage

    stage = load_stage(TINY_LLAMA, layer_start, layer_end, device="cpu", dtype="float32")

    @worker_node(role=role, relay_url="http://localhost:8002")
    def process(payload):
        return stage.handle(payload)

    process.start(node_id=node_id)


@pytest.mark.asyncio
async def test_replicas_keep_sessions_without_replays(relay_server):
    """Two replicas per stage: affinity keeps each session on the replica with its cache."""
    from distllm.hf import PipelineGenerator

    workers = [
        SPAWN.Process(target=run_stage_replica, args=(0, 0, "rep-stage-0", f"r0-{i}"))
        for i in range(2)
    ] + [
        SPAWN.Process(target=run_stage_replica, args=(1, 1, "rep-stage-1", f"r1-{i}"))
        for i in range(2)
    ]
    for p in workers:
        p.start()
    try:
        prompt = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])
        async with Cluster(relay_server) as cluster:
            generator = PipelineGenerator(cluster.call, ["rep-stage-0", "rep-stage-1"])
            results = await asyncio.wait_for(
                asyncio.gather(*(generator.generate(prompt, 8) for _ in range(3))), timeout=180
            )
        assert all(r.cache_recoveries == 0 for r in results)
        assert all(r.token_ids == results[0].token_ids for r in results)
    finally:
        for p in workers:
            p.terminate()



def run_batched_stage(layer_start, layer_end, role):
    from distllm.hf import load_stage

    stage = load_stage(TINY_LLAMA, layer_start, layer_end, device="cpu", dtype="float32")

    @worker_node(role=role, relay_url="http://localhost:8002", max_batch=8, batched=True)
    def process(payloads):
        return stage.handle_batch(payloads)

    process.start(node_id=role)


@pytest.mark.asyncio
async def test_batched_workers_serve_concurrent_sessions(relay_server):
    from distllm.hf import PipelineGenerator

    workers = [
        SPAWN.Process(target=run_batched_stage, args=(0, 0, "batch-stage-0")),
        SPAWN.Process(target=run_batched_stage, args=(1, 1, "batch-stage-1")),
    ]
    for p in workers:
        p.start()
    try:
        prompts = [torch.tensor([[1, 5, 9, 13, 21, 34, 55]]), torch.tensor([[2, 4, 8]]),
                   torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]]), torch.tensor([[7, 7]])]
        async with Cluster(relay_server) as cluster:
            generator = PipelineGenerator(cluster.call, ["batch-stage-0", "batch-stage-1"])
            concurrent = await asyncio.wait_for(
                asyncio.gather(*(generator.generate(p, 8) for p in prompts)), timeout=180
            )
            sequential = [await generator.generate(p, 8) for p in prompts]
        assert [r.token_ids for r in concurrent] == [r.token_ids for r in sequential]
        assert all(r.cache_recoveries == 0 for r in concurrent)
    finally:
        for p in workers:
            p.terminate()
