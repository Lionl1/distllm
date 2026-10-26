"""Client/worker behaviour when the relay or the network misbehaves."""

import time

import httpx
import pytest
import torch

from distllm import Cluster, TaskLostError, Worker
from distllm.relay import app


@pytest.mark.asyncio
async def test_wait_for_unknown_task_fails_fast():
    cluster = Cluster("http://relay")
    cluster.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    started = time.monotonic()
    with pytest.raises(TaskLostError):
        await cluster.wait_for("no-such-task", timeout=30)
    assert time.monotonic() - started < 2
    await cluster.close()


@pytest.mark.asyncio
async def test_wait_for_backs_off_on_relay_errors():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(500)

    cluster = Cluster("http://relay")
    cluster.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(TimeoutError):
        await cluster.wait_for("t", timeout=1.5)
    # Без паузы были бы тысячи запросов за 1.5 с.
    assert len(calls) <= 4
    await cluster.close()


@pytest.mark.asyncio
async def test_result_upload_is_retried(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(request.url.params.get("worker_id"))
        if len(attempts) < 3:
            raise httpx.ConnectError("tunnel hiccup")
        return httpx.Response(200, json={"status": "ok"})

    async def no_sleep(_):
        return None

    worker = Worker("r", "http://relay", node_id="w1")
    worker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("distllm.worker.asyncio.sleep", no_sleep)

    assert await worker._post_result("t", b"payload", {"worker_id": "w1"}) is True
    assert attempts == ["w1", "w1", "w1"]
    await worker.client.aclose()


@pytest.mark.asyncio
async def test_result_upload_gives_up_without_raising(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("down")

    async def no_sleep(_):
        return None

    worker = Worker("r", "http://relay")
    worker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("distllm.worker.asyncio.sleep", no_sleep)
    assert await worker._post_result("t", b"x", {}) is False
    await worker.client.aclose()


@pytest.mark.asyncio
async def test_generator_recovers_from_lost_task():
    transformers = pytest.importorskip("transformers")
    from distllm.hf import PipelineGenerator, load_stage

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    prompt = torch.tensor([[1, 5, 9, 13, 21]])
    stages = {f"s{i}": load_stage(model_id, i, i, device="cpu", dtype="float32") for i in range(2)}
    calls = {"n": 0}

    async def flaky(role, payload):
        calls["n"] += 1
        if calls["n"] == 5:
            raise TaskLostError("relay restarted")
        return stages[role].handle(payload)

    async def reliable(role, payload):
        return stages[role].handle(payload)

    expected = (await PipelineGenerator(reliable, list(stages)).generate(prompt, 6)).token_ids
    result = await PipelineGenerator(flaky, list(stages)).generate(prompt, 6)
    assert result.token_ids == expected
    assert result.cache_recoveries == 1


@pytest.mark.asyncio
async def test_start_inside_running_loop_points_to_run_async():
    from distllm import worker_node

    @worker_node(role="nb", relay_url="http://relay")
    def process(payload):
        return payload

    with pytest.raises(RuntimeError, match="run_async"):
        process.start()
    assert callable(process.run_async)
