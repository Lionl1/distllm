"""Correctness of pipeline-split generation against the full model.

Uses tiny random checkpoints from the Hub (a few MB each), runs on CPU.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from distllm.hf import CacheRecoveryError, PipelineGenerator, load_stage  # noqa: E402

MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",  # Llama, safetensors
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",  # Qwen2.5
    "yujiepan/qwen3-tiny-random",  # Qwen3 (q/k norm), tied embeddings
    "HuggingFaceM4/tiny-random-MistralForCausalLM",  # Mistral, pytorch_model.bin
]
PROMPT = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])
NEW_TOKENS = 10


def _reference_greedy(model_id: str, input_ids: torch.Tensor, steps: int) -> list[int]:
    """Ground truth: full model, full sequence every step, no cache, no generation_config."""
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    seq = input_ids
    out: list[int] = []
    with torch.inference_mode():
        for _ in range(steps):
            next_id = int(model(seq).logits[0, -1].argmax())
            out.append(next_id)
            seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
    return out


def _split_points(n_layers: int, n_stages: int) -> list[tuple[int, int]]:
    bounds = [round(i * n_layers / n_stages) for i in range(n_stages + 1)]
    return [(bounds[i], bounds[i + 1] - 1) for i in range(n_stages)]


def _make_stages(model_id: str, n_stages: int):
    n_layers = transformers.AutoConfig.from_pretrained(model_id).num_hidden_layers
    n_stages = min(n_stages, n_layers)
    return [
        load_stage(model_id, a, b, device="cpu", dtype="float32")
        for a, b in _split_points(n_layers, n_stages)
    ]


def _direct_call(stages, calls=None):
    by_role = {f"stage-{i}": s for i, s in enumerate(stages)}

    async def call(role, payload):
        if calls is not None:
            calls.append((role, payload.get("start_pos"), payload.get("close", False)))
        return by_role[role].handle(payload)

    return call, list(by_role)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MODELS)
async def test_split_generation_matches_full_model(model_id):
    expected = _reference_greedy(model_id, PROMPT, NEW_TOKENS)
    stages = _make_stages(model_id, n_stages=2)
    call, roles = _direct_call(stages)

    result = await PipelineGenerator(call, roles).generate(PROMPT, NEW_TOKENS)

    assert result.token_ids == expected
    assert result.cache_recoveries == 0
    # Сессии закрыты после генерации.
    assert all(s.cached_length("any") == 0 for s in stages)


@pytest.mark.asyncio
async def test_without_kv_cache_gives_same_tokens():
    model_id = MODELS[0]
    stages = _make_stages(model_id, n_stages=2)
    call, roles = _direct_call(stages)

    cached = await PipelineGenerator(call, roles).generate(PROMPT, NEW_TOKENS)
    recomputed = await PipelineGenerator(call, roles, use_kv_cache=False).generate(PROMPT, NEW_TOKENS)

    assert cached.token_ids == recomputed.token_ids


@pytest.mark.asyncio
async def test_kv_cache_sends_only_new_tokens():
    stages = _make_stages(MODELS[0], n_stages=2)
    calls: list = []
    call, roles = _direct_call(stages, calls)

    await PipelineGenerator(call, roles).generate(PROMPT, 4)

    starts = [start for role, start, close in calls if role == "stage-0" and not close]
    assert starts == [0, 7, 8, 9]


@pytest.mark.asyncio
async def test_recovers_when_a_stage_loses_its_cache():
    model_id = MODELS[2]
    expected = _reference_greedy(model_id, PROMPT, NEW_TOKENS)
    stages = _make_stages(model_id, n_stages=2)
    inner, roles = _direct_call(stages)
    steps = {"n": 0}

    async def flaky_call(role, payload):
        # На 4-м проходе второй стадии «перезапускаем узел» — он теряет все кэши.
        if role == "stage-1" and not payload.get("close"):
            steps["n"] += 1
            if steps["n"] == 4:
                stages[1].drop_sessions()
        return await inner(role, payload)

    result = await PipelineGenerator(flaky_call, roles).generate(PROMPT, NEW_TOKENS)

    assert result.token_ids == expected
    assert result.cache_recoveries == 1


@pytest.mark.asyncio
async def test_gives_up_when_cache_is_lost_every_time():
    stages = _make_stages(MODELS[0], n_stages=2)
    inner, roles = _direct_call(stages)

    async def always_forgetful(role, payload):
        if role == "stage-1":
            stages[1].drop_sessions()
            if payload.get("start_pos", 0) == 0 and "hidden_states" in payload:
                # Даже полный повтор «падает»: эмулируем узел, который теряет кэш постоянно.
                return {"cache_miss": True, "cached": 0}
        return await inner(role, payload)

    with pytest.raises(CacheRecoveryError):
        await PipelineGenerator(always_forgetful, roles).generate(PROMPT, NEW_TOKENS)


def test_stage_loads_only_its_layers():
    model_id = MODELS[0]
    full = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    first, second = _make_stages(model_id, n_stages=2)

    assert first.is_first and not first.is_last
    assert second.is_last and not second.is_first
    assert first.base.embed_tokens is not None and first.model.lm_head is None
    assert second.base.embed_tokens is None and second.model.lm_head is not None

    full_params = sum(p.numel() for p in full.parameters())
    stage_params = [sum(p.numel() for p in s.model.parameters()) for s in (first, second)]
    assert all(n < full_params for n in stage_params)


def test_invalid_layer_range():
    with pytest.raises(ValueError, match="Invalid layer range"):
        load_stage(MODELS[0], 1, 99, device="cpu")


def test_stage_rejects_unexpected_position():
    (stage,) = _make_stages(MODELS[0], n_stages=1)
    stage.handle({"session_id": "s", "start_pos": 0, "input_ids": PROMPT})

    assert stage.handle({"session_id": "s", "start_pos": 3, "input_ids": PROMPT[:, :1]}) == {
        "cache_miss": True,
        "cached": PROMPT.shape[1],
        "stop_chain": True,
    }
    assert stage.handle({"session_id": "other", "start_pos": 5, "input_ids": PROMPT[:, :1]})["cache_miss"]


@pytest.mark.asyncio
async def test_route_call_gives_same_tokens():
    """With a route the stage outputs are forwarded as is: emulate the relay chain directly."""
    model_id = MODELS[1]
    expected = _reference_greedy(model_id, PROMPT, NEW_TOKENS)
    stages = _make_stages(model_id, n_stages=2)
    call, roles = _direct_call(stages)
    by_role = dict(zip(roles, stages))
    routed = []

    async def route_call(route, payload):
        routed.append(list(route))
        for role in route:
            payload = by_role[role].handle(payload)
            if payload.get("stop_chain"):
                break
        return payload

    result = await PipelineGenerator(call, roles, route_call=route_call).generate(PROMPT, NEW_TOKENS)

    assert result.token_ids == expected
    assert routed and all(r == roles for r in routed)


def test_intermediate_stage_output_is_next_payload():
    first, second = _make_stages(MODELS[0], n_stages=2)
    out = first.handle({"session_id": "s", "start_pos": 0, "input_ids": PROMPT})

    assert out["session_id"] == "s" and out["start_pos"] == 0
    assert "logits" in second.handle(out)
