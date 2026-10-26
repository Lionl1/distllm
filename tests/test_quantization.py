"""Quantized stages (bitsandbytes nf4 / fp4 / int8). bitsandbytes >= 0.46 runs on CPU too."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
bnb = pytest.importorskip("bitsandbytes")

from distllm.hf import PipelineGenerator, load_stage  # noqa: E402

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
QWEN3_TIED = "yujiepan/qwen3-tiny-random"
PROMPT = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])


def _stages(model_id, quantize, n_stages):
    n_layers = transformers.AutoConfig.from_pretrained(model_id).num_hidden_layers
    bounds = [round(i * n_layers / n_stages) for i in range(n_stages + 1)]
    return [
        load_stage(model_id, bounds[i], bounds[i + 1] - 1, device="cpu", dtype="float32", quantize=quantize)
        for i in range(n_stages)
    ]


async def _generate(stages, steps=8):
    by_role = {f"s{i}": s for i, s in enumerate(stages)}

    async def call(role, payload):
        return by_role[role].handle(payload)

    return (await PipelineGenerator(call, list(by_role)).generate(PROMPT, steps)).token_ids


def _linear_weight_bytes(stage):
    total = 0
    for module in stage.base.layers.modules():
        if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            total += module.weight.numel() * module.weight.element_size()
    return total


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", [LLAMA, QWEN3_TIED])
@pytest.mark.parametrize("method", ["nf4", "int8"])
async def test_split_quantized_equals_single_stage_quantized(model_id, method):
    # Квантование слоя не зависит от того, где проходит граница стадий.
    single = await _generate(_stages(model_id, method, 1))
    split = await _generate(_stages(model_id, method, 2))
    assert split == single


@pytest.mark.parametrize("method,max_ratio", [("nf4", 0.3), ("fp4", 0.3), ("int8", 0.3)])
def test_quantized_layers_are_smaller(method, max_ratio):
    (full,) = _stages(LLAMA, None, 1)
    (quantized,) = _stages(LLAMA, method, 1)
    assert _linear_weight_bytes(quantized) < max_ratio * _linear_weight_bytes(full)


def test_only_block_linears_are_quantized():
    (stage,) = _stages(QWEN3_TIED, "nf4", 1)
    block_linears = [m for m in stage.base.layers.modules() if isinstance(m, torch.nn.Linear)]
    assert block_linears and all(isinstance(m, bnb.nn.Linear4bit) for m in block_linears)
    # lm_head и эмбеддинги остаются в исходной точности (lm_head связан с эмбеддингами).
    assert type(stage.model.lm_head) is torch.nn.Linear
    assert stage.model.lm_head.weight.dtype == torch.float32
    assert not any(p.is_meta for p in stage.model.parameters())


def test_quantized_logits_stay_close_for_int8():
    (full,) = _stages(LLAMA, None, 1)
    (quantized,) = _stages(LLAMA, "int8", 1)
    a = full.handle({"session_id": "a", "start_pos": 0, "input_ids": PROMPT})["logits"]
    b = quantized.handle({"session_id": "b", "start_pos": 0, "input_ids": PROMPT})["logits"]
    assert ((a - b).norm() / a.norm()).item() < 0.05


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="quantize must be one of"):
        load_stage(LLAMA, 0, 0, device="cpu", quantize="int3")
