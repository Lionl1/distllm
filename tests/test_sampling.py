import pytest
import torch

from distllm.hf import PipelineGenerator, SamplingParams, load_stage, sample_next

transformers = pytest.importorskip("transformers")
MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])


def test_greedy_cases_are_argmax():
    logits = torch.tensor([0.1, 3.0, 2.0, -1.0])
    assert sample_next(logits, SamplingParams()) == 1
    assert sample_next(logits, SamplingParams(temperature=1.5, top_k=1)) == 1


def test_top_k_and_top_p_restrict_candidates():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    g = torch.Generator().manual_seed(0)
    picks_k = {sample_next(logits, SamplingParams(temperature=1.0, top_k=2), g) for _ in range(300)}
    picks_p_low = {sample_next(logits, SamplingParams(temperature=1.0, top_p=0.5), g) for _ in range(300)}
    picks_p_high = {sample_next(logits, SamplingParams(temperature=1.0, top_p=0.85), g) for _ in range(300)}
    picks_all = {sample_next(logits, SamplingParams(temperature=1.0), g) for _ in range(300)}
    assert picks_k == {0, 1}
    assert picks_p_low == {0}
    assert picks_p_high == {0, 1}
    assert picks_all == {0, 1, 2}


@pytest.mark.parametrize("bad", [dict(temperature=-1), dict(top_k=0), dict(top_p=0), dict(top_p=1.5)])
def test_invalid_params(bad):
    with pytest.raises(ValueError):
        SamplingParams(**bad)


def _reference_sampling(params, steps):
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    rng = params.generator()
    seq, out = PROMPT, []
    with torch.inference_mode():
        for _ in range(steps):
            token = sample_next(model(seq).logits[0, -1], params, rng)
            out.append(token)
            seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
    return out


@pytest.mark.asyncio
async def test_seeded_pipeline_sampling_matches_full_model_and_is_reproducible():
    stages = {f"s{i}": load_stage(MODEL, i, i, device="cpu", dtype="float32") for i in range(2)}

    async def call(role, payload):
        return stages[role].handle(payload)

    params = SamplingParams(temperature=0.8, top_k=50, top_p=0.95, seed=1234)
    generator = PipelineGenerator(call, list(stages))
    first = await generator.generate(PROMPT, 12, sampling=params)
    second = await generator.generate(PROMPT, 12, sampling=params)
    other_seed = await generator.generate(PROMPT, 12, sampling=SamplingParams(temperature=0.8, seed=99))

    assert first.token_ids == second.token_ids
    assert first.token_ids == _reference_sampling(params, 12)
    assert other_seed.token_ids != first.token_ids
