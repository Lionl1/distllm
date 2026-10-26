import pytest
import torch

from distllm.hf import AuditConfig, PipelineGenerator, VerificationError, load_stage

transformers = pytest.importorskip("transformers")
MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = torch.tensor([[1, 5, 9, 13, 21, 34, 55]])


def _stages(prefix, dtype="float32"):
    return {f"{prefix}-{i}": load_stage(MODEL, i, i, device="cpu", dtype=dtype) for i in range(2)}


def _caller(*groups, tamper=None):
    stages = {}
    for group in groups:
        stages.update(group)

    async def call(role, payload):
        response = stages[role].handle(payload)
        if tamper is not None and not payload.get("close"):
            response = tamper(role, response)
        return response

    return call


@pytest.mark.asyncio
async def test_honest_pipeline_passes_every_audit():
    main, audit = _stages("stage"), _stages("audit")
    generator = PipelineGenerator(
        _caller(main, audit), list(main), audit=AuditConfig(roles=list(audit), rate=1.0)
    )
    result = await generator.generate(PROMPT, 6)
    reference = await PipelineGenerator(_caller(main), list(main)).generate(PROMPT, 6)
    assert result.audits == 6
    assert result.token_ids == reference.token_ids


@pytest.mark.asyncio
async def test_lower_precision_does_not_raise_false_alarms():
    main, audit = _stages("stage", dtype="float16"), _stages("audit", dtype="float32")
    generator = PipelineGenerator(
        _caller(main, audit), list(main), audit=AuditConfig(roles=list(audit), rate=1.0)
    )
    assert (await generator.generate(PROMPT, 4)).audits == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_role", ["stage-0", "stage-1"])
async def test_tampering_stage_is_located(bad_role):
    main, audit = _stages("stage"), _stages("audit")
    torch.manual_seed(0)

    def tamper(role, response):
        if role != bad_role:
            return response
        if "hidden_states" in response:  # нечестная промежуточная стадия
            return {**response, "hidden_states": torch.randn_like(response["hidden_states"]) * 10}
        return {"logits": torch.randn_like(response["logits"])}  # нечестная последняя стадия

    generator = PipelineGenerator(
        _caller(main, audit, tamper=tamper), list(main), audit=AuditConfig(roles=list(audit), rate=1.0)
    )
    with pytest.raises(VerificationError) as err:
        await generator.generate(PROMPT, 4)
    assert err.value.suspects == [bad_role]
    assert err.value.similarity < 0.99


@pytest.mark.asyncio
async def test_rate_zero_never_audits_and_roles_must_mirror():
    main, audit = _stages("stage"), _stages("audit")
    generator = PipelineGenerator(_caller(main, audit), list(main), audit=AuditConfig(roles=list(audit), rate=0))
    assert (await generator.generate(PROMPT, 3)).audits == 0
    with pytest.raises(ValueError):
        PipelineGenerator(_caller(main), list(main), audit=AuditConfig(roles=["only-one"]))
