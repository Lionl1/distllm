import pytest
import torch

from distllm.hf import load_stage

transformers = pytest.importorskip("transformers")
MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "yujiepan/qwen3-tiny-random",
]
PROMPTS = [
    torch.tensor([[1, 5, 9, 13, 21, 34, 55]]),
    torch.tensor([[2, 4, 8]]),
    torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]]),
]


def _pipeline(model_id):
    stages = [load_stage(model_id, i, i, device="cpu", dtype="float32") for i in range(2)]
    for stage in stages:
        stage.min_batch = 2  # batch even small groups on CPU to exercise the batched path
    return stages


def _step(stages, payloads, batched):
    """One pipeline step for several sessions; returns last-stage results."""
    for stage in stages:
        if batched:
            results = stage.handle_batch(payloads)
            assert not any(isinstance(r, Exception) for r in results), results
        else:
            results = [stage.handle(p) for p in payloads]
        payloads = results
    return payloads


@pytest.mark.parametrize("model_id", MODELS)
def test_batched_decode_equals_one_by_one(model_id):
    single, batched = _pipeline(model_id), _pipeline(model_id)
    sequences = [p.clone() for p in PROMPTS]
    for stages in (single, batched):  # prefill each session separately
        for i, prompt in enumerate(PROMPTS):
            _step(stages, [{"session_id": f"s{i}", "start_pos": 0, "input_ids": prompt}], batched=False)

    next_tokens = [None] * len(PROMPTS)
    for i, prompt in enumerate(PROMPTS):
        out = _step(single, [{"session_id": f"s{i}", "start_pos": 0, "input_ids": prompt}], batched=False)
        next_tokens[i] = int(out[0]["logits"].argmax())
    for _ in range(6):
        payloads = [
            {"session_id": f"s{i}", "start_pos": sequences[i].shape[1], "input_ids": torch.tensor([[t]])}
            for i, t in enumerate(next_tokens)
        ]
        expected = _step(single, payloads, batched=False)
        got = _step(batched, payloads, batched=True)
        for e, g in zip(expected, got):
            torch.testing.assert_close(g["logits"], e["logits"], atol=1e-4, rtol=1e-4)
        for i, t in enumerate(next_tokens):
            sequences[i] = torch.cat([sequences[i], torch.tensor([[t]])], dim=1)
        next_tokens = [int(e["logits"].argmax()) for e in expected]
    assert [s.cached_length(f"s{i}") for s in batched for i in range(3)] == [
        seq.shape[1] for _ in batched for seq in sequences
    ]


def test_mixed_batch_falls_back_for_prefill_close_and_misses():
    stages = _pipeline(MODELS[0])
    first = stages[0]
    first.handle({"session_id": "a", "start_pos": 0, "input_ids": PROMPTS[0]})
    first.handle({"session_id": "b", "start_pos": 0, "input_ids": PROMPTS[1]})
    results = first.handle_batch([
        {"session_id": "a", "start_pos": 7, "input_ids": torch.tensor([[1]])},   # decode
        {"session_id": "b", "start_pos": 3, "input_ids": torch.tensor([[1]])},   # decode
        {"session_id": "c", "start_pos": 0, "input_ids": PROMPTS[2]},           # prefill
        {"session_id": "d", "start_pos": 5, "input_ids": torch.tensor([[1]])},   # unknown: miss
        {"session_id": "e", "close": True},                                     # close
    ])
    assert "hidden_states" in results[0] and "hidden_states" in results[1]
    assert results[2]["hidden_states"].shape[1] == PROMPTS[2].shape[1]
    assert results[3]["cache_miss"]
    assert results[4] == {"closed": True}
    assert [first.cached_length(s) for s in "abc"] == [8, 4, 11]


def test_error_in_batch_is_reported_per_request_and_drops_sessions(monkeypatch):
    stage = load_stage(MODELS[0], 0, 1, device="cpu", dtype="float32")
    stage.min_batch = 2
    for sid, prompt in (("a", PROMPTS[0]), ("b", PROMPTS[1])):
        stage.handle({"session_id": sid, "start_pos": 0, "input_ids": prompt})

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(stage, "base", boom)
    results = stage.handle_batch([
        {"session_id": "a", "start_pos": 7, "input_ids": torch.tensor([[1]])},
        {"session_id": "b", "start_pos": 3, "input_ids": torch.tensor([[1]])},
    ])
    assert all(isinstance(r, RuntimeError) for r in results)
    assert stage.cached_length("a") == stage.cached_length("b") == 0


def test_cpu_stage_batches_only_from_four_sessions(monkeypatch):
    stage = load_stage(MODELS[0], 0, 1, device="cpu", dtype="float32")
    assert stage.min_batch == 4
    calls = []
    monkeypatch.setattr(stage, "_decode_batch", lambda ps: calls.append(len(ps)) or [{} for _ in ps])
    for n in (2, 4):
        for i in range(n):
            stage.handle({"session_id": f"n{n}-{i}", "start_pos": 0, "input_ids": PROMPTS[0]})
        stage.handle_batch([
            {"session_id": f"n{n}-{i}", "start_pos": 7, "input_ids": torch.tensor([[1]])} for i in range(n)
        ])
    assert calls == [4]
