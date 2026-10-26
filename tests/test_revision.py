import re

import pytest
import torch

transformers = pytest.importorskip("transformers")

from distllm.hf import load_stage, resolve_revision  # noqa: E402

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PROMPT = torch.tensor([[1, 5, 9, 13]])


def test_resolve_revision_gives_a_commit_hash(tmp_path):
    sha = resolve_revision(MODEL)
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert resolve_revision(MODEL, sha) == sha
    assert resolve_revision(str(tmp_path)) is None  # local directory: nothing to pin


def test_pinned_stage_loads_the_same_weights():
    sha = resolve_revision(MODEL)
    pinned = load_stage(MODEL, 0, 1, device="cpu", dtype="float32", revision=sha)
    plain = load_stage(MODEL, 0, 1, device="cpu", dtype="float32")
    assert sha[:12] in pinned.name
    a = pinned.handle({"session_id": "a", "start_pos": 0, "input_ids": PROMPT})["logits"]
    b = plain.handle({"session_id": "b", "start_pos": 0, "input_ids": PROMPT})["logits"]
    assert torch.equal(a, b)


def test_revision_reaches_every_download(monkeypatch):
    import huggingface_hub

    seen = []
    real_download = huggingface_hub.hf_hub_download

    def download(repo_id, filename, **kwargs):
        seen.append((filename, kwargs.get("revision")))
        return real_download(repo_id, filename, **kwargs)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    sha = resolve_revision(MODEL)
    load_stage(MODEL, 0, 0, device="cpu", dtype="float32", revision=sha)
    assert seen and all(revision == sha for _, revision in seen)


def test_unknown_revision_fails():
    with pytest.raises(ValueError, match="Revision '0+' not found"):
        load_stage(MODEL, 0, 0, device="cpu", revision="0" * 40)


def test_pinned_cached_model_loads_offline(monkeypatch):
    """A pinned model already in the cache must load without the network."""
    sha = resolve_revision(MODEL)
    load_stage(MODEL, 0, 0, device="cpu", dtype="float32", revision=sha)  # fill the cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)

    def no_network(*args, **kwargs):
        raise AssertionError("the Hub API must not be called offline")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", no_network)
    stage = load_stage(MODEL, 0, 0, device="cpu", dtype="float32", revision=sha)
    assert stage.handle({"session_id": "s", "start_pos": 0, "input_ids": PROMPT})["hidden_states"].shape[1] == 4
