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
    real_list, real_download = huggingface_hub.list_repo_files, huggingface_hub.hf_hub_download

    def list_files(repo_id, **kwargs):
        seen.append(("list", kwargs.get("revision")))
        return real_list(repo_id, **kwargs)

    def download(repo_id, filename, **kwargs):
        seen.append((filename, kwargs.get("revision")))
        return real_download(repo_id, filename, **kwargs)

    monkeypatch.setattr(huggingface_hub, "list_repo_files", list_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    sha = resolve_revision(MODEL)
    load_stage(MODEL, 0, 0, device="cpu", dtype="float32", revision=sha)
    assert seen and all(revision == sha for _, revision in seen)


def test_unknown_revision_fails():
    with pytest.raises(ValueError, match="Revision '0+' not found"):
        load_stage(MODEL, 0, 0, device="cpu", revision="0" * 40)
