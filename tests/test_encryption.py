import asyncio

import httpx
import pytest
import torch

pytest.importorskip("cryptography")

from distllm import Cluster, PayloadDecryptionError, Worker  # noqa: E402
from distllm import relay  # noqa: E402
from distllm.crypto import MAGIC, PayloadCipher, cipher_from, generate_key, is_sealed  # noqa: E402

KEY = generate_key()


def test_seal_open_roundtrip_and_fresh_nonces():
    cipher = PayloadCipher(KEY)
    a, b = cipher.seal(b"hidden states"), cipher.seal(b"hidden states")
    assert a.startswith(MAGIC) and a != b  # random nonce per message
    assert cipher.open(a) == cipher.open(b) == b"hidden states"


def test_tampered_wrong_key_and_plaintext_are_refused():
    cipher = PayloadCipher(KEY)
    sealed = bytearray(cipher.seal(b"payload"))
    sealed[-1] ^= 1
    with pytest.raises(PayloadDecryptionError):
        cipher.open(bytes(sealed))
    with pytest.raises(PayloadDecryptionError):
        PayloadCipher(generate_key()).open(cipher.seal(b"payload"))
    with pytest.raises(PayloadDecryptionError, match="unencrypted"):
        cipher.open(b"\x81\xa1a\x01")  # plain msgpack: a relay could inject it


def test_keys_and_passphrases(monkeypatch):
    assert len(KEY) == 43  # 32 bytes, base64url without padding
    # The same passphrase gives the same key on every node.
    sealed = PayloadCipher("correct horse battery staple").seal(b"x")
    assert PayloadCipher("correct horse battery staple").open(sealed) == b"x"
    with pytest.raises(ValueError, match="too short"):
        PayloadCipher("short")
    monkeypatch.delenv("DISTLLM_PAYLOAD_KEY", raising=False)
    assert cipher_from(None) is None
    monkeypatch.setenv("DISTLLM_PAYLOAD_KEY", KEY)
    assert cipher_from(None) is not None


def _reset_relay():
    for state in (relay.nodes, relay.tasks, relay.pending_queues, relay.worker_queues, relay.affinity):
        state.clear()


@pytest.fixture()
def asgi():
    _reset_relay()  # earlier tests leave tasks behind
    yield httpx.ASGITransport(app=relay.app)
    _reset_relay()


async def _run(asgi, worker_key, client_key, payload, route=None):
    """Two chained echo stages behind the in-process relay; returns (result or error, relay bodies)."""
    workers = []
    for role in ("s0", "s1"):
        def stage(p, role=role):
            return {**p, "seen_by": p.get("seen_by", []) + [role]}

        w = Worker(role, "http://relay", node_id=role, payload_key=worker_key)
        w.client = httpx.AsyncClient(transport=asgi, base_url="http://relay")
        workers.append((w, asyncio.create_task(w.run(stage))))
    cluster = Cluster("http://relay", payload_key=client_key)
    cluster.client = httpx.AsyncClient(transport=asgi, base_url="http://relay")
    try:
        if route:
            result = await cluster.call_route(["s0", "s1"], payload, timeout=10)
        else:
            result = await cluster.call("s0", payload, timeout=10)
    except Exception as e:
        result = e
    bodies = [t.payload for t in relay.tasks.values()] + [
        t.result for t in relay.tasks.values() if getattr(t, "result", None)
    ]
    for _, task in workers:
        task.cancel()
    await asyncio.gather(*(t for _, t in workers), return_exceptions=True)
    return result, bodies


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [False, True])
async def test_relay_only_sees_ciphertext(asgi, route):
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    result, bodies = await _run(asgi, KEY, KEY, {"session_id": "s", "hidden": hidden}, route=route)
    assert torch.equal(result["hidden"], hidden)
    assert result["seen_by"] == (["s0", "s1"] if route else ["s0"])
    assert bodies and all(is_sealed(b) for b in bodies)
    assert not any(hidden.numpy().tobytes() in b for b in bodies)


@pytest.mark.asyncio
async def test_key_mismatch_fails_the_task_instead_of_hanging(asgi):
    result, _ = await _run(asgi, generate_key(), KEY, {"x": 1})
    assert isinstance(result, RuntimeError) and "PayloadDecryptionError" in str(result)
    # A worker without the key refuses ciphertext; a client with the key refuses plaintext.
    result, _ = await _run(asgi, None, KEY, {"x": 1})
    assert "set DISTLLM_PAYLOAD_KEY" in str(result)
    result, _ = await _run(asgi, KEY, None, {"x": 1})
    assert "unencrypted" in str(result)
