import os
import pickle

import msgpack
import pytest
import torch
import numpy as np
from distllm.transport import pack_tensor, unpack_tensor, pack_payload, unpack_payload

def test_tensor_serialization():
    # Test simple tensor
    original = torch.tensor([1.0, 2.0, 3.0])
    packed = pack_tensor(original)
    unpacked = unpack_tensor(packed)
    
    assert torch.equal(original, unpacked)
    assert unpacked.dtype == torch.float32

def test_payload_serialization():
    # Test dictionary with nested tensors
    payload = {
        "id": "test-123",
        "data": torch.randn(2, 2),
        "nested": {
            "val": torch.tensor([42])
        }
    }
    
    packed = pack_payload(payload)
    unpacked = unpack_payload(packed)
    
    assert unpacked["id"] == "test-123"
    assert torch.allclose(unpacked["data"], payload["data"])
    assert torch.equal(unpacked["nested"]["val"], payload["nested"]["val"])

def test_bfloat16_serialization():
    # Test direct bf16 tensor
    original = torch.tensor([1.5, 2.5, 3.5], dtype=torch.bfloat16)
    packed = pack_tensor(original)
    unpacked = unpack_tensor(packed)
    
    assert torch.equal(original, unpacked)
    assert unpacked.dtype == torch.bfloat16

    # Test bf16 tensor inside a payload dict
    payload = {
        "x": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    }
    packed = pack_payload(payload)
    unpacked = unpack_payload(packed)
    
    assert torch.equal(payload["x"], unpacked["x"])
    assert unpacked["x"].dtype == torch.bfloat16

def test_half_precision_stays_half_size():
    hidden = torch.randn(1, 64, 896, dtype=torch.float16)
    packed = pack_payload({"hidden_states": hidden})
    unpacked = unpack_payload(packed)

    assert unpacked["hidden_states"].dtype == torch.float16
    assert torch.equal(unpacked["hidden_states"], hidden)
    # 2 bytes per value plus a small header — not 4 bytes as with float32.
    assert len(packed) < hidden.numel() * 2 + 256

def test_bfloat16_not_inflated_to_float32():
    hidden = torch.randn(1, 64, 896, dtype=torch.bfloat16)
    packed = pack_payload({"h": hidden})

    assert len(packed) < hidden.numel() * 2 + 256
    assert torch.equal(unpack_payload(packed)["h"], hidden)

def test_dtypes_shapes_and_edge_cases():
    cases = [
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.tensor([True, False, True]),
        torch.tensor(3.5),  # 0-dim
        torch.empty(0, 4),
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 2),  # non-contiguous
    ]
    for original in cases:
        restored = unpack_payload(pack_payload({"t": original}))["t"]
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape
        assert torch.equal(restored, original)
        restored.add_(0) if restored.is_floating_point() else None  # writable

def test_legacy_format_still_decodes():
    legacy = msgpack_packb_legacy({"x": np.array([1.5, 2.5], dtype=np.float32)})
    restored = unpack_payload(legacy)["x"]
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, torch.tensor([1.5, 2.5], dtype=torch.bfloat16))

def numpy_wire(arr):
    """An ndarray the way msgpack-numpy 0.4.8 put it on the wire."""
    return {b"nd": True, b"type": arr.dtype.str, b"kind": b"", b"shape": list(arr.shape), b"data": arr.tobytes()}


def msgpack_packb_legacy(values):
    return msgpack.packb(
        {k: {"__tensor__": True, "arr": numpy_wire(v), "is_bf16": True} for k, v in values.items()},
        use_bin_type=True,
    )


def test_legacy_bare_array_still_decodes():
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    restored = unpack_tensor(msgpack.packb(numpy_wire(arr), use_bin_type=True))
    assert torch.equal(restored, torch.from_numpy(arr))
    restored.add_(1)  # writable copy


def test_object_arrays_are_not_unpickled(monkeypatch):
    # msgpack-numpy decoded this map with pickle.loads: any peer could run code on the
    # relay, on every worker and on the clients.
    monkeypatch.delenv("DISTLLM_PWNED", raising=False)

    class Payload:
        def __reduce__(self):
            return (exec, ("import os; os.environ['DISTLLM_PWNED'] = '1'",))

    evil = {b"nd": True, b"type": [], b"kind": b"O", b"shape": [1], b"data": pickle.dumps(Payload())}
    bodies = [
        msgpack.packb(evil, use_bin_type=True),
        msgpack.packb({"hidden_states": evil}, use_bin_type=True),
        msgpack.packb({"x": {"__tensor__": True, "arr": evil, "is_bf16": True}}, use_bin_type=True),
    ]
    for body in bodies:
        with pytest.raises(ValueError):
            unpack_payload(body)
        with pytest.raises(ValueError):
            unpack_tensor(body)
    assert "DISTLLM_PWNED" not in os.environ


@pytest.mark.parametrize("descr", ["O", "V8", "S4", "U4", "M8[s]", "f4,f4", "no-such-dtype", 7])
def test_legacy_array_accepts_only_numbers(descr):
    body = msgpack.packb({b"nd": True, b"type": descr, b"kind": b"", b"shape": [1], b"data": b"\x00" * 8}, use_bin_type=True)
    with pytest.raises(ValueError):
        unpack_payload(body)


def test_msgpack_is_left_unpatched():
    # A global patch would bring the unpickling decoder back for every msgpack call.
    assert msgpack.unpackb.__module__.split(".")[0] == "msgpack"
    assert msgpack.packb.__module__.split(".")[0] == "msgpack"
    with pytest.raises(TypeError):
        msgpack.packb(np.zeros(2))


def test_numpy_values_are_packed_as_tensors_and_scalars():
    restored = unpack_payload(pack_payload({"a": np.arange(4, dtype=np.float32)[::2], "n": np.int64(7), "f": np.float32(0.5)}))
    assert torch.equal(restored["a"], torch.tensor([0.0, 2.0]))
    assert restored["n"] == 7 and isinstance(restored["n"], int)
    assert restored["f"] == 0.5 and isinstance(restored["f"], float)
