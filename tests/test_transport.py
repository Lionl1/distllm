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

def msgpack_packb_legacy(values):
    import msgpack
    return msgpack.packb(
        {k: {"__tensor__": True, "arr": v, "is_bf16": True} for k, v in values.items()},
        use_bin_type=True,
    )
