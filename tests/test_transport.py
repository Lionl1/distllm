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
