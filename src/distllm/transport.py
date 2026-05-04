import msgpack
import msgpack_numpy as m
import numpy as np
import torch
import base64

# Patches msgpack to support numpy arrays
m.patch()

def pack_tensor(tensor: torch.Tensor) -> bytes:
    """Serializes a torch tensor to msgpack bytes."""
    arr = tensor.detach().cpu().numpy()
    return msgpack.packb(arr, use_bin_type=True)

def unpack_tensor(data: bytes) -> torch.Tensor:
    """Deserializes msgpack bytes to a torch tensor."""
    arr = msgpack.unpackb(data, raw=False)
    return torch.from_numpy(arr.copy())

def pack_payload(payload: dict) -> bytes:
    """Packs a dictionary into msgpack bytes, handling nested tensors."""
    def encode(obj):
        if isinstance(obj, torch.Tensor):
            return {"__tensor__": pack_tensor(obj)}
        return obj

    return msgpack.packb(payload, default=encode, use_bin_type=True)

def unpack_payload(data: bytes) -> dict:
    """Unpacks msgpack bytes into a dictionary, reconstructing tensors."""
    def decode(obj):
        if isinstance(obj, dict) and "__tensor__" in obj:
            return unpack_tensor(obj["__tensor__"])
        return obj

    return msgpack.unpackb(data, object_hook=decode, raw=False)
