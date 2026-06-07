import msgpack
import msgpack_numpy as m
import numpy as np
import torch

# Patches msgpack to support numpy arrays
m.patch()

def pack_tensor(tensor: torch.Tensor) -> bytes:
    """Serializes a torch tensor to msgpack bytes, supporting bfloat16."""
    is_bf16 = tensor.dtype == torch.bfloat16
    if is_bf16:
        arr = tensor.to(torch.float32).detach().cpu().numpy()
    else:
        arr = tensor.detach().cpu().numpy()
    return msgpack.packb({"arr": arr, "is_bf16": is_bf16}, use_bin_type=True)

def unpack_tensor(data: bytes) -> torch.Tensor:
    """Deserializes msgpack bytes to a torch tensor."""
    unpacked = msgpack.unpackb(data, raw=False)
    if isinstance(unpacked, dict) and "arr" in unpacked:
        arr = unpacked["arr"]
        is_bf16 = unpacked.get("is_bf16", False)
        tensor = torch.from_numpy(arr.copy())
        if is_bf16:
            return tensor.to(torch.bfloat16)
        return tensor
    # Backward compatibility with older formats (direct numpy array serialization)
    return torch.from_numpy(unpacked.copy())

def pack_payload(payload: dict) -> bytes:
    """Packs a dictionary into msgpack bytes, handling nested tensors in one pass."""
    def encode(obj):
        if isinstance(obj, torch.Tensor):
            is_bf16 = obj.dtype == torch.bfloat16
            if is_bf16:
                arr = obj.to(torch.float32).detach().cpu().numpy()
            else:
                arr = obj.detach().cpu().numpy()
            return {"__tensor__": True, "arr": arr, "is_bf16": is_bf16}
        return obj

    return msgpack.packb(payload, default=encode, use_bin_type=True)

def unpack_payload(data: bytes) -> dict:
    """Unpacks msgpack bytes into a dictionary, reconstructing tensors."""
    def decode(obj):
        if isinstance(obj, dict) and "__tensor__" in obj:
            if "arr" in obj:
                arr = obj["arr"]
                is_bf16 = obj.get("is_bf16", False)
                tensor = torch.from_numpy(arr.copy())
                if is_bf16:
                    return tensor.to(torch.bfloat16)
                return tensor
            else:
                # Backward compatibility
                return unpack_tensor(obj["__tensor__"])
        return obj

    return msgpack.unpackb(data, object_hook=decode, raw=False)
