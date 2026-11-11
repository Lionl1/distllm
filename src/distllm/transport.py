import msgpack
import numpy as np
import torch

# msgpack-numpy is not used here on purpose. Its decoder unpickles object arrays
# (`pickle.loads` on bytes from the network), and `msgpack_numpy.patch()` put that decoder
# into every `msgpack.unpackb` of the process, so any peer could run code on the relay,
# the workers and the clients. The old numpy wire format is read by `_decode_legacy_array`.

# Tensors travel as raw bytes in their own dtype (fp16/bf16 stay 2 bytes per value).
# The old format converted everything through numpy, which turned bf16 into float32
# and doubled the traffic; it is still accepted when decoding.
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "float16", "bfloat16", "float32", "float64",
        "int8", "uint8", "int16", "int32", "int64", "bool",
    )
}


def _encode_tensor(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().contiguous().cpu()
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _DTYPES:
        raise TypeError(f"Unsupported tensor dtype for transport: {tensor.dtype}")
    data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    return {"__tensor__": True, "dtype": dtype, "shape": list(tensor.shape), "data": data}


def _decode_legacy_array(obj: dict) -> np.ndarray:
    """A numpy array as msgpack-numpy wrote it: {b"nd", b"type", b"kind", b"shape", b"data"}.

    Only plain numbers: b"kind" marks pickled objects (b"O") and structured records (b"V"),
    which distllm never sent.
    """
    descr = obj.get(b"type")
    if isinstance(descr, bytes):
        descr = descr.decode("ascii", "replace")
    if obj.get(b"nd") is not True or obj.get(b"kind") or not isinstance(descr, str):
        raise ValueError("Unsupported numpy array in payload")
    try:
        dtype = np.dtype(descr)
    except TypeError as exc:
        raise ValueError(f"Unsupported numpy dtype in payload: {descr!r}") from exc
    if dtype.kind not in "biuf":
        raise ValueError(f"Unsupported numpy dtype in payload: {descr!r}")
    try:
        return np.frombuffer(obj[b"data"], dtype=dtype).reshape(obj[b"shape"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed numpy array in payload") from exc


def _decode_legacy(obj: dict):
    return _decode_legacy_array(obj) if b"nd" in obj else obj


def _decode_tensor(obj: dict) -> torch.Tensor:
    if "data" in obj:
        dtype = _DTYPES.get(obj["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported tensor dtype in payload: {obj['dtype']!r}")
        shape = tuple(obj["shape"])
        if not obj["data"]:
            return torch.empty(shape, dtype=dtype)
        # bytearray: torch.frombuffer needs a writable buffer to return a normal tensor.
        return torch.frombuffer(bytearray(obj["data"]), dtype=dtype).reshape(shape)
    # Legacy format: float32 numpy array plus a bf16 flag.
    tensor = torch.from_numpy(np.asarray(obj["arr"]).copy())
    return tensor.to(torch.bfloat16) if obj.get("is_bf16", False) else tensor


def pack_tensor(tensor: torch.Tensor) -> bytes:
    """Serializes a torch tensor to msgpack bytes, preserving its dtype."""
    return msgpack.packb(_encode_tensor(tensor), use_bin_type=True)


def unpack_tensor(data: bytes) -> torch.Tensor:
    """Deserializes msgpack bytes to a torch tensor."""
    unpacked = msgpack.unpackb(data, object_hook=_decode_legacy, raw=False)
    if isinstance(unpacked, dict) and ("data" in unpacked or "arr" in unpacked):
        return _decode_tensor(unpacked)
    # Backward compatibility with older formats (direct numpy array serialization)
    return torch.from_numpy(unpacked.copy())


def pack_payload(payload: dict) -> bytes:
    """Packs a dictionary into msgpack bytes, handling nested tensors in one pass."""
    def encode(obj):
        if isinstance(obj, torch.Tensor):
            return _encode_tensor(obj)
        # numpy values used to be packed by msgpack-numpy; arrays now arrive as tensors.
        if isinstance(obj, np.ndarray):
            return _encode_tensor(torch.from_numpy(np.ascontiguousarray(obj).copy()))
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    return msgpack.packb(payload, default=encode, use_bin_type=True)


def unpack_payload(data: bytes) -> dict:
    """Unpacks msgpack bytes into a dictionary, reconstructing tensors."""
    def decode(obj):
        if b"nd" in obj:
            return _decode_legacy_array(obj)
        if "__tensor__" in obj:
            if "data" in obj or "arr" in obj:
                return _decode_tensor(obj)
            # Backward compatibility
            return unpack_tensor(obj["__tensor__"])
        return obj

    return msgpack.unpackb(data, object_hook=decode, raw=False)
