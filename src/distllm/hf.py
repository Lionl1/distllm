"""Pipeline-parallel inference for Hugging Face decoder-only models.

Each worker holds a contiguous slice of transformer layers (a *stage*) and a
per-session KV cache. The client drives token-by-token generation through the
stages; if a stage lost its cache (restart, replacement node), it answers with
``cache_miss`` and the client replays the prefix to rebuild it.

A stage is built as a *smaller model of the same architecture*: the config gets
``num_hidden_layers = layer_end - layer_start + 1``, so the model's own forward
computes rotary embeddings, the causal mask and cache indexing correctly for any
architecture transformers supports in that form (Llama, Qwen2/3, Mistral, ...).
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import torch

_LAYER_RE = re.compile(r"(^|\.)layers\.(\d+)\.")


# --------------------------------------------------------------------------- #
# Loading only the weights a stage needs
# --------------------------------------------------------------------------- #

def _resolve_device(device: Optional[str]) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(dtype: Optional[str], device: torch.device) -> torch.dtype:
    if dtype is None or dtype == "auto":
        # fp16 matmuls are slow or unsupported on CPU.
        return torch.float16 if device.type == "cuda" else torch.float32
    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}") from exc


def _weight_files(model_id: str, needed: set[str], token: Optional[str]) -> list[str]:
    """Return local paths of only those checkpoint files that contain needed tensors."""
    from huggingface_hub import hf_hub_download, list_repo_files

    local = os.path.isdir(model_id)
    files = os.listdir(model_id) if local else list_repo_files(model_id, token=token)

    def fetch(name: str) -> str:
        return os.path.join(model_id, name) if local else hf_hub_download(model_id, name, token=token)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name in files:
            with open(fetch(index_name), encoding="utf-8") as fh:
                weight_map: dict[str, str] = json.load(fh)["weight_map"]
            shards = sorted({weight_map[k] for k in needed if k in weight_map})
            return [fetch(s) for s in shards]

    for single in ("model.safetensors", "pytorch_model.bin"):
        if single in files:
            return [fetch(single)]
    raise FileNotFoundError(f"No safetensors/bin checkpoint found for {model_id}")


def _read_tensors(path: str, wanted: dict[str, str]) -> dict[str, torch.Tensor]:
    """Read checkpoint tensors {stage_name: checkpoint_name} that exist in this file."""
    out: dict[str, torch.Tensor] = {}
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as fh:
            keys = set(fh.keys())
            for name, src in wanted.items():
                if src in keys:
                    out[name] = fh.get_tensor(src)
        return out

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for name, src in wanted.items():
        if src in state:
            out[name] = state[src]
    return out


def _stage_config(config: Any, layer_start: int, layer_end: int) -> Any:
    stage_config = copy.deepcopy(config)
    stage_config.num_hidden_layers = layer_end - layer_start + 1
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        stage_config.layer_types = list(layer_types[layer_start : layer_end + 1])
    return stage_config


def load_stage(
    model_id: str,
    layer_start: int,
    layer_end: int,
    *,
    device: Optional[str] = "auto",
    dtype: Optional[str] = "auto",
    token: Optional[str] = None,
    max_sessions: int = 8,
    session_ttl: float = 600.0,
) -> "PipelineStage":
    """Build a stage holding layers [layer_start, layer_end] (inclusive) of ``model_id``.

    Only the tensors of this stage are read from disk, so a node needs memory for
    its own slice, not for the whole model.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, token=token)
    n_layers = config.num_hidden_layers
    if not (0 <= layer_start <= layer_end < n_layers):
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}] for a model with {n_layers} layers"
        )
    is_first = layer_start == 0
    is_last = layer_end == n_layers - 1
    tied = bool(getattr(config, "tie_word_embeddings", False))

    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype, torch_device)
    stage_config = _stage_config(config, layer_start, layer_end)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(stage_config)

    # stage parameter name -> checkpoint tensor name
    wanted: dict[str, str] = {}
    for name in model.state_dict():
        match = _LAYER_RE.search(name)
        if match:
            index = int(match.group(2)) + layer_start
            wanted[name] = f"{name[: match.start(2)]}{index}{name[match.end(2):]}"
        elif "embed_tokens" in name:
            if is_first or (is_last and tied):
                wanted[name] = name
        elif is_last:
            wanted[name] = name  # final norm, lm_head

    tensors: dict[str, torch.Tensor] = {}
    for path in _weight_files(model_id, set(wanted.values()), token):
        tensors.update(_read_tensors(path, wanted))

    if is_last and "lm_head.weight" in wanted and "lm_head.weight" not in tensors and tied:
        embed_key = next((k for k in tensors if "embed_tokens" in k), None)
        if embed_key is not None:
            tensors["lm_head.weight"] = tensors[embed_key]

    missing = sorted(k for k in wanted if k not in tensors)
    if missing:
        raise ValueError(
            f"Checkpoint of {model_id} has no tensors {missing[:4]}... "
            "Only decoder-only *ForCausalLM checkpoints whose tensor names match the "
            "model's parameter names are supported (multimodal checkpoints are not)."
        )

    tensors = {
        k: (v.to(torch_dtype) if v.is_floating_point() else v).to(torch_device) for k, v in tensors.items()
    }
    model.load_state_dict(tensors, strict=False, assign=True)

    base = getattr(model, model.base_model_prefix)
    if is_last and model.lm_head.weight.is_meta:
        # Tied embeddings: assign=True replaced the shared parameter only on embed_tokens.
        embed_key = next((k for k in tensors if "embed_tokens" in k), None)
        if embed_key is None:
            raise ValueError("lm_head weight not found in checkpoint")
        model.lm_head.weight = torch.nn.Parameter(tensors[embed_key], requires_grad=False)
    # Non-persistent buffers (e.g. rotary inv_freq) are computed in __init__ and stayed on meta.
    if getattr(base, "rotary_emb", None) is not None:
        base.rotary_emb = type(base.rotary_emb)(config=stage_config, device=torch_device)
    if not is_first:
        base.embed_tokens = None
    if not is_last:
        base.norm = torch.nn.Identity()
        model.lm_head = None

    leftover = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta
    ]
    if leftover:
        raise ValueError(f"Unsupported architecture: tensors left uninitialized: {leftover[:4]}")

    model.eval()
    return PipelineStage(
        model=model,
        config=stage_config,
        is_first=is_first,
        is_last=is_last,
        device=torch_device,
        dtype=torch_dtype,
        max_sessions=max_sessions,
        session_ttl=session_ttl,
        name=f"{model_id}[{layer_start}:{layer_end}]",
    )


# --------------------------------------------------------------------------- #
# Stage: runs on a worker, keeps KV cache per session
# --------------------------------------------------------------------------- #

@dataclass
class _Session:
    cache: Any
    length: int
    last_used: float = field(default_factory=time.time)


class PipelineStage:
    """A slice of layers plus a bounded LRU of per-session KV caches.

    Protocol (``handle`` payload):
      - ``session_id``: str, ``start_pos``: int — position of the first new token;
      - ``input_ids`` (first stage) or ``hidden_states`` (other stages) for the new tokens;
      - ``rewind``: True — allow ``start_pos`` below the cached length: the cache is
        cropped to ``start_pos`` first (speculative decoding drops rejected tokens);
      - ``output``: ``"logits"`` (default, last-token logits) or ``"argmax"``
        (greedy token id for *every* new position, for draft verification);
      - ``close``: True — drop the session cache.
    ``start_pos == 0`` always starts a fresh cache. Any other ``start_pos`` must equal
    the cached length (or be lower with ``rewind``), otherwise the stage answers
    ``{"cache_miss": True}``. Intermediate stages return a ready payload for the next
    stage: the same control fields plus ``hidden_states``.
    """

    CONTROL_KEYS = ("session_id", "start_pos", "rewind", "output")

    def __init__(
        self,
        *,
        model: Any,
        config: Any,
        is_first: bool,
        is_last: bool,
        device: torch.device,
        dtype: torch.dtype,
        max_sessions: int = 8,
        session_ttl: float = 600.0,
        name: str = "stage",
    ) -> None:
        self.model = model
        self.base = getattr(model, model.base_model_prefix)
        self.config = config
        self.is_first = is_first
        self.is_last = is_last
        self.device = device
        self.dtype = dtype
        self.max_sessions = max(1, max_sessions)
        self.session_ttl = session_ttl
        self.name = name
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.time()
        for sid in [s for s, sess in self._sessions.items() if now - sess.last_used > self.session_ttl]:
            del self._sessions[sid]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def cached_length(self, session_id: str) -> int:
        with self._lock:
            sess = self._sessions.get(session_id)
            return sess.length if sess else 0

    def drop_sessions(self) -> None:
        """Forget all caches (used to simulate a node restart in tests and benchmarks)."""
        with self._lock:
            self._sessions.clear()

    def handle(self, payload: dict) -> dict:
        from transformers import DynamicCache

        session_id = str(payload.get("session_id") or "")
        if not session_id:
            raise ValueError("payload must contain 'session_id'")

        with self._lock:
            if payload.get("close"):
                self._sessions.pop(session_id, None)
                return {"closed": True}

            start_pos = int(payload.get("start_pos", 0))
            sess = self._sessions.get(session_id)
            if start_pos == 0:
                sess = _Session(cache=DynamicCache(config=self.config), length=0)
            elif sess is not None and payload.get("rewind") and start_pos < sess.length:
                sess.cache.crop(start_pos)
                sess.length = start_pos
            if start_pos != 0 and (sess is None or sess.length != start_pos):
                # stop_chain: the relay must not forward this answer to the next stage.
                return {"cache_miss": True, "cached": sess.length if sess else 0, "stop_chain": True}

            if self.is_first:
                if "input_ids" not in payload:
                    raise ValueError("first stage expects 'input_ids'")
                inputs = {"input_ids": payload["input_ids"].to(self.device)}
                new_tokens = inputs["input_ids"].shape[1]
            else:
                if "hidden_states" not in payload:
                    raise ValueError("stage expects 'hidden_states'")
                hidden = payload["hidden_states"].to(self.device, self.dtype)
                inputs = {"inputs_embeds": hidden}
                new_tokens = hidden.shape[1]

            position_ids = torch.arange(
                start_pos, start_pos + new_tokens, device=self.device
            ).unsqueeze(0)

            with torch.inference_mode():
                out = self.base(
                    **inputs,
                    position_ids=position_ids,
                    past_key_values=sess.cache,
                    use_cache=True,
                )
                hidden = out.last_hidden_state
                if self.is_last and payload.get("output") == "argmax":
                    result = {"argmax": self.model.lm_head(hidden[0]).argmax(dim=-1).cpu()}
                elif self.is_last:
                    result = {"logits": self.model.lm_head(hidden[:, -1, :]).float().cpu()}
                else:
                    # The output is a ready payload for the next stage, so the relay can
                    # forward it as is (chained routes).
                    result = {k: payload[k] for k in self.CONTROL_KEYS if k in payload}
                    result["hidden_states"] = hidden.cpu()

            sess.length = start_pos + new_tokens
            sess.last_used = time.time()
            self._sessions[session_id] = sess
            self._sessions.move_to_end(session_id)
            self._evict()
            return result


# --------------------------------------------------------------------------- #
# Client: token-by-token generation through the stages
# --------------------------------------------------------------------------- #

CallFn = Callable[[str, dict], Awaitable[dict]]
RouteCallFn = Callable[[list, dict], Awaitable[dict]]


@dataclass
class GenerationResult:
    token_ids: list[int]
    prefill_seconds: float
    token_seconds: list[float]
    cache_recoveries: int

    @property
    def tokens_per_second(self) -> float:
        total = sum(self.token_seconds)
        return len(self.token_seconds) / total if total > 0 else 0.0


class CacheRecoveryError(RuntimeError):
    pass


class PipelineGenerator:
    """Greedy generation through pipeline stages.

    ``call(role, payload)`` sends a payload to one stage and returns its response —
    ``Cluster.call`` over the relay, or a direct function in tests.
    ``route_call(roles, payload)`` (optional, e.g. ``Cluster.call_route``) sends the
    payload through all stages in one request: the relay forwards hidden states
    from stage to stage and only the logits come back to the client.
    """

    def __init__(
        self,
        call: CallFn,
        roles: list[str],
        *,
        route_call: Optional[RouteCallFn] = None,
        use_kv_cache: bool = True,
        max_recoveries_per_step: int = 2,
    ) -> None:
        if not roles:
            raise ValueError("at least one stage role is required")
        self.call = call
        self.route_call = route_call
        self.roles = list(roles)
        self.use_kv_cache = use_kv_cache
        self.max_recoveries_per_step = max_recoveries_per_step

    async def _forward(self, session_id: str, start_pos: int, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Run new tokens through all stages; return last-token logits or None on cache miss."""
        payload: dict = {"session_id": session_id, "start_pos": start_pos, "input_ids": input_ids}
        if self.route_call is not None:
            response = await self.route_call(self.roles, payload)
            if response.get("cache_miss"):
                return None
            if "logits" not in response:
                raise RuntimeError("route did not end with logits — check the layer ranges of the roles")
            return response["logits"]
        for role in self.roles:
            response = await self.call(role, payload)
            if response.get("cache_miss"):
                return None
            if "logits" in response:
                return response["logits"]
            payload = {
                "session_id": session_id,
                "start_pos": start_pos,
                "hidden_states": response["hidden_states"],
            }
        raise RuntimeError("last stage did not return logits — check the layer ranges of the roles")

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: Optional[int | list[int]] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> GenerationResult:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape (1, seq_len): batching is not supported")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])

        session_id = uuid.uuid4().hex
        sequence = input_ids.clone()
        new_ids: list[int] = []
        token_seconds: list[float] = []
        prefill_seconds = 0.0
        recoveries = 0
        cached = 0  # tokens the stages already hold for this session

        try:
            for step in range(max_new_tokens):
                started = time.perf_counter()
                if self.use_kv_cache:
                    logits = await self._forward(session_id, cached, sequence[:, cached:])
                    attempts = 0
                    while logits is None:
                        # A stage lost its cache: replay the whole prefix to rebuild every stage.
                        attempts += 1
                        recoveries += 1
                        if attempts > self.max_recoveries_per_step:
                            raise CacheRecoveryError(
                                f"stages keep losing the cache (step {step}, {attempts - 1} replays)"
                            )
                        logits = await self._forward(session_id, 0, sequence)
                else:
                    logits = await self._forward(session_id, 0, sequence)
                    if logits is None:
                        raise RuntimeError("unexpected cache miss with start_pos=0")

                cached = sequence.shape[1]
                next_id = int(torch.argmax(logits[0]).item())
                elapsed = time.perf_counter() - started
                if step == 0:
                    prefill_seconds = elapsed
                else:
                    token_seconds.append(elapsed)

                new_ids.append(next_id)
                sequence = torch.cat([sequence, torch.tensor([[next_id]], dtype=sequence.dtype)], dim=1)
                if on_token is not None:
                    on_token(next_id)
                if next_id in eos:
                    break
        finally:
            await self._close(session_id)

        return GenerationResult(
            token_ids=new_ids,
            prefill_seconds=prefill_seconds,
            token_seconds=token_seconds,
            cache_recoveries=recoveries,
        )

    async def _close(self, session_id: str) -> None:
        for role in self.roles:
            try:
                await self.call(role, {"session_id": session_id, "close": True})
            except Exception:  # noqa: BLE001 — cleanup is best-effort, caches also expire by TTL
                pass


# --------------------------------------------------------------------------- #
# Speculative decoding: a local draft model proposes, the pipeline verifies
# --------------------------------------------------------------------------- #

@dataclass
class SpeculativeResult(GenerationResult):
    rounds: int = 0
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.proposed_draft_tokens if self.proposed_draft_tokens else 0.0

    @property
    def tokens_per_round(self) -> float:
        return len(self.token_ids) / self.rounds if self.rounds else 0.0


class _DraftModel:
    """Full small model on the client with its own KV cache."""

    def __init__(self, model: Any) -> None:
        from transformers import DynamicCache

        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.cache = DynamicCache(config=model.config)
        self.length = 0

    def propose(self, sequence: torch.Tensor, k: int) -> list[int]:
        """Greedy k tokens continuing ``sequence``; reuses the cache for the common prefix."""
        # The caller always appends at least one token (the target's bonus token) between
        # rounds, so there is something new to feed; keep one token if it ever isn't.
        keep = min(self.length, sequence.shape[1] - 1)
        if self.length > keep:
            self.cache.crop(keep)
            self.length = keep
        proposed: list[int] = []
        new = sequence[:, self.length:].to(self.device)
        with torch.inference_mode():
            for _ in range(k):
                logits = self.model(new, past_key_values=self.cache, use_cache=True).logits
                self.length += new.shape[1]
                token = int(logits[0, -1].argmax())
                proposed.append(token)
                new = torch.tensor([[token]], device=self.device)
        # The last proposed token was not fed: the cache covers sequence + proposed[:-1].
        return proposed

    def accept(self, valid_length: int) -> None:
        if self.length > valid_length:
            self.cache.crop(valid_length)
            self.length = valid_length


class SpeculativeGenerator:
    """Greedy speculative decoding over pipeline stages.

    Each round the local ``draft_model`` (same tokenizer, much smaller) proposes
    ``k`` tokens; the pipeline runs the pending token plus all proposals in one pass
    and returns the target model's greedy choice at every position. The longest
    agreeing prefix is accepted plus one token from the target, rejected positions
    are cropped from the stage caches with ``rewind``. The output is exactly what
    greedy decoding with the target model produces, but one pipeline round trip
    yields up to ``k + 1`` tokens.
    """

    def __init__(
        self,
        call: CallFn,
        roles: list[str],
        draft_model: Any,
        *,
        k: int = 4,
        route_call: Optional[RouteCallFn] = None,
        max_recoveries_per_step: int = 2,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.pipeline = PipelineGenerator(
            call, roles, route_call=route_call, max_recoveries_per_step=max_recoveries_per_step
        )
        self.draft = _DraftModel(draft_model)
        self.k = k
        self.max_recoveries_per_step = max_recoveries_per_step

    async def _verify(self, session_id: str, start_pos: int, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        payload: dict = {
            "session_id": session_id,
            "start_pos": start_pos,
            "rewind": True,
            "output": "argmax",
            "input_ids": tokens,
        }
        pipeline = self.pipeline
        if pipeline.route_call is not None:
            response = await pipeline.route_call(pipeline.roles, payload)
        else:
            response = {}
            for role in pipeline.roles:
                response = await pipeline.call(role, payload)
                if response.get("cache_miss") or "argmax" in response:
                    break
                payload = response
        if response.get("cache_miss"):
            return None
        if "argmax" not in response:
            raise RuntimeError("last stage did not return argmax — check the layer ranges of the roles")
        return response["argmax"]

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: Optional[int | list[int]] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> SpeculativeResult:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must have shape (1, seq_len): batching is not supported")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])

        session_id = uuid.uuid4().hex
        sequence = input_ids.clone()
        new_ids: list[int] = []
        token_seconds: list[float] = []
        prefill_seconds = 0.0
        recoveries = rounds = proposed_total = accepted_total = 0
        cached = 0  # valid tokens in the stage caches

        try:
            while len(new_ids) < max_new_tokens:
                started = time.perf_counter()
                k = min(self.k, max_new_tokens - len(new_ids))
                proposal = self.draft.propose(sequence, k)
                pending = sequence[:, cached:]
                candidate = torch.cat([pending, torch.tensor([proposal], dtype=sequence.dtype)], dim=1)

                target = await self._verify(session_id, cached, candidate)
                attempts = 0
                while target is None:
                    # A stage lost its cache: replay sequence + proposal from scratch.
                    attempts += 1
                    recoveries += 1
                    if attempts > self.max_recoveries_per_step:
                        raise CacheRecoveryError(f"stages keep losing the cache (round {rounds})")
                    full = torch.cat([sequence, torch.tensor([proposal], dtype=sequence.dtype)], dim=1)
                    replay = await self._verify(session_id, 0, full)
                    # The replay covers every position; candidate == full[cached:].
                    target = None if replay is None else replay[cached:]
                rounds += 1

                # target[i] is the target model's next token after candidate[: i + 1].
                offset = pending.shape[1] - 1
                predictions = target[offset:].tolist()  # k + 1 values
                accepted = 0
                while accepted < k and proposal[accepted] == predictions[accepted]:
                    accepted += 1
                round_tokens = proposal[:accepted] + [predictions[accepted]]
                proposed_total += k
                accepted_total += accepted

                stop = False
                emitted: list[int] = []
                for token in round_tokens:
                    if len(new_ids) >= max_new_tokens:
                        break
                    new_ids.append(token)
                    emitted.append(token)
                    if on_token is not None:
                        on_token(token)
                    if token in eos:
                        stop = True
                        break

                old_length = sequence.shape[1]
                sequence = torch.cat([sequence, torch.tensor([emitted], dtype=sequence.dtype)], dim=1)
                # Stages hold pending + proposal; valid are pending + accepted drafts.
                cached = old_length + accepted
                self.draft.accept(old_length + accepted)

                elapsed = time.perf_counter() - started
                if rounds == 1:
                    prefill_seconds = elapsed  # includes the prompt; not counted in token_seconds
                else:
                    token_seconds.extend([elapsed / len(emitted)] * len(emitted))
                if stop:
                    break
        finally:
            await self.pipeline._close(session_id)

        return SpeculativeResult(
            token_ids=new_ids,
            prefill_seconds=prefill_seconds,
            token_seconds=token_seconds,
            cache_recoveries=recoveries,
            rounds=rounds,
            proposed_draft_tokens=proposed_total,
            accepted_draft_tokens=accepted_total,
        )
