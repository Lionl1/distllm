"""Plan which layers each machine should hold, and print the commands to run.

Sizes come from the model architecture (built on the meta device, nothing is
downloaded but config.json). Machines are listed in pipeline order; the first one
also holds the embeddings, the last one the final norm and lm_head. A machine may
appear twice, e.g. to keep the first and the last stage on your own computer:

    uv run python examples/plan_split.py --model-id Qwen/Qwen2.5-14B-Instruct \\
        --nodes me:6:2,alice:10,bob:10,me:6:2 --quantize nf4 --context 4096

``name:GB:layers`` fixes a stage's layer count (here: thin first and last stages on your
machine, so friends only see hidden states); other stages get layers in proportion to
their machine's free memory. Memory per node is in GB of the device that runs the stage (VRAM for a GPU, RAM for
CPU). The estimate covers weights and the KV cache for one --context-long session;
leave headroom for activations and the framework (the planner keeps 15%).
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import torch

BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "int8": 1.0, "nf4": 0.53, "fp4": 0.53}
HEADROOM = 0.85
_LAYER_RE = re.compile(r"(^|\.)layers\.(\d+)\.")


def layer_sizes(model_id: str, token: str | None = None):
    """Parameter counts: per transformer layer, embeddings, and final norm + lm_head."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, token=token)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    per_layer: dict[int, int] = defaultdict(int)
    embed = head = 0
    for name, param in model.named_parameters():
        match = _LAYER_RE.search(name)
        if match:
            per_layer[int(match.group(2))] += param.numel()
        elif "embed_tokens" in name:
            embed += param.numel()
        else:
            head += param.numel()
    if getattr(config, "tie_word_embeddings", False) and "lm_head.weight" not in dict(model.named_parameters()):
        head += embed  # the last stage keeps its own copy of the tied matrix
    layers = [per_layer[i] for i in range(config.num_hidden_layers)]
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return config, layers, embed, head, kv_heads * head_dim


def plan(layers, embed, head, kv_width, nodes, quantize, dtype, context):
    """Split layers across stages in proportion to each machine's free memory.

    ``nodes``: [(machine, capacity_gb, fixed_layer_count_or_None), ...] in pipeline
    order. A machine listed several times shares its memory between those stages.
    """
    weight_bytes = BYTES_PER_PARAM[quantize or dtype]
    dense_bytes = BYTES_PER_PARAM[dtype]  # embeddings / lm_head are never quantized
    kv_per_layer = 2 * kv_width * context * BYTES_PER_PARAM[dtype]
    avg_layer = sum(layers) / len(layers) * weight_bytes + kv_per_layer
    n_stages, n_layers = len(nodes), len(layers)

    capacity = {}
    stages_on = defaultdict(list)
    for i, (machine, gb, _) in enumerate(nodes):
        capacity.setdefault(machine, gb * 1e9 * HEADROOM)
        stages_on[machine].append(i)
    fixed_cost = [0.0] * n_stages
    fixed_cost[0] += embed * dense_bytes
    fixed_cost[-1] += head * dense_bytes

    counts = [count for _, _, count in nodes]
    free_layers = n_layers - sum(c for c in counts if c)
    flexible = [i for i, c in enumerate(counts) if not c]
    if free_layers < len(flexible):
        raise SystemExit("fixed layer counts leave too few layers for the other stages")
    # Weight of a flexible stage: its machine's memory left after fixed costs and fixed
    # stages, shared equally among the machine's flexible stages.
    weights = {}
    for machine, idx in stages_on.items():
        left = capacity[machine] - sum(fixed_cost[i] for i in idx)
        left -= sum((counts[i] or 0) * avg_layer for i in idx)
        flex = [i for i in idx if not counts[i]]
        for i in flex:
            weights[i] = max(left, 0.0) / len(flex)
    total_weight = sum(weights[i] for i in flexible) or 1.0
    for i in flexible:
        counts[i] = max(1, int(free_layers * weights[i] / total_weight))
    # Hand out the rounding remainder to the stages with most spare memory per layer.
    while sum(counts) < n_layers:
        best = max(flexible, key=lambda i: weights[i] / (counts[i] + 1))
        counts[best] += 1
    while sum(counts) > n_layers:
        worst = max((i for i in flexible if counts[i] > 1), key=lambda i: counts[i] / max(weights[i], 1.0))
        counts[worst] -= 1

    stages, start = [], 0
    used_by = defaultdict(float)
    for i, (machine, gb, _) in enumerate(nodes):
        a, b = start, start + counts[i] - 1
        used = fixed_cost[i] + sum(layers[a : b + 1]) * weight_bytes + counts[i] * kv_per_layer
        used_by[machine] += used
        stages.append((machine, a, b, used / 1e9, gb))
        start = b + 1
    for machine, used in used_by.items():
        if used > capacity[machine]:
            raise SystemExit(
                f"{machine}: needs ~{used / 1e9:.1f} GB, has {capacity[machine] / HEADROOM / 1e9:.1f} GB "
                f"({int((1 - HEADROOM) * 100)}% kept as headroom) — add memory/nodes or quantize"
            )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--nodes", required=True,
                        help="name:GB[:layers],... in pipeline order; repeat a name to give one machine several stages")
    parser.add_argument("--quantize", choices=["nf4", "fp4", "int8"], default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                        help="dtype of the stages (CPU stages: float32 unless you pass --dtype)")
    parser.add_argument("--context", type=int, default=4096, help="tokens of KV cache to reserve per session")
    parser.add_argument("--relay-url", default="https://RELAY")
    parser.add_argument("--token", default=None, help="Hugging Face token for gated models")
    args = parser.parse_args()

    nodes = []
    for item in args.nodes.split(","):
        name, gb, *count = item.split(":")
        nodes.append((name.strip(), float(gb), int(count[0]) if count else None))

    config, layers, embed, head, kv_width = layer_sizes(args.model_id, args.token)
    stages = plan(layers, embed, head, kv_width, nodes, args.quantize, args.dtype, args.context)

    print(f"{args.model_id}: {len(layers)} layers, {sum(layers) / 1e9:.1f}B params in blocks, "
          f"weights {args.quantize or args.dtype}, KV for {args.context} tokens\n")
    print(f"{'machine':<10} {'role':<9} {'layers':<10} {'est. GB':>8} / {'machine GB':>10}")
    for i, (name, a, b, used, cap) in enumerate(stages):
        print(f"{name:<10} stage-{i:<3} {a:>3}..{b:<5} {used:>8.1f} / {cap:>10.1f}")

    print("\nCommands:")
    quant = f" --quantize {args.quantize}" if args.quantize else ""
    for i, (name, a, b, _, _) in enumerate(stages):
        print(f"  [{name}] uv run python examples/distributed_hf/worker_hf.py --model-id {args.model_id} "
              f"--layer-start {a} --layer-end {b} --role stage-{i} --relay-url {args.relay_url}"
              f" --dtype {args.dtype}{quant}")
    roles = ",".join(f"stage-{i}" for i in range(len(stages)))
    print(f"  [client] uv run python examples/distributed_hf/client_hf.py --model-id {args.model_id} "
          f"--workers {roles} --relay-url {args.relay_url} --route")


if __name__ == "__main__":
    main()
