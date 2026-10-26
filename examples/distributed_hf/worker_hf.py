"""Pipeline stage worker: holds layers [layer_start, layer_end] of a HF model.

Example (two stages of a 24-layer model):
    python worker_hf.py --model-id Qwen/Qwen2.5-0.5B --layer-start 0  --layer-end 11 --role stage-0
    python worker_hf.py --model-id Qwen/Qwen2.5-0.5B --layer-start 12 --layer-end 23 --role stage-1
"""

import argparse

from distllm import worker_node
from distllm.hf import load_stage


def create_worker(
    model_id, layer_start, layer_end, role, relay_url, token=None, device="auto", dtype="auto", relay_token=None,
    quantize=None, max_batch=8,
):
    stage = load_stage(
        model_id, layer_start, layer_end, device=device, dtype=dtype, token=token, quantize=quantize
    )
    print(
        f"[v] Stage {stage.name} ready on {stage.device} ({stage.dtype}). "
        f"First: {stage.is_first}, Last: {stage.is_last}"
    )

    # Decode steps of up to max_batch concurrent sessions share one forward pass.
    @worker_node(role=role, relay_url=relay_url, token=relay_token, max_batch=max_batch, batched=True)
    def process(payloads):
        return stage.handle_batch(payloads)

    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--relay-token", type=str, default=None,
                        help="relay worker token (default: $DISTLLM_WORKER_TOKEN or $DISTLLM_TOKEN)")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu | mps")
    parser.add_argument("--dtype", type=str, default="auto", help="auto | float16 | bfloat16 | float32")
    parser.add_argument("--quantize", type=str, default=None, choices=["nf4", "fp4", "int8"],
                        help="quantize the stage's transformer blocks with bitsandbytes")
    parser.add_argument("--max-batch", type=int, default=8,
                        help="decode steps of up to this many sessions run in one forward (1 = no batching)")
    args = parser.parse_args()
    create_worker(
        args.model_id, args.layer_start, args.layer_end, args.role, args.relay_url,
        token=args.token, device=args.device, dtype=args.dtype, relay_token=args.relay_token,
        quantize=args.quantize, max_batch=args.max_batch,
    ).start()
