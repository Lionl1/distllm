"""Thin client: holds only the tokenizer and drives generation through the stages."""

import argparse
import asyncio

from transformers import AutoTokenizer

from distllm import Cluster
from distllm.hf import PipelineGenerator


async def generate(prompt, model_id, relay_url, worker_roles, max_new_tokens=50, token=None):
    print("[*] Initializing Thin Client (Tokenizer only)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print(f"[*] Prompt: {prompt}")
    print(f"[*] Generating tokens via {len(worker_roles)} stages: {', '.join(worker_roles)}")

    async with Cluster(relay_url) as cluster:
        generator = PipelineGenerator(cluster.call, worker_roles)
        result = await generator.generate(
            input_ids,
            max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            on_token=lambda tid: print(tokenizer.decode([tid]), end="", flush=True),
        )

    print("\n\n[v] Generation complete.")
    print(
        f"[i] prefill {result.prefill_seconds * 1000:.0f} ms, "
        f"{result.tokens_per_second:.2f} tokens/s, cache recoveries: {result.cache_recoveries}"
    )
    return prompt + tokenizer.decode(result.token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--workers", type=str, default="stage-0,stage-1", help="stage roles in order")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    worker_roles = [w.strip() for w in args.workers.split(",")]
    asyncio.run(generate(args.prompt, args.model_id, args.relay_url, worker_roles, args.max_tokens, args.token))
