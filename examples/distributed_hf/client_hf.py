"""Thin client: holds only the tokenizer and drives generation through the stages."""

import argparse
import asyncio

from transformers import AutoTokenizer

from distllm import Cluster
from distllm.hf import PipelineGenerator, SamplingParams, SpeculativeGenerator


async def generate(
    prompt, model_id, relay_url, worker_roles, max_new_tokens=50, token=None,
    route=False, draft_model_id=None, speculative_k=4, relay_token=None, sampling=None,
):
    print("[*] Initializing Thin Client (Tokenizer only)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print(f"[*] Prompt: {prompt}")
    print(f"[*] Generating tokens via {len(worker_roles)} stages: {', '.join(worker_roles)}")

    async with Cluster(relay_url, token=relay_token) as cluster:
        route_call = cluster.call_route if route else None
        sampling = sampling or SamplingParams()
        if draft_model_id and not sampling.greedy:
            raise SystemExit("speculative decoding is greedy-only: drop --temperature or --draft-model-id")
        if draft_model_id:
            from transformers import AutoModelForCausalLM

            print(f"[*] Speculative decoding: draft {draft_model_id}, k={speculative_k}")
            draft = AutoModelForCausalLM.from_pretrained(draft_model_id, token=token)
            generator = SpeculativeGenerator(
                cluster.call, worker_roles, draft, k=speculative_k, route_call=route_call
            )
        else:
            generator = PipelineGenerator(cluster.call, worker_roles, route_call=route_call)
        extra = {} if draft_model_id else {"sampling": sampling}
        result = await generator.generate(
            input_ids,
            max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            on_token=lambda tid: print(tokenizer.decode([tid]), end="", flush=True),
            **extra,
        )

    print("\n\n[v] Generation complete.")
    print(
        f"[i] prefill {result.prefill_seconds * 1000:.0f} ms, "
        f"{result.tokens_per_second:.2f} tokens/s, cache recoveries: {result.cache_recoveries}"
    )
    if hasattr(result, "acceptance_rate"):
        print(f"[i] draft acceptance {result.acceptance_rate:.0%}, {result.tokens_per_round:.2f} tokens per round")
    return prompt + tokenizer.decode(result.token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--workers", type=str, default="stage-0,stage-1", help="stage roles in order")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--relay-token", type=str, default=None, help="relay client token (default: $DISTLLM_TOKEN)")
    parser.add_argument("--route", action="store_true", help="chain stages relay-side (one request per step)")
    parser.add_argument("--draft-model-id", type=str, default=None,
                        help="small model with the same tokenizer for speculative decoding")
    parser.add_argument("--speculative-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    worker_roles = [w.strip() for w in args.workers.split(",")]
    asyncio.run(generate(
        args.prompt, args.model_id, args.relay_url, worker_roles, args.max_tokens, args.token,
        route=args.route, draft_model_id=args.draft_model_id, speculative_k=args.speculative_k,
        relay_token=args.relay_token,
        sampling=SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed),
    ))
