import torch
import asyncio
import argparse
from transformers import AutoTokenizer
from distllm import Cluster

async def generate(prompt, model_id, relay_url, worker_roles, max_new_tokens=50, token=None):
    print(f"[*] Initializing Thin Client (Tokenizer only)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    
    cluster = Cluster(relay_url)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Generating tokens via {len(worker_roles)} workers...")

    generated_ids = input_ids
    
    for i in range(max_new_tokens):
        # 1. First worker handles input_ids -> hidden_states
        task_1 = await cluster.submit(worker_roles[0], {"input_ids": generated_ids})
        res = await cluster.wait_for(task_1)
        
        # 2. Intermediate workers handle hidden_states -> hidden_states
        for role in worker_roles[1:-1]:
            task = await cluster.submit(role, res)
            res = await cluster.wait_for(task)
            
        # 3. Last worker handles hidden_states -> logits
        if len(worker_roles) > 1:
            task_last = await cluster.submit(worker_roles[-1], res)
            res = await cluster.wait_for(task_last)
        
        logits = res["logits"]
        
        # Greedy Decoding
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        token_text = tokenizer.decode(next_token_id[0][0])
        print(token_text, end="", flush=True)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    print("\n\n[v] Generation complete.")
    return tokenizer.decode(generated_ids[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--workers", type=str, default="worker-1,worker-2")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    worker_roles = [w.strip() for w in args.workers.split(",")]
    asyncio.run(generate(args.prompt, args.model_id, args.relay_url, worker_roles, args.max_tokens, args.token))
