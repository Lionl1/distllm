"""
client_hf.py - Distributed Inference Entry Node

This script coordinates multiple distllm workers to perform 
autoregressive text generation with large models.

Architecture:
1. Entry Node (This script): Tokenization, Embedding, LM Head, and Generation Loop.
2. Workers: Process slices of transformer layers.

Example for Gemma-2-9B (42 layers) split across 2 workers:
    Terminal 1 (Relay):
        uv run distllm relay
    Terminal 2 (Worker 1 - Layers 0-20):
        uv run python worker_hf.py --role worker-1 --layer-start 0 --layer-end 20 --model-id google/gemma-2-9b-it
    Terminal 3 (Worker 2 - Layers 21-41):
        uv run python worker_hf.py --role worker-2 --layer-start 21 --layer-end 41 --model-id google/gemma-2-9b-it
    Terminal 4 (This Client):
        uv run python client_hf.py --prompt "Hello" --model-id google/gemma-2-9b-it --workers "worker-1,worker-2"
"""

import torch
import asyncio
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from distllm import Cluster

def find_norm(model):
    """Robustly finds the final normalization layer."""
    # Common names for the final norm in many architectures
    norm_names = ["norm", "model.norm", "transformer.ln_f", "transformer.final_layernorm", "decoder.final_layernorm"]
    
    for name, module in model.named_modules():
        if name in norm_names:
            print(f"[*] Found normalization layer: {name}")
            return module
            
    # Fallback to model.model.norm if exists
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
        
    print("[!] Warning: Could not find normalization layer, using Identity.")
    return torch.nn.Identity()

async def generate(prompt, model_id, relay_url, worker_roles, max_new_tokens=50):
    print(f"[*] Initializing Client with model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model on CPU for embeddings and head
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        trust_remote_code=True
    )
    
    # Extract entry and exit components
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    norm = find_norm(model)

    cluster = Cluster(relay_url)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Generating tokens (using workers: {', '.join(worker_roles)})...")

    generated_ids = input_ids
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            # 1. Entry: Embedding
            hidden_states = embed_tokens(generated_ids)
            
            # 2. Sequential Workers
            for role in worker_roles:
                task = await cluster.submit(role, {"hidden_states": hidden_states})
                result = await cluster.wait_for(task)
                hidden_states = result["hidden_states"]
            
            # 3. Exit: Final Norm + Head
            hidden_states = norm(hidden_states)
            logits = lm_head(hidden_states[:, -1, :])
            
            # Greedy Decoding
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            # Stream output
            token_text = tokenizer.decode(next_token_id[0][0])
            print(token_text, end="", flush=True)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    print("\n\n[v] Generation complete.")
    return tokenizer.decode(generated_ids[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--workers", type=str, default="worker-1,worker-2", help="Comma-separated list of worker roles in order")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    worker_roles = [w.strip() for w in args.workers.split(",")]
    asyncio.run(generate(args.prompt, args.model_id, args.relay_url, worker_roles, args.max_tokens))
