"""
client_hf.py - Distributed Inference Entry Node

This script demonstrates how to coordinate multiple distllm workers to perform 
autoregressive text generation with a large model like Llama-3.

Architecture:
1. Entry Node (This script): Tokenization, Embedding, LM Head, and Generation Loop.
2. Worker 1: Processes layers 0-15.
3. Worker 2: Processes layers 16-31.

How to run:

Terminal 1 (Relay):
    uv run distllm relay

Terminal 2 (Worker 1 - Layers 0-15):
    uv run python worker_hf.py --role worker-1 --layer-start 0 --layer-end 16 --model-id meta-llama/Meta-Llama-3-8B

Terminal 3 (Worker 2 - Layers 16-31):
    uv run python worker_hf.py --role worker-2 --layer-start 16 --layer-end 32 --model-id meta-llama/Meta-Llama-3-8B

Terminal 4 (This Client):
    uv run python client_hf.py --prompt "The future of distributed AI is"
"""

import torch
import asyncio
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from distllm import Cluster

async def generate(prompt, model_id, relay_url, max_new_tokens=20):
    print(f"[*] Initializing Client with model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # To keep the client lightweight, we only load the embeddings and the head.
    # We can do this by loading the model with an empty device map or just extracting them.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        trust_remote_code=True
    )
    
    # Extract entry and exit components
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    norm = model.model.norm if hasattr(model.model, "norm") else torch.nn.Identity()

    cluster = Cluster(relay_url)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Generating tokens...")

    generated_ids = input_ids
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            # 1. Entry: Embedding
            hidden_states = embed_tokens(generated_ids)
            
            # 2. Worker 1: Layers 0-15
            task_1 = await cluster.submit("worker-1", {"hidden_states": hidden_states})
            res_1 = await cluster.wait_for(task_1)
            hidden_states = res_1["hidden_states"]
            
            # 3. Worker 2: Layers 16-31
            task_2 = await cluster.submit("worker-2", {"hidden_states": hidden_states})
            res_2 = await cluster.wait_for(task_2)
            hidden_states = res_2["hidden_states"]
            
            # 4. Exit: Final Norm + Head
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
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(generate(args.prompt, args.model_id, args.relay_url))
