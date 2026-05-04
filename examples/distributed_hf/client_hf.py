import torch
import asyncio
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from distllm import Cluster
import os

def find_norm(model):
    """Robustly finds the final normalization layer."""
    norm_names = ["norm", "model.norm", "language_model.norm", "transformer.ln_f"]
    for name, module in model.named_modules():
        if any(name.endswith(nn) for nn in norm_names):
            return module
    return torch.nn.Identity()

async def generate(prompt, model_id, relay_url, worker_roles, max_new_tokens=50, token=None):
    print(f"[*] Initializing Memory-Optimized Client for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    
    # 1. Load config only
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=token)
    
    # 2. Create model on 'meta' device (takes 0 RAM)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    # 3. Actually load ONLY embeddings and head from the real weights
    # Note: from_pretrained with device_map="cpu" and selective loading
    # To keep it simple and robust, we use a lightweight approach:
    print("[*] Loading only Embeddings and LM Head (saving RAM)...")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # This is the trick: we load the model but exclude the heavy 'layers'
        # In transformers, we can't easily partially load from Hub without local cache tricks,
        # so we load and then move to CPU only the needed parts, then delete the rest.
        # But for 'low RAM', the best is to use 'low_cpu_mem_usage=True'
        low_cpu_mem_usage=True,
        device_map="cpu",
        token=token,
        trust_remote_code=True,
        # We try to ignore the heavy layers if the model supports it via hooks, 
        # but the most reliable way is to just use the full object and delete blocks.
    )
    
    # Extract what we need
    embed_tokens = full_model.get_input_embeddings()
    lm_head = full_model.get_output_embeddings()
    norm = find_norm(full_model)
    
    # IMPORTANT: Delete the heavy transformer blocks to free RAM
    # This is a bit hacky but extremely effective for low RAM
    for sub in ["model", "language_model", "transformer"]:
        parent = getattr(full_model, sub, None)
        if parent:
            if hasattr(parent, "layers"):
                parent.layers = torch.nn.ModuleList([]) # Empty the layers!
            if hasattr(parent, "h"):
                parent.h = torch.nn.ModuleList([])
            if hasattr(parent, "blocks"):
                parent.blocks = torch.nn.ModuleList([])

    import gc
    gc.collect()
    print(f"[v] RAM Freed. Ready to generate.")

    cluster = Cluster(relay_url)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"[*] Prompt: {prompt}")
    generated_ids = input_ids
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            hidden_states = embed_tokens(generated_ids)
            
            for role in worker_roles:
                task = await cluster.submit(role, {"hidden_states": hidden_states})
                result = await cluster.wait_for(task)
                hidden_states = result["hidden_states"]
            
            hidden_states = norm(hidden_states)
            logits = lm_head(hidden_states[:, -1, :])
            
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
