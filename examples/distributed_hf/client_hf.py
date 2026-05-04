import torch
import asyncio
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from distllm import Cluster

def find_norm(model):
    """Robustly finds the final normalization layer."""
    # Priority names
    norm_names = ["norm", "model.norm", "language_model.norm", "transformer.ln_f"]
    
    for name, module in model.named_modules():
        if any(name.endswith(nn) for nn in norm_names):
            print(f"[*] Found normalization layer candidate: {name}")
            return module
            
    # Generic search for any RMSNorm or LayerNorm that looks like a final one
    best_norm = None
    for name, module in model.named_modules():
        if "Norm" in module.__class__.__name__:
            best_norm = module # Keep the last one found
            
    if best_norm:
        return best_norm
        
    print("[!] Warning: Could not find normalization layer, using Identity.")
    return torch.nn.Identity()

async def generate(prompt, model_id, relay_url, worker_roles, max_new_tokens=50, token=None):
    print(f"[*] Initializing Client with model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", 
        trust_remote_code=True,
        token=token
    )
    
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
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    args = parser.parse_args()

    worker_roles = [w.strip() for w in args.workers.split(",")]
    asyncio.run(generate(args.prompt, args.model_id, args.relay_url, worker_roles, args.max_tokens, args.token))
