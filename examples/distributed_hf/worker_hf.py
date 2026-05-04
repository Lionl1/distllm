import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import argparse
from distllm import worker_node
import gc

def find_layers(model):
    """Exhaustively searches for the transformer layers ModuleList."""
    for sub in ["language_model", "text_model", "transformer", "model"]:
        obj = getattr(model, sub, None)
        if obj is not None and obj != model:
             res = find_layers(obj)
             if res is not None:
                 return res
    if isinstance(model, torch.nn.ModuleList) and len(model) > 0:
        if any(hasattr(model[0], a) for a in ["self_attn", "mlp"]):
            return model
    for name, child in model.named_children():
        if "vision" in name.lower() or "audio" in name.lower(): continue
        res = find_layers(child)
        if res is not None: return res
    return None

def find_norm(model):
    norm_names = ["norm", "model.norm", "language_model.norm", "transformer.ln_f"]
    for name, module in model.named_modules():
        if any(name.endswith(nn) for nn in norm_names): return module
    return torch.nn.Identity()

def load_worker_model(model_id, layer_start, layer_end, token=None):
    print(f"[*] Loading model {model_id} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True, token=token
    )
    all_layers = find_layers(model)
    if all_layers is None: raise ValueError("Could not find layers")
    
    layers = all_layers[layer_start : layer_end + 1]
    is_first = (layer_start == 0)
    is_last = (layer_end == len(all_layers) - 1)
    
    components = {
        "layers": layers,
        "is_first": is_first,
        "is_last": is_last,
        "embed": model.get_input_embeddings() if is_first else None,
        "norm": find_norm(model) if is_last else None,
        "head": model.get_output_embeddings() if is_last else None
    }
    print(f"[v] Node configured. First: {is_first}, Last: {is_last}, Layers: {len(layers)}")
    return components

def create_worker(model_id, layer_start, layer_end, role, relay_url, token=None):
    comp = load_worker_model(model_id, layer_start, layer_end, token=token)

    @worker_node(role=role, relay_url=relay_url)
    def process(payload):
        with torch.no_grad():
            # 1. Handle Input
            if comp["is_first"] and "input_ids" in payload:
                # Client sent raw IDs
                h = comp["embed"](payload["input_ids"])
            else:
                h = payload["hidden_states"]
            
            # Ensure on correct device
            device = next(comp["layers"][0].parameters()).device
            h = h.to(device)
            
            # 2. Process Layers
            for layer in comp["layers"]:
                out = layer(h)
                h = out[0] if isinstance(out, (list, tuple)) else out
            
            # 3. Handle Output
            if comp["is_last"]:
                # Apply final norm and head to get logits
                h = comp["norm"](h)
                logits = comp["head"](h[:, -1, :]) # Only last token
                return {"logits": logits.cpu()}
            else:
                return {"hidden_states": h.cpu()}

    return process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()
    create_worker(args.model_id, args.layer_start, args.layer_end, args.role, args.relay_url, args.token).start()
