import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import argparse
from distllm import worker_node
import gc

def find_layers(model):
    """
    Exhaustively searches the model for the transformer layers ModuleList.
    Skips vision components for VLM models like Gemma4.
    """
    # 1. Prioritize language_model / text_model
    for sub in ["language_model", "text_model", "transformer", "model"]:
        obj = getattr(model, sub, None)
        if obj is not None and obj != model:
             res = find_layers(obj)
             if res is not None:
                 return res

    # 2. Check if the current object is the ModuleList we want
    if isinstance(model, torch.nn.ModuleList) and len(model) > 0:
        # Check if it looks like a transformer block (has attn/mlp)
        # AND check that it's not a small list (vision usually has fewer layers)
        if any(hasattr(model[0], a) for a in ["self_attn", "mlp"]):
            return model

    # 3. Recursive search through children, SKIPPING vision
    for name, child in model.named_children():
        if "vision" in name.lower() or "audio" in name.lower():
            continue
        res = find_layers(child)
        if res is not None:
            return res
                
    return None

def load_worker_model(model_id, layer_start, layer_end):
    print(f"[*] Loading model {model_id} with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    all_layers = find_layers(model)
    if all_layers is None:
        raise ValueError(f"Could not automatically find text layers for model architecture: {type(model)}")

    print(f"[*] Found total of {len(all_layers)} layers in the model.")
    
    # Slice the layers
    layers = all_layers[layer_start : layer_end + 1]
    print(f"[v] Successfully isolated {len(layers)} layers (Index {layer_start} to {layer_end}).")
    return layers

def create_worker(model_id, layer_start, layer_end, role, relay_url):
    layers = load_worker_model(model_id, layer_start, layer_end)

    @worker_node(role=role, relay_url=relay_url)
    def process_layers(payload):
        hidden_states = payload["hidden_states"]
        device = next(layers[0].parameters()).device
        hidden_states = hidden_states.to(device)
        
        with torch.no_grad():
            for layer in layers:
                outputs = layer(hidden_states)
                if isinstance(outputs, (list, tuple)):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs
        
        return {"hidden_states": hidden_states.cpu()}

    return process_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    worker = create_worker(args.model_id, args.layer_start, args.layer_end, args.role, args.relay_url)
    worker.start()
