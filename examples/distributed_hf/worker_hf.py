import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import argparse
from distllm import worker_node
import gc

def find_layers(model):
    """
    Exhaustively searches the model for the transformer layers ModuleList.
    """
    # 1. Check if the current object is the ModuleList we want
    if isinstance(model, torch.nn.ModuleList) and len(model) > 0:
        # Layers usually have 'self_attn' or 'mlp'
        if any(hasattr(model[0], a) for a in ["self_attn", "mlp", "attention", "block"]):
            return model

    # 2. Check common attributes
    for attr in ["h", "layers", "blocks", "layer", "block"]:
        obj = getattr(model, attr, None)
        if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
            return obj

    # 3. Recursive search through all children
    for name, child in model.named_children():
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
        print(f"[!] DEBUG: Model children: {[n for n, _ in model.named_children()]}")
        raise ValueError(f"Could not automatically find layers for model architecture: {type(model)}")

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
