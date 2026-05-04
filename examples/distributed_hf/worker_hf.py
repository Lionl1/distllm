import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import argparse
from distllm import worker_node
import gc

# To use this, you need: pip install bitsandbytes accelerate
# This script loads a slice of a large model in 4-bit to save memory.

def find_layers(model):
    """
    Robustly finds the transformer block list in various model architectures.
    Supports Llama, Gemma, GPT-2, Qwen, and many others.
    """
    # 1. Check common attributes directly
    for attr in ["h", "layers", "blocks", "layer", "block"]:
        sub_obj = getattr(model, attr, None)
        if isinstance(sub_obj, (torch.nn.ModuleList, list)):
            return sub_obj
    
    # 2. Check model.model, model.transformer, model.decoder recursively
    # This covers cases like Llama/Gemma (model.model.layers) and others.
    for sub in ["model", "transformer", "decoder"]:
        if hasattr(model, sub):
            res = find_layers(getattr(model, sub))
            if res is not None:
                return res
                
    return None

def load_worker_model(model_id, layer_start, layer_end):
    print(f"[*] Loading model {model_id} with 4-bit quantization...")
    
    # Modern BitsAndBytes configuration for transformers >= 5.0.0
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # To truly save RAM, we use device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    all_layers = find_layers(model)
    if all_layers is None:
        # Diagnostic print to help users identify the model structure
        print(f"[!] Could not find layers. Model structure: {model}")
        raise ValueError(f"Could not automatically find layers for model architecture: {type(model)}")

    # Slice the layers (inclusive of layer_end as per requirements)
    layers = all_layers[layer_start : layer_end + 1]

    print(f"[v] Successfully isolated {len(layers)} layers (Index {layer_start} to {layer_end}).")
    
    # Note: We keep the full model object in memory because the layers 
    # often depend on the parent config or shared constants, but only 
    # the target layers are active in our worker.
    return layers

def create_worker(model_id, layer_start, layer_end, role, relay_url):
    layers = load_worker_model(model_id, layer_start, layer_end)

    @worker_node(role=role, relay_url=relay_url)
    def process_layers(payload):
        hidden_states = payload["hidden_states"]
        
        # Ensure hidden_states is on the correct device
        # We assume the layers have been dispatched to a GPU by accelerate
        device = next(layers[0].parameters()).device
        hidden_states = hidden_states.to(device)
        
        with torch.no_grad():
            for layer in layers:
                # Standard HF forward pass
                outputs = layer(hidden_states)
                hidden_states = outputs[0]
        
        return {"hidden_states": hidden_states.cpu()}

    return process_layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--relay-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    worker = create_worker(args.model_id, args.layer_start, args.layer_end, args.role, args.relay_url)
    worker.start()
