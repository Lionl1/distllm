import torch
from transformers import AutoModelForCausalLM, AutoConfig
import argparse
from distllm import worker_node
import gc

# To use this, you need: pip install bitsandbytes accelerate
# This script loads a slice of a large model in 4-bit to save memory.

def load_worker_model(model_id, layer_start, layer_end):
    print(f"[*] Loading layers {layer_start} to {layer_end} of {model_id} in 4-bit...")
    
    # We load the model with 4-bit quantization. 
    # To truly save RAM, we use device_map="auto" which will handle the bitsandbytes magic.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Extract the layer slice. 
    # For Llama/Gemma, layers are in model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers[layer_start:layer_end]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style
        layers = model.transformer.h[layer_start:layer_end]
    else:
        raise ValueError("Unsupported model architecture for automatic slicing.")

    print(f"[v] Successfully isolated {len(layers)} layers.")
    
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
                # Standard HF forward pass: (hidden_states, attention_mask, position_ids, ...)
                # We pass only hidden_states for simplicity in this MVP.
                # Note: For production, you'd also pass the attention_mask and position_ids.
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
