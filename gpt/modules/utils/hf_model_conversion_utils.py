import os
import sys
import argparse
import yaml
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

def check_transposed(hf_key):
    """
    Returns True if the Hugging Face weight parameter is represented
    as a Conv1D layer and requires transposition from standard PyTorch Linear shape.
    """
    conv1d_suffixes = [
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight"
    ]
    return any(hf_key.endswith(suffix) for suffix in conv1d_suffixes)

def analyze_and_convert(checkpoint_path, config_path=None, n_heads_override=None, output_dir="huggingface_model", verify=True):
    print("=" * 80)
    print(f"Starting Analysis and Conversion of Checkpoint: {checkpoint_path}")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist.")
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Found 'model_state_dict' in checkpoint.")
    else:
        state_dict = checkpoint
        print("Using root dictionary as state_dict.")

    print(f"Checkpoint contains {len(state_dict)} state keys.")

    # 1. Dynamically Infer Architecture
    print("\n--- Dynamically Inferring Architecture Parameters ---")
    
    # Extract vocab_size and d_model from embedding
    if "embedding.weight" in state_dict:
        embedding_shape = state_dict["embedding.weight"].shape
        vocab_size = embedding_shape[0]
        d_model = embedding_shape[1]
        print(f"Inferred vocab_size: {vocab_size} (from embedding.weight shape {list(embedding_shape)})")
        print(f"Inferred d_model:    {d_model} (from embedding.weight shape {list(embedding_shape)})")
    else:
        print("Error: Could not find 'embedding.weight' in checkpoint state_dict.")
        sys.exit(1)

    # Extract context_length from learnt_position_embedding
    if "learnt_position_embedding.weight" in state_dict:
        pos_embedding_shape = state_dict["learnt_position_embedding.weight"].shape
        context_length = pos_embedding_shape[0]
        print(f"Inferred context_length: {context_length} (from learnt_position_embedding.weight shape {list(pos_embedding_shape)})")
    else:
        print("Warning: 'learnt_position_embedding.weight' not found. Defaulting context_length to 1024.")
        context_length = 1024

    # Extract number of layers (n_layers)
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("transformer_layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))
    if layer_indices:
        n_layers = max(layer_indices) + 1
        print(f"Inferred n_layers:   {n_layers} (from maximum layer index found: {max(layer_indices)})")
    else:
        print("Warning: Could not infer n_layers from keys starting with 'transformer_layers.'. Defaulting to 12.")
        n_layers = 12

    # Extract number of heads (n_heads)
    n_heads = None
    if n_heads_override is not None:
        n_heads = n_heads_override
        print(f"Using command-line override n_heads: {n_heads}")
    elif config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                model_cfg = config_data.get("model", {})
                n_heads = model_cfg.get("n_heads")
                if n_heads is not None:
                    print(f"Inferred n_heads:    {n_heads} (from YAML config '{config_path}')")
        except Exception as e:
            print(f"Warning: Failed to load n_heads from config file: {e}")
            
    if n_heads is None:
        # Heuristic: standard head dim is 64
        n_heads = d_model // 64
        print(f"Inferred n_heads:    {n_heads} (using heuristic d_model // 64)")

    # 2. Build Mappings & Compare with Standard HF Model
    print("\n--- Constructing Key Mapping and Verifying Shape Compatibility ---")
    
    # Initialize target standard model
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_embd=d_model,
        n_layer=n_layers,
        n_head=n_heads,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    
    print("Instantiating standard GPT2LMHeadModel with inferred config...")
    hf_ref_model = GPT2LMHeadModel(hf_config)
    hf_ref_state = hf_ref_model.state_dict()

    # Build key mapping dictionary: hf_key -> custom_key
    hf_to_custom = {
        "transformer.wte.weight": "embedding.weight",
        "transformer.wpe.weight": "learnt_position_embedding.weight",
        "transformer.ln_f.weight": "final_layer_norm.weight",
        "transformer.ln_f.bias": "final_layer_norm.bias",
        "lm_head.weight": "embedding.weight",  # Weight tying
    }

    for i in range(n_layers):
        p_c = f"transformer_layers.{i}"
        p_h = f"transformer.h.{i}"
        
        # Layer norms
        hf_to_custom[f"{p_h}.ln_1.weight"] = f"{p_c}.layer_norm_mha.weight"
        hf_to_custom[f"{p_h}.ln_1.bias"] = f"{p_c}.layer_norm_mha.bias"
        hf_to_custom[f"{p_h}.ln_2.weight"] = f"{p_c}.layer_norm_ffn.weight"
        hf_to_custom[f"{p_h}.ln_2.bias"] = f"{p_c}.layer_norm_ffn.bias"
        
        # Attention
        hf_to_custom[f"{p_h}.attn.c_attn.weight"] = f"{p_c}.MHA.in_proj_weight"
        hf_to_custom[f"{p_h}.attn.c_attn.bias"] = f"{p_c}.MHA.in_proj_bias"
        hf_to_custom[f"{p_h}.attn.c_proj.weight"] = f"{p_c}.MHA.out_proj.weight"
        hf_to_custom[f"{p_h}.attn.c_proj.bias"] = f"{p_c}.MHA.out_proj.bias"
        
        # MLP / FFN
        hf_to_custom[f"{p_h}.mlp.c_fc.weight"] = f"{p_c}.FFN.linear_expansion.weight"
        hf_to_custom[f"{p_h}.mlp.c_fc.bias"] = f"{p_c}.FFN.linear_expansion.bias"
        hf_to_custom[f"{p_h}.mlp.c_proj.weight"] = f"{p_c}.FFN.linear_projection.weight"
        hf_to_custom[f"{p_h}.mlp.c_proj.bias"] = f"{p_c}.FFN.linear_projection.bias"

    # Print table header
    col_widths = [45, 45, 18, 18, 10]
    header = f"{'Hugging Face Key':<{col_widths[0]}} | {'Custom Key':<{col_widths[1]}} | {'HF Shape':<{col_widths[2]}} | {'Custom Shape':<{col_widths[3]}} | {'Status':<{col_widths[4]}}"
    print(header)
    print("-" * len(header))

    mismatches = 0
    missing = 0
    hf_mapped_state_dict = {}

    for hf_key in sorted(hf_ref_state.keys()):
        custom_key = hf_to_custom.get(hf_key)
        hf_shape = list(hf_ref_state[hf_key].shape)
        
        if custom_key is None:
            print(f"{hf_key:<{col_widths[0]}} | {'(No mapping)':<{col_widths[1]}} | {str(hf_shape):<{col_widths[2]}} | {'-':<{col_widths[3]}} | {'MISSING_MAP':<{col_widths[4]}}")
            missing += 1
            continue

        if custom_key not in state_dict:
            print(f"{hf_key:<{col_widths[0]}} | {custom_key:<{col_widths[1]}} | {str(hf_shape):<{col_widths[2]}} | {'Not Found':<{col_widths[3]}} | {'MISSING':<{col_widths[4]}}")
            missing += 1
            continue

        custom_weight = state_dict[custom_key]
        custom_shape = list(custom_weight.shape)
        
        # Determine expected shape (transposed if it's Conv1D-compatible weight)
        is_transposed = check_transposed(hf_key)
        expected_shape = [hf_shape[1], hf_shape[0]] if is_transposed else hf_shape

        if custom_shape != expected_shape:
            status = "MISMATCH"
            mismatches += 1
        else:
            status = "OK"
            # Extract and process weight
            weight_tensor = custom_weight.clone()
            if is_transposed:
                weight_tensor = weight_tensor.t().contiguous()
            hf_mapped_state_dict[hf_key] = weight_tensor

        print(f"{hf_key:<{col_widths[0]}} | {custom_key:<{col_widths[1]}} | {str(hf_shape):<{col_widths[2]}} | {str(custom_shape):<{col_widths[3]}} | {status:<{col_widths[4]}}")

    print("-" * len(header))
    print(f"Comparison Summary: {mismatches} mismatches, {missing} missing keys out of {len(hf_ref_state)} target keys.")

    if mismatches > 0 or missing > 0:
        print("\n[ERROR] Model architecture comparison failed. Cannot convert weight keys safely.")
        sys.exit(1)

    print("\n[SUCCESS] Model architecture and weight shapes match perfectly!")

    # 3. Load Mapped Weights into HF Model
    print("\n--- Loading Converted Weights ---")
    missing_keys, unexpected_keys = hf_ref_model.load_state_dict(hf_mapped_state_dict, strict=True)
    print("State dict loaded successfully!")
    print(f"Missing keys (should be empty): {missing_keys}")
    print(f"Unexpected keys (should be empty): {unexpected_keys}")

    # 4. Save converted Model and Tokenizer
    print(f"\n--- Saving Converted Model and Tokenizer to '{output_dir}' ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and save tokenizer
    print("Loading base tokenizer 'openai-community/gpt2'...")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    
    hf_ref_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Hugging Face model saved successfully to: {os.path.abspath(output_dir)}")

    # 5. Optional verification
    if verify:
        print("\n--- Running Generation Verification ---")
        try:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            print(f"Loading saved model onto device: {device}...")
            model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
            tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
            model.eval()

            prompt = "Deep learning is"
            print(f"Prompt: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated output text: '{generated_text}'")
            print("\nVerification SUCCESSFUL!")
        except Exception as e:
            print(f"\n[ERROR] Verification failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Analyze and convert custom GPT-2 checkpoints to standard Hugging Face format.")
    parser.add_argument(
        "--checkpoint", 
        default="assets/models/20260325_184232/checkpoint_12900.pt", 
        help="Path to the custom model checkpoint file (.pt)."
    )
    parser.add_argument(
        "--config", 
        default=None, 
        help="Optional path to model config YAML file (e.g. configs/gpt2.yaml)."
    )
    parser.add_argument(
        "--n_heads", 
        type=int, 
        default=None, 
        help="Optional override for the number of attention heads."
    )
    parser.add_argument(
        "--output_dir", 
        default="assets/models/20260325_184232/huggingface_model", 
        help="Local directory path to save the converted model and tokenizer."
    )
    parser.add_argument(
        "--no_verify", 
        action="store_false", 
        dest="verify", 
        help="Skip running the generation verification step."
    )

    args = parser.parse_args()
    
    analyze_and_convert(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        n_heads_override=args.n_heads,
        output_dir=args.output_dir,
        verify=args.verify
    )

if __name__ == "__main__":
    main()
