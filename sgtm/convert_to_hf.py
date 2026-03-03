"""
Convert Facebook ESM library checkpoints to HuggingFace EsmForMaskedLM format.

The ESM-experiments project trains ESM-2 using the Facebook Research `esm` library,
saving state dicts with Facebook-style keys. The evalz evaluation framework uses
HuggingFace `transformers` EsmForMaskedLM. This script bridges the two.

The conversion is a pure key rename — no weight transpositions or shape changes.

Usage:
  python -m sgtm.convert_to_hf --checkpoint models/sgtm/holdout/final_model.pt --model-size 35M --output-dir models/hf/holdout
  python -m sgtm.convert_to_hf --checkpoint models/sgtm/sgtm_attn_mlp/final_model.pt --model-size 35M --output-dir models/hf/sgtm_attn_mlp

The output directory can be loaded with:
  from transformers import EsmForMaskedLM
  model = EsmForMaskedLM.from_pretrained("models/hf/holdout")
"""

import argparse
import re
import os

import torch


# HuggingFace reference model names for loading the correct EsmConfig
HF_REFERENCE = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}


def build_key_mapping(num_layers: int) -> dict:
    """Build the Facebook -> HuggingFace key rename mapping."""
    mapping = {
        # Embeddings
        "embed_tokens.weight": "esm.embeddings.word_embeddings.weight",
        # Post-encoder layer norm
        "emb_layer_norm_after.weight": "esm.encoder.emb_layer_norm_after.weight",
        "emb_layer_norm_after.bias": "esm.encoder.emb_layer_norm_after.bias",
        # Contact head
        "contact_head.regression.weight": "esm.contact_head.regression.weight",
        "contact_head.regression.bias": "esm.contact_head.regression.bias",
        # LM head (most keys are the same)
        "lm_head.dense.weight": "lm_head.dense.weight",
        "lm_head.dense.bias": "lm_head.dense.bias",
        "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        "lm_head.bias": "lm_head.bias",
        "lm_head.weight": "lm_head.decoder.weight",
    }

    for i in range(num_layers):
        fb = f"layers.{i}"
        hf = f"esm.encoder.layer.{i}"

        # Attention
        for proj, hf_name in [("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")]:
            mapping[f"{fb}.self_attn.{proj}.weight"] = f"{hf}.attention.self.{hf_name}.weight"
            mapping[f"{fb}.self_attn.{proj}.bias"] = f"{hf}.attention.self.{hf_name}.bias"

        mapping[f"{fb}.self_attn.out_proj.weight"] = f"{hf}.attention.output.dense.weight"
        mapping[f"{fb}.self_attn.out_proj.bias"] = f"{hf}.attention.output.dense.bias"

        # Rotary embeddings
        mapping[f"{fb}.self_attn.rot_emb.inv_freq"] = f"{hf}.attention.self.rotary_embeddings.inv_freq"

        # Layer norms
        mapping[f"{fb}.self_attn_layer_norm.weight"] = f"{hf}.attention.LayerNorm.weight"
        mapping[f"{fb}.self_attn_layer_norm.bias"] = f"{hf}.attention.LayerNorm.bias"
        mapping[f"{fb}.final_layer_norm.weight"] = f"{hf}.LayerNorm.weight"
        mapping[f"{fb}.final_layer_norm.bias"] = f"{hf}.LayerNorm.bias"

        # MLP
        mapping[f"{fb}.fc1.weight"] = f"{hf}.intermediate.dense.weight"
        mapping[f"{fb}.fc1.bias"] = f"{hf}.intermediate.dense.bias"
        mapping[f"{fb}.fc2.weight"] = f"{hf}.output.dense.weight"
        mapping[f"{fb}.fc2.bias"] = f"{hf}.output.dense.bias"

    return mapping


# Layer counts per model size
NUM_LAYERS = {"8M": 6, "35M": 12, "150M": 30, "650M": 33}


def convert_state_dict(fb_state_dict: dict, model_size: str) -> dict:
    """Rename Facebook ESM keys to HuggingFace format."""
    num_layers = NUM_LAYERS[model_size]
    mapping = build_key_mapping(num_layers)

    hf_state_dict = {}
    unmapped = []

    for fb_key, tensor in fb_state_dict.items():
        if fb_key in mapping:
            hf_state_dict[mapping[fb_key]] = tensor
        else:
            unmapped.append(fb_key)

    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped keys: {unmapped}")

    # Check coverage
    expected = len(mapping)
    actual = len(hf_state_dict)
    if actual != expected:
        print(f"WARNING: Expected {expected} keys, got {actual}")

    return hf_state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert Facebook ESM checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (state dict)")
    parser.add_argument("--model-size", required=True, choices=list(HF_REFERENCE.keys()),
                        help="Model size (determines architecture config)")
    parser.add_argument("--output-dir", required=True, help="Output directory for HF model")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Handle wrapped checkpoints (from periodic saves)
    if "model_state_dict" in state_dict:
        print("  Detected wrapped checkpoint, extracting model_state_dict")
        state_dict = state_dict["model_state_dict"]

    print(f"  {len(state_dict)} keys in Facebook format")

    # Convert keys
    hf_state_dict = convert_state_dict(state_dict, args.model_size)
    print(f"  {len(hf_state_dict)} keys in HuggingFace format")

    # Load HF config and create model
    from transformers import EsmConfig, EsmForMaskedLM

    ref_name = HF_REFERENCE[args.model_size]
    print(f"Loading config from {ref_name}...")
    config = EsmConfig.from_pretrained(ref_name)

    print("Creating EsmForMaskedLM and loading weights...")
    model = EsmForMaskedLM(config)

    # Load with strict=False to handle tied weights (lm_head.decoder.weight = embeddings)
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)

    # Filter out expected missing/unexpected from weight tying
    missing = [k for k in missing if "lm_head.decoder.weight" not in k]
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"\nSaved HuggingFace model to {args.output_dir}")

    # Also save the tokenizer so the directory is self-contained
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ref_name)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved tokenizer to {args.output_dir}")
    except Exception as e:
        print(f"Tokenizer save skipped ({e}). Load tokenizer separately from {ref_name}.")


if __name__ == "__main__":
    main()
