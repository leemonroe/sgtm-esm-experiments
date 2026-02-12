"""
Ablation experiments on the trained SGTM model.

Tests alternative ablation strategies to diagnose why full zeroing
destroyed general capability:
  1. Noise reinitialization (replace forget params with retain-distribution noise)
  2. Attention-only ablation (zero only attention forget params, keep MLP)
  3. MLP-only ablation (zero only MLP forget params, keep attention)
"""

import copy
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import esm

from sgtm.data_pipeline import MLMCollator
from sgtm.masking import build_sgtm_masks


def compute_perplexity(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, repr_layers=[], return_contacts=False)
            logits = output["logits"]
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()

    return math.exp(total_loss / max(total_tokens, 1))


def evaluate_all(model, test_loaders, device):
    return {name: compute_perplexity(model, loader, device)
            for name, loader in test_loaders.items()}


def ablate_noise_reinit(model, forget_mask, retain_mask):
    """Replace forget params with noise matching retain param statistics."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in forget_mask:
                continue
            retain_vals = param.data[retain_mask[name]]
            mean = retain_vals.mean().item()
            std = retain_vals.std().item()
            noise = torch.normal(mean, std, size=(forget_mask[name].sum().item(),),
                                 device=param.device, dtype=param.dtype)
            param.data[forget_mask[name]] = noise


def ablate_attention_only(model, forget_mask):
    """Zero only attention forget params (q/k/v/out_proj), leave MLP intact."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in forget_mask:
                continue
            if "self_attn" in name:
                param.data[forget_mask[name]] = 0.0


def ablate_mlp_only(model, forget_mask):
    """Zero only MLP forget params (fc1/fc2), leave attention intact."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in forget_mask:
                continue
            if "fc1" in name or "fc2" in name:
                param.data[forget_mask[name]] = 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models/sgtm")
    parser.add_argument("--data-dir", default="data/sgtm")
    parser.add_argument("--output-dir", default="results/sgtm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    collator = MLMCollator(alphabet, mask_ratio=0.15, max_length=1022)

    test_loaders = {}
    for split in ("forget", "adjacent", "retain"):
        ds = load_from_disk(os.path.join(args.data_dir, split, "test"))
        test_loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                         collate_fn=collator, shuffle=False)

    # Load trained SGTM model
    ckpt = os.path.join(args.models_dir, "sgtm", "final_model.pt")
    print(f"Loading SGTM model from {ckpt}")

    def load_fresh():
        model = esm.model.esm2.ESM2(
            num_layers=6, embed_dim=320, attention_heads=20, alphabet=alphabet)
        model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
        model = model.to(args.device)
        model.eval()
        return model

    results = {}

    # 0. Pre-ablation baseline
    print("\n[0] SGTM pre-ablation")
    model = load_fresh()
    results["sgtm"] = evaluate_all(model, test_loaders, args.device)
    print(f"    forget={results['sgtm']['forget']:.2f}  adjacent={results['sgtm']['adjacent']:.2f}  retain={results['sgtm']['retain']:.2f}")

    # 1. Full zero ablation (reproduce original result)
    print("\n[1] Full zero ablation")
    model = load_fresh()
    retain_mask, forget_mask = build_sgtm_masks(model)
    from sgtm.masking import ablate
    ablate(model, forget_mask)
    results["zero_all"] = evaluate_all(model, test_loaders, args.device)
    print(f"    forget={results['zero_all']['forget']:.2f}  adjacent={results['zero_all']['adjacent']:.2f}  retain={results['zero_all']['retain']:.2f}")

    # 2. Noise reinitialization
    print("\n[2] Noise reinitialization")
    model = load_fresh()
    retain_mask, forget_mask = build_sgtm_masks(model)
    ablate_noise_reinit(model, forget_mask, retain_mask)
    results["noise_reinit"] = evaluate_all(model, test_loaders, args.device)
    print(f"    forget={results['noise_reinit']['forget']:.2f}  adjacent={results['noise_reinit']['adjacent']:.2f}  retain={results['noise_reinit']['retain']:.2f}")

    # 3. Attention-only ablation
    print("\n[3] Attention-only zero ablation")
    model = load_fresh()
    _, forget_mask = build_sgtm_masks(model)
    ablate_attention_only(model, forget_mask)
    results["zero_attn_only"] = evaluate_all(model, test_loaders, args.device)
    print(f"    forget={results['zero_attn_only']['forget']:.2f}  adjacent={results['zero_attn_only']['adjacent']:.2f}  retain={results['zero_attn_only']['retain']:.2f}")

    # 4. MLP-only ablation
    print("\n[4] MLP-only zero ablation")
    model = load_fresh()
    _, forget_mask = build_sgtm_masks(model)
    ablate_mlp_only(model, forget_mask)
    results["zero_mlp_only"] = evaluate_all(model, test_loaders, args.device)
    print(f"    forget={results['zero_mlp_only']['forget']:.2f}  adjacent={results['zero_mlp_only']['adjacent']:.2f}  retain={results['zero_mlp_only']['retain']:.2f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'Forget PPL':>12} {'Adjacent PPL':>14} {'Retain PPL':>12}")
    print(f"{'-'*70}")
    for name, r in results.items():
        print(f"{name:<25} {r['forget']:>12.2f} {r['adjacent']:>14.2f} {r['retain']:>12.2f}")
    print(f"{'='*70}")

    out_path = os.path.join(args.output_dir, "ablation_experiments.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
