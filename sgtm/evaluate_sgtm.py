"""
SGTM evaluation: compute MLM perplexity on held-out test splits for each model,
produce comparison table and plots.

Usage:
  python sgtm/evaluate_sgtm.py --models-dir models/sgtm --data-dir data/sgtm --device cuda
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import esm

from sgtm.data_pipeline import MLMCollator
from sgtm.masking import ablate, build_sgtm_masks


def compute_perplexity(model, loader, device, max_batches=None):
    """Compute MLM perplexity: exp(average cross-entropy on masked tokens)."""
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
            n_tokens = (labels != -100).sum().item()

            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load an ESM-2 8M model from a state_dict checkpoint."""
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = esm.model.esm2.ESM2(
        num_layers=6, embed_dim=320, attention_heads=20, alphabet=alphabet,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, alphabet


def evaluate_model(model, test_loaders, device):
    """Evaluate a model on all test splits, returning perplexity dict."""
    results = {}
    for name, loader in test_loaders.items():
        ppl = compute_perplexity(model, loader, device)
        results[name] = ppl
    return results


def plot_perplexity_comparison(all_results, output_path):
    """Bar chart comparing perplexity across models and data splits."""
    model_names = list(all_results.keys())
    splits = ["forget", "adjacent", "retain"]
    x = np.arange(len(splits))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        ppls = [all_results[model_name].get(s, 0) for s in splits]
        bars = ax.bar(x + i * width, ppls, width, label=model_name)
        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{ppl:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Data Split")
    ax.set_ylabel("Perplexity")
    ax.set_title("MLM Perplexity by Model and Data Category")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved perplexity comparison: {output_path}")


def plot_ablation_comparison(pre_ablation, post_ablation, output_path):
    """Before/after ablation comparison for SGTM model."""
    splits = ["forget", "adjacent", "retain"]
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    pre_ppls = [pre_ablation.get(s, 0) for s in splits]
    post_ppls = [post_ablation.get(s, 0) for s in splits]

    bars1 = ax.bar(x - width / 2, pre_ppls, width, label="SGTM (pre-ablation)")
    bars2 = ax.bar(x + width / 2, post_ppls, width, label="SGTM (post-ablation)")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Data Split")
    ax.set_ylabel("Perplexity")
    ax.set_title("Effect of Ablating Forget Parameters")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ablation comparison: {output_path}")


def plot_training_curves(models_dir, output_path):
    """Plot training loss curves for all modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for entry in sorted(os.listdir(models_dir)):
        history_path = os.path.join(models_dir, entry, "training_history.json")
        if not os.path.exists(history_path):
            continue
        mode = entry
        with open(history_path) as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        if not train_loss:
            continue

        steps = [entry["step"] for entry in train_loss]
        losses = [entry["loss"] for entry in train_loss]
        ax.plot(steps, losses, label=mode, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("MLM Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SGTM evaluation")
    parser.add_argument("--models-dir", default="models/sgtm")
    parser.add_argument("--data-dir", default="data/sgtm")
    parser.add_argument("--output-dir", default="results/sgtm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run names to evaluate (default: auto-detect)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test sets
    print("Loading test datasets...")
    _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    collator = MLMCollator(alphabet, mask_ratio=0.15, max_length=args.max_length)

    test_loaders = {}
    for split in ("forget", "adjacent", "retain"):
        ds = load_from_disk(os.path.join(args.data_dir, split, "test"))
        test_loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, collate_fn=collator, shuffle=False,
        )
        print(f"  {split}: {len(ds)} sequences")

    # Determine which runs to evaluate
    if args.runs:
        run_names = [r.strip() for r in args.runs.split(",")]
    else:
        # Auto-detect: find all subdirs with final_model.pt
        run_names = []
        for entry in sorted(os.listdir(args.models_dir)):
            ckpt = os.path.join(args.models_dir, entry, "final_model.pt")
            if os.path.exists(ckpt):
                run_names.append(entry)
        print(f"Auto-detected runs: {run_names}")

    # Evaluate each model
    all_results = {}

    for run_name in run_names:
        # Seed for reproducible masking across models
        torch.manual_seed(42)

        ckpt_path = os.path.join(args.models_dir, run_name, "final_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {run_name}: {ckpt_path} not found")
            continue

        print(f"\n{'='*40}")
        print(f"Evaluating: {run_name}")
        print(f"{'='*40}")

        model, _ = load_model_from_checkpoint(ckpt_path, args.device)
        results = evaluate_model(model, test_loaders, args.device)
        all_results[run_name] = results

        print(f"  forget PPL:   {results['forget']:.2f}")
        print(f"  adjacent PPL: {results['adjacent']:.2f}")
        print(f"  retain PPL:   {results['retain']:.2f}")

        # For SGTM runs: also evaluate after ablation
        if run_name.startswith("sgtm"):
            # Try loading saved masks; fall back to recomputing
            masks_path = os.path.join(args.models_dir, run_name, "masks.pt")
            if os.path.exists(masks_path):
                saved = torch.load(masks_path, map_location=args.device, weights_only=True)
                retain_mask = saved["retain_mask"]
                forget_mask = saved["forget_mask"]
                print(f"  Loaded masks from {masks_path}")
            else:
                print(f"  WARNING: No masks.pt found! Using DEFAULT 3-head masks.")
                print(f"  This is WRONG for 1-head variants. Check {masks_path}")
                retain_mask, forget_mask = build_sgtm_masks(model)

            ablate(model, forget_mask)
            ablated_results = evaluate_model(model, test_loaders, args.device)
            all_results[f"{run_name}_ablated"] = ablated_results

            print(f"\n  [after ablation]")
            print(f"  forget PPL:   {ablated_results['forget']:.2f}")
            print(f"  adjacent PPL: {ablated_results['adjacent']:.2f}")
            print(f"  retain PPL:   {ablated_results['retain']:.2f}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<20} {'Forget PPL':>12} {'Adjacent PPL':>14} {'Retain PPL':>12}")
    print(f"{'-'*60}")
    for name, results in all_results.items():
        print(f"{name:<20} {results['forget']:>12.2f} {results['adjacent']:>14.2f} {results['retain']:>12.2f}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(args.output_dir, "perplexity_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plots
    if len(all_results) > 1:
        plot_perplexity_comparison(
            all_results,
            os.path.join(args.output_dir, "perplexity_comparison.png"),
        )

    for run_name in run_names:
        ablated_key = f"{run_name}_ablated"
        if run_name in all_results and ablated_key in all_results:
            plot_ablation_comparison(
                all_results[run_name],
                all_results[ablated_key],
                os.path.join(args.output_dir, f"ablation_{run_name}.png"),
            )

    plot_training_curves(args.models_dir, os.path.join(args.output_dir, "training_curves.png"))

    # Success criteria check for each SGTM variant
    baseline_retain = all_results.get("baseline", {}).get("retain")
    for run_name in run_names:
        ablated_key = f"{run_name}_ablated"
        if ablated_key in all_results and run_name in all_results:
            forget_increase = all_results[ablated_key]["forget"] / all_results[run_name]["forget"]
            print(f"\nSuccess metrics ({run_name}):")
            print(f"  Forget PPL increase after ablation: {forget_increase:.2f}x")
            if baseline_retain:
                retain_ratio = all_results[ablated_key]["retain"] / baseline_retain
                print(f"  Retain PPL ratio (ablated/baseline): {retain_ratio:.2f}x")
            print(f"  Goal: forget increase >> 1, retain ratio ≈ 1")


if __name__ == "__main__":
    main()
