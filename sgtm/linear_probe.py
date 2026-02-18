"""
Linear probe evaluation for SGTM models.

Extracts mean-pooled last-layer embeddings and trains logistic regression
classifiers to measure how well each model condition retains/forgets
viral classification capability.

Tasks:
  1. Human-infecting vs non-human viral protein classification
  2. Viral vs non-viral protein classification

Conditions: baseline, holdout, sgtm (pre-ablation), sgtm (post-ablation)

Usage:
  python -m sgtm.linear_probe --model-size 8M --models-dir models/sgtm --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtm.masking import ablate, build_sgtm_masks
from sgtm.model_config import get_config, load_alphabet, load_model_from_checkpoint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_virus_sequences(raw_dir="data/raw"):
    """Load human and non-human viral sequences from TSV files."""
    sequences = []
    labels = []

    for fname, label in [("virus_human.tsv", 1), ("virus_nonhuman.tsv", 0)]:
        path = os.path.join(raw_dir, fname)
        with open(path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    sequences.append(parts[3])
                    labels.append(label)

    return sequences, labels


def load_viral_vs_nonviral(raw_dir="data/raw", n_nonviral=500, seed=42):
    """Load balanced viral vs non-viral sequences for Task 2.

    Uses the virus TSV files for viral, and a sample of Swiss-Prot
    retain sequences (from data/sgtm/retain/test) for non-viral.
    """
    from datasets import load_from_disk

    # Viral sequences
    viral_seqs = []
    for fname in ("virus_human.tsv", "virus_nonhuman.tsv"):
        path = os.path.join(raw_dir, fname)
        with open(path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    viral_seqs.append(parts[3])

    # Non-viral from retain test set
    retain_test = load_from_disk("data/sgtm/retain/test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(retain_test), size=min(n_nonviral, len(retain_test)), replace=False)
    nonviral_seqs = [retain_test[int(i)]["sequence"] for i in indices]

    sequences = viral_seqs + nonviral_seqs
    labels = [1] * len(viral_seqs) + [0] * len(nonviral_seqs)
    return sequences, labels


def extract_embeddings(model, alphabet, sequences, device, batch_size=8):
    """Extract mean-pooled last-layer embeddings."""
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"s{j}", seq[:1022]) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers],
                            return_contacts=False)
            reps = results["representations"][model.num_layers]

            for j, seq in enumerate(batch_seqs):
                seq_len = min(len(seq), 1022)
                emb = reps[j, 1:seq_len + 1, :].mean(dim=0)
                embeddings.append(emb.cpu().numpy())

    return np.array(embeddings)


def logreg_cv(embeddings, labels, cv=5):
    """Logistic regression with cross-validation."""
    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, embeddings, labels, cv=cv)
    return float(scores.mean()), float(scores.std())


def main():
    parser = argparse.ArgumentParser(description="Linear probe evaluation for SGTM")
    parser.add_argument("--model-size", type=str, default="8M")
    parser.add_argument("--models-dir", default="models/sgtm")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results/sgtm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.model_size)
    if args.wandb_project is None:
        args.wandb_project = f"sgtm-esm2-{cfg.name.lower()}"

    os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"linear-probe-{cfg.name}",
            job_type="linear_probe",
            config=vars(args),
        )

    alphabet = load_alphabet()

    # Load evaluation data
    print("Loading Task 1: human-infecting vs non-human viral...")
    task1_seqs, task1_labels = load_virus_sequences(args.raw_dir)
    print(f"  {sum(task1_labels)} human-infecting, {len(task1_labels) - sum(task1_labels)} non-human")

    print("Loading Task 2: viral vs non-viral...")
    task2_seqs, task2_labels = load_viral_vs_nonviral(args.raw_dir)
    print(f"  {sum(task2_labels)} viral, {len(task2_labels) - sum(task2_labels)} non-viral")

    # Discover model conditions
    conditions = []
    for entry in sorted(os.listdir(args.models_dir)):
        ckpt = os.path.join(args.models_dir, entry, "final_model.pt")
        if os.path.exists(ckpt):
            conditions.append(entry)
    print(f"\nConditions found: {conditions}")

    all_results = {}

    for cond_name in conditions:
        ckpt_path = os.path.join(args.models_dir, cond_name, "final_model.pt")
        print(f"\n{'=' * 50}")
        print(f"Condition: {cond_name}")
        print(f"{'=' * 50}")

        model, _ = load_model_from_checkpoint(cfg, ckpt_path, args.device)

        # Pre-ablation embeddings
        emb1 = extract_embeddings(model, alphabet, task1_seqs, args.device, args.batch_size)
        emb2 = extract_embeddings(model, alphabet, task2_seqs, args.device, args.batch_size)

        t1_acc, t1_std = logreg_cv(emb1, task1_labels)
        t2_acc, t2_std = logreg_cv(emb2, task2_labels)

        print(f"  Task 1 (human vs non-human): {t1_acc:.3f} +/- {t1_std:.3f}")
        print(f"  Task 2 (viral vs non-viral): {t2_acc:.3f} +/- {t2_std:.3f}")

        all_results[cond_name] = {
            "task1_human_vs_nonhuman": {"mean": t1_acc, "std": t1_std},
            "task2_viral_vs_nonviral": {"mean": t2_acc, "std": t2_std},
        }

        # For SGTM runs: also evaluate post-ablation
        if cond_name.startswith("sgtm"):
            masks_path = os.path.join(args.models_dir, cond_name, "masks.pt")
            if os.path.exists(masks_path):
                saved = torch.load(masks_path, map_location=args.device, weights_only=True)
                forget_mask = saved["forget_mask"]
            else:
                _, forget_mask = build_sgtm_masks(
                    model, head_dim=cfg.head_dim,
                    forget_head_indices=list(cfg.default_forget_heads),
                    forget_mlp_start=cfg.default_forget_mlp_start,
                )

            ablate(model, forget_mask)

            emb1_abl = extract_embeddings(model, alphabet, task1_seqs, args.device, args.batch_size)
            emb2_abl = extract_embeddings(model, alphabet, task2_seqs, args.device, args.batch_size)

            t1_acc_abl, t1_std_abl = logreg_cv(emb1_abl, task1_labels)
            t2_acc_abl, t2_std_abl = logreg_cv(emb2_abl, task2_labels)

            abl_name = f"{cond_name}_ablated"
            print(f"\n  [post-ablation]")
            print(f"  Task 1: {t1_acc_abl:.3f} +/- {t1_std_abl:.3f}")
            print(f"  Task 2: {t2_acc_abl:.3f} +/- {t2_std_abl:.3f}")

            all_results[abl_name] = {
                "task1_human_vs_nonhuman": {"mean": t1_acc_abl, "std": t1_std_abl},
                "task2_viral_vs_nonviral": {"mean": t2_acc_abl, "std": t2_std_abl},
            }

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'Condition':<30} {'Task1 (h vs nh)':>16} {'Task2 (v vs nv)':>16}")
    print(f"{'-' * 70}")
    for name, r in all_results.items():
        t1 = r["task1_human_vs_nonhuman"]
        t2 = r["task2_viral_vs_nonviral"]
        print(f"{name:<30} {t1['mean']:.3f}±{t1['std']:.3f}    {t2['mean']:.3f}±{t2['std']:.3f}")
    print(f"{'=' * 70}")

    # Save
    out_path = os.path.join(args.output_dir, "linear_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    if use_wandb:
        for name, r in all_results.items():
            for task_name, scores in r.items():
                wandb.log({f"probe/{name}/{task_name}": scores["mean"]})
        wandb.finish()


if __name__ == "__main__":
    main()
