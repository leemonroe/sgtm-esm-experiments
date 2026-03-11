"""
Linear probe evaluation for SGTM models.

Extracts mean-pooled last-layer embeddings and trains logistic regression
classifiers to measure how well each model condition retains/forgets
knowledge matching the forget boundary.

Probes are constructed automatically from the data splits:
  - Forget vs All (adjacent + retain): matches the forget boundary exactly
  - Forget vs Adjacent: hardest test (within-domain discrimination)
  - Forget vs Retain: easiest test (cross-domain discrimination)

Works for any forget task (coarse, fine, family) without hardcoded categories.

Usage:
  python -m sgtm.linear_probe --model-size 8M --models-dir models/sgtm --data-dir data/sgtm/family_coronaviridae --device cuda
  python -m sgtm.linear_probe --model-size 35M --models-dir models/sgtm --data-dir data/sgtm/coarse --runs "holdout-35m,sgtm-ret25-35m" --device cuda
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


# ── Data loading ──────────────────────────────────────────────────────

def load_split_sequences(data_dir, split_name, partition="val"):
    """Load sequences from a data split. Returns list of sequences or None."""
    from datasets import load_from_disk
    split_path = os.path.join(data_dir, split_name, partition)
    if not os.path.isdir(split_path):
        return None
    ds = load_from_disk(split_path)
    return [row["sequence"] for row in ds]


def build_probe_tasks(data_dir, seed=42):
    """Build probe tasks adaptively from whatever splits exist.

    Returns a dict of {task_name: (sequences, labels, description)}.
    """
    forget_seqs = load_split_sequences(data_dir, "forget")
    adjacent_seqs = load_split_sequences(data_dir, "adjacent")
    retain_seqs = load_split_sequences(data_dir, "retain")

    if forget_seqs is None:
        print("WARNING: No forget/val split found — cannot build probes.")
        return {}

    rng = np.random.RandomState(seed)
    tasks = {}

    # Task 1: Forget vs All (the primary probe — matches forget boundary)
    neg_seqs = []
    if adjacent_seqs:
        neg_seqs.extend(adjacent_seqs)
    if retain_seqs:
        # Subsample retain to balance with forget + adjacent
        n_retain = max(len(forget_seqs) - len(neg_seqs), len(forget_seqs))
        n_retain = min(n_retain, len(retain_seqs))
        idx = rng.choice(len(retain_seqs), size=n_retain, replace=False)
        neg_seqs.extend([retain_seqs[i] for i in idx])

    if neg_seqs:
        # Balance: subsample negative to match positive (or vice versa)
        if len(neg_seqs) > len(forget_seqs) * 3:
            idx = rng.choice(len(neg_seqs), size=len(forget_seqs) * 2, replace=False)
            neg_seqs = [neg_seqs[i] for i in idx]

        seqs = forget_seqs + neg_seqs
        labels = [1] * len(forget_seqs) + [0] * len(neg_seqs)
        n_pos, n_neg = len(forget_seqs), len(neg_seqs)
        tasks["forget_vs_all"] = (seqs, labels,
            f"Forget vs All ({n_pos} forget, {n_neg} other)")

    # Task 2: Forget vs Adjacent (fine-grained, only if adjacent exists)
    if adjacent_seqs:
        # Balance classes
        if len(adjacent_seqs) > len(forget_seqs) * 3:
            idx = rng.choice(len(adjacent_seqs), size=len(forget_seqs) * 2, replace=False)
            adj_sample = [adjacent_seqs[i] for i in idx]
        else:
            adj_sample = adjacent_seqs

        seqs = forget_seqs + adj_sample
        labels = [1] * len(forget_seqs) + [0] * len(adj_sample)
        tasks["forget_vs_adjacent"] = (seqs, labels,
            f"Forget vs Adjacent ({len(forget_seqs)} forget, {len(adj_sample)} adjacent)")

    # Task 3: Forget vs Retain (cross-domain)
    if retain_seqs:
        n_retain = min(len(forget_seqs) * 2, len(retain_seqs))
        idx = rng.choice(len(retain_seqs), size=n_retain, replace=False)
        ret_sample = [retain_seqs[i] for i in idx]

        seqs = forget_seqs + ret_sample
        labels = [1] * len(forget_seqs) + [0] * len(ret_sample)
        tasks["forget_vs_retain"] = (seqs, labels,
            f"Forget vs Retain ({len(forget_seqs)} forget, {len(ret_sample)} retain)")

    return tasks


# ── Embedding extraction ─────────────────────────────────────────────

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
    """Logistic regression with cross-validation using balanced accuracy."""
    lr = LogisticRegression(max_iter=1000)
    scores = cross_val_score(lr, embeddings, labels, cv=cv, scoring='balanced_accuracy')
    return float(scores.mean()), float(scores.std())


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Linear probe evaluation for SGTM")
    parser.add_argument("--model-size", type=str, default="8M")
    parser.add_argument("--models-dir", default="models/sgtm")
    parser.add_argument("--data-dir", default="data/sgtm")
    parser.add_argument("--output-dir", default="results/sgtm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated list of run names to evaluate (default: all)")
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

    # Build probe tasks from data splits
    print(f"Building probe tasks from {args.data_dir}...")
    tasks = build_probe_tasks(args.data_dir)
    if not tasks:
        print("No probe tasks could be built. Check data directory.")
        return

    for name, (seqs, labels, desc) in tasks.items():
        print(f"  {name}: {desc}")

    # Discover model conditions
    if args.runs:
        run_names = [r.strip() for r in args.runs.split(",")]
    else:
        run_names = None
    conditions = []
    for entry in sorted(os.listdir(args.models_dir)):
        ckpt = os.path.join(args.models_dir, entry, "final_model.pt")
        if os.path.exists(ckpt):
            if run_names is None or entry in run_names:
                conditions.append(entry)
    print(f"\nConditions: {conditions}")

    all_results = {}

    for cond_name in conditions:
        ckpt_path = os.path.join(args.models_dir, cond_name, "final_model.pt")
        print(f"\n{'=' * 50}")
        print(f"Condition: {cond_name}")
        print(f"{'=' * 50}")

        model, _ = load_model_from_checkpoint(cfg, ckpt_path, args.device)

        # Pre-ablation probes
        cond_results = {}
        for task_name, (seqs, labels, desc) in tasks.items():
            emb = extract_embeddings(model, alphabet, seqs, args.device, args.batch_size)
            acc, std = logreg_cv(emb, labels)
            print(f"  {task_name}: {acc:.3f} +/- {std:.3f}")
            cond_results[task_name] = {"mean": acc, "std": std}

        all_results[cond_name] = cond_results

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

            abl_name = f"{cond_name}_ablated"
            abl_results = {}
            print(f"\n  [post-ablation]")

            for task_name, (seqs, labels, desc) in tasks.items():
                emb = extract_embeddings(model, alphabet, seqs, args.device, args.batch_size)
                acc, std = logreg_cv(emb, labels)
                print(f"  {task_name}: {acc:.3f} +/- {std:.3f}")
                abl_results[task_name] = {"mean": acc, "std": std}

            all_results[abl_name] = abl_results

    # Summary table
    task_labels = {name: name.replace("_", " ").title() for name in tasks}
    header = f"{'Condition':<35}" + "".join(f" {label:>20}" for label in task_labels.values())
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for name, r in all_results.items():
        row = f"{name:<35}"
        for task_name in task_labels:
            if task_name in r:
                row += f" {r[task_name]['mean']:.3f}±{r[task_name]['std']:.3f}      "
            else:
                row += f" {'N/A':>20}"
        print(row)
    print(f"{'=' * len(header)}")

    # Save
    size_tag = args.model_size.lower().replace("-", "")
    out_path = os.path.join(args.output_dir, f"linear_probe_results_{size_tag}.json")

    # Include task metadata in output
    output = {
        "task_descriptions": {name: desc for name, (_, _, desc) in tasks.items()},
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    if use_wandb:
        for name, r in all_results.items():
            for task_name, scores in r.items():
                wandb.log({f"probe/{name}/{task_name}": scores["mean"]})
        wandb.finish()


if __name__ == "__main__":
    main()
