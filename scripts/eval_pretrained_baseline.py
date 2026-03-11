"""
Evaluate pretrained ESM-2 models (from Meta, trained on UniRef50) on SGTM data splits.

This provides a baseline: how well does a full pretrained ESM-2 model
predict Coronaviridae sequences? Data filtering and SGTM should be
compared against this ceiling.

Note: our SGTM models are trained FROM SCRATCH (not fine-tuned from pretrained),
so pretrained ESM-2 will have much better PPL. The comparison shows:
  - Pretrained ESM-2: best possible PPL (saw all data including viral)
  - Data filtering (holdout): trained from scratch without forget data
  - SGTM: trained from scratch with gradient masking
  - SGTM ablated: after zeroing forget parameters

Usage:
  python scripts/eval_pretrained_baseline.py --model-size 8M --data-dir data/sgtm/family_coronaviridae --output-dir results/sgtm_p2/family_coronaviridae --device cuda
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtm.data_pipeline import MLMCollator
from sgtm.model_config import load_alphabet

# Map our size names to ESM-2 pretrained model names
PRETRAINED_MODELS = {
    "8M": "esm2_t6_8M_UR50D",
    "35M": "esm2_t12_35M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}


def evaluate_ppl(model, loader, device, max_batches=200):
    """Compute MLM perplexity."""
    import torch.nn as nn
    model.eval()
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, repr_layers=[], return_contacts=False)
            logits = output["logits"]
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    return math.exp(min(avg_loss, 20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1022)
    args = parser.parse_args()

    import esm
    model_name = PRETRAINED_MODELS[args.model_size]
    print(f"Loading pretrained {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(f"esm2_{model_name.split('esm2_')[0]}" if "esm2_" not in model_name else model_name)

    # Use the standard esm.pretrained loader
    load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = load_fn()
    model = model.to(args.device)
    model.eval()
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data splits
    collator = MLMCollator(alphabet, mask_ratio=0.15, max_length=args.max_length, seed=42)
    pin = args.device != "cpu"

    results = {}
    for split in ("forget", "adjacent", "retain"):
        split_path = os.path.join(args.data_dir, split, "test")
        if not os.path.isdir(split_path):
            # Fall back to val
            split_path = os.path.join(args.data_dir, split, "val")
        if not os.path.isdir(split_path):
            print(f"  {split}: not found, skipping")
            continue

        ds = load_from_disk(split_path)
        loader = DataLoader(
            ds, batch_size=args.batch_size, collate_fn=collator, shuffle=False,
            num_workers=2, pin_memory=pin,
        )
        ppl = evaluate_ppl(model, loader, args.device)
        results[split] = ppl
        print(f"  {split}: PPL = {ppl:.2f} ({len(ds)} sequences)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    size_tag = args.model_size.lower()
    out_path = os.path.join(args.output_dir, f"pretrained_baseline_{size_tag}.json")
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "ppl": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
