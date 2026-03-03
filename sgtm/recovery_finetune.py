"""
Fine-tuning recovery experiment for SGTM.

Tests how quickly a holdout (data-filtered) model recovers viral protein
capability when fine-tuned on the forget set. This measures tamper-resistance:
if an adversary with limited compute can recover the filtered capability,
data filtering is not a robust safeguard.

Protocol:
  1. Load a trained holdout checkpoint
  2. Fine-tune on forget data only (999 human-infecting viral proteins)
  3. Evaluate PPL on forget/adjacent/retain every N steps
  4. Log curves to wandb and save results JSON

Usage:
  python -m sgtm.recovery_finetune --model-size 8M --checkpoint models/sgtm/holdout/final_model.pt --device cuda
  python -m sgtm.recovery_finetune --model-size 35M --checkpoint models/sgtm/holdout/final_model.pt --device cuda
  python -m sgtm.recovery_finetune --model-size 8M --checkpoint models/sgtm/holdout/final_model.pt --device cpu --batch-size 4 --max-steps 500  # local test
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtm.data_pipeline import MLMCollator
from sgtm.model_config import get_config, load_alphabet, load_model_from_checkpoint

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_mlm_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    output = model(input_ids, repr_layers=[], return_contacts=False)
    logits = output["logits"]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def evaluate(model, loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            loss = compute_mlm_loss(model, batch, device)
            total_loss += loss.item()
            n += 1
    model.train()
    avg_loss = total_loss / max(n, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Recovery fine-tuning experiment")
    parser.add_argument("--model-size", type=str, required=True, help="8M or 35M")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to holdout final_model.pt")
    parser.add_argument("--data-dir", default="data/sgtm")
    parser.add_argument("--output-dir", default="results/sgtm/recovery")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Fine-tuning hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than pretraining to avoid catastrophic forgetting of retain)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max fine-tuning steps (expect recovery well before this)")
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Evaluation frequency
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Evaluate every N steps (frequent for recovery curves)")
    parser.add_argument("--num-workers", type=int, default=2)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    cfg = get_config(args.model_size)
    if args.wandb_project is None:
        args.wandb_project = f"sgtm-esm2-{cfg.name.lower()}"

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model from holdout checkpoint
    # ------------------------------------------------------------------
    print(f"Loading {cfg.name} from {args.checkpoint}...")
    model, alphabet = load_model_from_checkpoint(cfg, args.checkpoint, args.device)
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Data: fine-tune on forget train, evaluate on all val splits
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    collator = MLMCollator(alphabet, mask_ratio=args.mask_ratio, max_length=args.max_length)

    forget_train = load_from_disk(os.path.join(args.data_dir, "forget", "train"))
    forget_val = load_from_disk(os.path.join(args.data_dir, "forget", "val"))
    retain_val = load_from_disk(os.path.join(args.data_dir, "retain", "val"))

    print(f"  forget train: {len(forget_train)} sequences")
    print(f"  forget val:   {len(forget_val)}")

    has_adjacent = os.path.isdir(os.path.join(args.data_dir, "adjacent", "val"))
    if has_adjacent:
        adjacent_val = load_from_disk(os.path.join(args.data_dir, "adjacent", "val"))
        print(f"  adjacent val: {len(adjacent_val)}")
    else:
        print("  adjacent val: not present (coarse task)")

    print(f"  retain val:   {len(retain_val)}")

    pin = args.device != "cpu"
    train_loader = DataLoader(
        forget_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=pin, drop_last=True,
    )

    val_loaders = {
        "forget": DataLoader(forget_val, batch_size=args.batch_size,
                             collate_fn=collator, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin),
        "retain": DataLoader(retain_val, batch_size=args.batch_size,
                             collate_fn=collator, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin),
    }
    if has_adjacent:
        val_loaders["adjacent"] = DataLoader(
            adjacent_val, batch_size=args.batch_size,
            collate_fn=collator, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin,
        )

    # ------------------------------------------------------------------
    # Evaluate baseline (before any fine-tuning)
    # ------------------------------------------------------------------
    print("\nBaseline evaluation (step 0)...")
    history = []
    baseline = {"step": 0}
    for name, loader in val_loaders.items():
        loss = evaluate(model, loader, args.device)
        ppl = math.exp(min(loss, 20))  # cap to avoid overflow
        baseline[f"{name}_loss"] = loss
        baseline[f"{name}_ppl"] = ppl
        print(f"  {name}: loss={loss:.4f}, ppl={ppl:.2f}")
    history.append(baseline)

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    run_name = args.run_name or f"recovery-{cfg.name}-holdout"
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group="recovery",
            config={
                **vars(args),
                "experiment": "recovery_finetune",
                "source_checkpoint": args.checkpoint,
                "model_size": cfg.name,
            },
        )
        # Log baseline
        wandb.log({f"recovery/{k}": v for k, v in baseline.items()}, step=0)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=0)

    # ------------------------------------------------------------------
    # Fine-tuning loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RECOVERY FINE-TUNING: {cfg.name} holdout -> forget data")
    print(f"LR={args.lr}, batch={args.batch_size}, max_steps={args.max_steps}")
    print(f"Eval every {args.eval_interval} steps")
    print(f"{'='*60}\n")

    train_iter = iter(train_loader)
    running_loss = 0.0
    start_time = time.time()

    for step in tqdm(range(1, args.max_steps + 1), desc="Recovery fine-tuning"):
        # Get batch (cycle through forget data)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        loss = compute_mlm_loss(model, batch, args.device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Evaluate
        if step % args.eval_interval == 0 or step == args.max_steps:
            avg_train_loss = running_loss / args.eval_interval
            running_loss = 0.0

            record = {"step": step, "train_loss": avg_train_loss}
            eval_msg = f"step {step}: train={avg_train_loss:.4f}"

            for name, loader in val_loaders.items():
                val_loss = evaluate(model, loader, args.device)
                ppl = math.exp(min(val_loss, 20))
                record[f"{name}_loss"] = val_loss
                record[f"{name}_ppl"] = ppl
                eval_msg += f" | {name}={val_loss:.4f} (ppl {ppl:.1f})"

            tqdm.write(eval_msg)
            history.append(record)

            if use_wandb:
                log_dict = {f"recovery/{k}": v for k, v in record.items()}
                log_dict["recovery/lr"] = optimizer.param_groups[0]["lr"]
                wandb.log(log_dict, step=step)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time

    split_names = list(val_loaders.keys())
    results = {
        "experiment": "recovery_finetune",
        "model_size": cfg.name,
        "source_checkpoint": args.checkpoint,
        "hyperparameters": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
        },
        "baseline_ppl": {s: baseline[f"{s}_ppl"] for s in split_names},
        "final_ppl": {s: history[-1][f"{s}_ppl"] for s in split_names},
        "history": history,
        "total_time_s": elapsed,
    }

    out_path = os.path.join(args.output_dir, f"recovery_{cfg.name.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed/60:.1f} minutes")

    # Summary
    print(f"\n{'='*60}")
    print("RECOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Baseline':>12} {'Final':>12} {'Change':>12}")
    print(f"{'-'*60}")
    for split in split_names:
        b = baseline[f"{split}_ppl"]
        f_val = history[-1][f"{split}_ppl"]
        if b > 0:
            change = (f_val - b) / b * 100
            print(f"{split:<20} {b:>12.2f} {f_val:>12.2f} {change:>+11.1f}%")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
