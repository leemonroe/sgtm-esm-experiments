"""
SGTM training script for ESM-2 from scratch.

Three modes:
  baseline: all data, no gradient masking (control)
  holdout:  all data minus forget set (strict filtering control)
  sgtm:    all data, gradient masking localizes viral knowledge to forget params

Usage:
  python -m sgtm.train_sgtm --mode sgtm --model-size 8M --device cuda
  python -m sgtm.train_sgtm --mode sgtm --model-size 35M --device cuda
  python -m sgtm.train_sgtm --mode sgtm --model-size 8M --max-steps 5 --device cpu --batch-size 2  # smoke test
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sgtm.data_pipeline import MLMCollator
from sgtm.masking import (
    ablate, adjust_gradients, build_sgtm_masks, register_gradient_routing_hooks,
)
from sgtm.model_config import get_config, load_alphabet, create_model

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Multi-dataset loader (simplified from SGTM reference, single-GPU)
# ---------------------------------------------------------------------------

class MultiDataLoader:
    """Manages multiple DataLoaders, serving batches by dataset name with auto-reset."""

    def __init__(self, datasets, batch_size, collate_fn, shuffle=True,
                 num_workers=0, pin_memory=False):
        self.loaders = {}
        self.iterators = {}
        self.epochs = {}
        for name, ds in datasets.items():
            self.loaders[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, drop_last=True,
            )
            self.iterators[name] = iter(self.loaders[name])
            self.epochs[name] = 0

    def get_batch(self, name):
        try:
            return next(self.iterators[name])
        except StopIteration:
            self.epochs[name] += 1
            self.iterators[name] = iter(self.loaders[name])
            return next(self.iterators[name])


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def compute_mlm_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    output = model(input_ids, repr_layers=[], return_contacts=False)
    logits = output["logits"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


def evaluate(model, loader, device, max_batches=None):
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
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Build data_split_order
# ---------------------------------------------------------------------------

def build_data_split_order(
    n_forget, n_adjacent, n_ambiguous, n_retain,
    upsample_forget=50, upsample_adjacent=150,
    total_steps=None, seed=42,
):
    """
    Build a shuffled list of data source labels, one per training step.

    Upsampling factors ensure viral proteins appear frequently despite being
    a tiny fraction of the corpus. Labels map to SGTM modes in the training loop:
      "forget" -> forget mode, "adjacent" -> default mode, "retain" -> retain mode.
    """
    order = (
        ["forget"] * int(n_forget * upsample_forget)
        + ["adjacent"] * int(n_adjacent * upsample_adjacent)
        + ["retain"] * n_retain
    )
    # n_ambiguous kept in signature for backward compat but no longer used
    rng = random.Random(seed)
    rng.shuffle(order)

    if total_steps and len(order) > total_steps:
        order = order[:total_steps]

    return order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SGTM training for ESM-2")
    parser.add_argument("--mode", choices=["baseline", "holdout", "sgtm"], required=True)
    parser.add_argument("--model-size", type=str, default="8M",
                        help="Model size: 8M or 35M (default: 8M)")
    parser.add_argument("--data-dir", default="data/sgtm")
    parser.add_argument("--output-dir", default="models/sgtm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers (default: 4)")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Upsampling — default 1.0 (no upsampling) matches the original SGTM paper
    parser.add_argument("--upsample-forget", type=float, default=1)
    parser.add_argument("--upsample-adjacent", type=float, default=1)

    # SGTM configuration
    parser.add_argument("--forget-heads", type=str, default=None,
                        help="Comma-separated forget head indices (default: from model config)")
    parser.add_argument("--forget-mlp-start", type=int, default=None,
                        help="Start index of forget MLP dims (default: from model config)")
    parser.add_argument("--masking-strategy", choices=["gradient_zeroing", "gradient_routing"],
                        default="gradient_zeroing",
                        help="gradient_zeroing: zero grads post-backward; "
                             "gradient_routing: detach activations in forward pass")
    parser.add_argument("--mask-embeddings", action="store_true", default=False,
                        help="Zero embedding gradients in forget mode (matches SGTM paper)")

    # Retain mode split — controls what fraction of each data source
    # trains with retain mode vs default mode. The SGTM paper uses 10%
    # for retain and adjacent data (90% default, 10% retain), meaning
    # forget parameters still get gradient signal from most non-forget data.
    parser.add_argument("--retain-retain-perc", type=float, default=10,
                        help="Percent of retain-source steps that use retain mode (default: 10). "
                             "Remainder uses default mode (all params update).")
    parser.add_argument("--adjacent-retain-perc", type=float, default=10,
                        help="Percent of adjacent-source steps that use retain mode (default: 10)")
    parser.add_argument("--forget-retain-perc", type=float, default=0,
                        help="Percent of forget-source steps that use retain mode (default: 0)")

    # Logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=2000)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. models/sgtm/holdout/checkpoint_10000.pt)")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project name (default: sgtm-esm2-{model_size})")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")

    args = parser.parse_args()

    # Load model config and resolve defaults
    cfg = get_config(args.model_size)
    if args.forget_heads is None:
        args.forget_head_indices = list(cfg.default_forget_heads)
    else:
        args.forget_head_indices = [int(h) for h in args.forget_heads.split(",")]
    if args.forget_mlp_start is None:
        args.forget_mlp_start = cfg.default_forget_mlp_start
    if args.wandb_project is None:
        args.wandb_project = f"sgtm-esm2-{cfg.name.lower()}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build a descriptive run name
    if args.mode == "sgtm":
        n_heads = len(args.forget_head_indices)
        strategy_short = "routing" if args.masking_strategy == "gradient_routing" else "zeroing"
        default_run_name = f"sgtm-{n_heads}head-{strategy_short}"
    else:
        default_run_name = args.mode

    run_name = args.run_name or default_run_name
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"train-{run_name}",
            group="rerun-v2",
            config={
                **vars(args),
                "run_name": run_name,
                "model_size": cfg.name,
                "head_dim": cfg.head_dim,
                "mlp_dim": cfg.mlp_dim,
                "effective_batch_size": args.batch_size * args.grad_accum,
            },
        )
    elif not args.no_wandb:
        print("Warning: wandb not installed, logging disabled")

    # ------------------------------------------------------------------
    # Model (from scratch)
    # ------------------------------------------------------------------
    print(f"Initializing ESM-2 {cfg.name} from scratch (mode={args.mode})...")
    model, alphabet = create_model(cfg)
    model = model.to(args.device)
    print(f"Device: {args.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Masks and routing hooks (only for sgtm mode)
    # ------------------------------------------------------------------
    retain_mask, forget_mask = None, None
    routing_hooks, sgtm_mode_ref = None, None
    if args.mode == "sgtm":
        retain_mask, forget_mask = build_sgtm_masks(
            model, forget_head_indices=args.forget_head_indices,
            forget_mlp_start=args.forget_mlp_start,
            head_dim=cfg.head_dim,
        )
        print(f"SGTM masks built: {len(forget_mask)} parameter tensors")
        print(f"  Forget heads: {args.forget_head_indices} (head_dim={cfg.head_dim})")
        print(f"  Forget MLP start: {args.forget_mlp_start} / {cfg.mlp_dim}")
        print(f"  Strategy: {args.masking_strategy}")

        if args.masking_strategy == "gradient_routing":
            routing_hooks, sgtm_mode_ref = register_gradient_routing_hooks(
                model, forget_head_indices=args.forget_head_indices,
                forget_mlp_start=args.forget_mlp_start,
                head_dim=cfg.head_dim,
            )
            print(f"  Registered {len(routing_hooks)} gradient routing hooks")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    collator = MLMCollator(alphabet, mask_ratio=args.mask_ratio, max_length=args.max_length)

    # Discover available categories from the data directory
    # Supports both coarse (forget/retain) and fine (forget/adjacent/retain)
    has_adjacent = os.path.isdir(os.path.join(args.data_dir, "adjacent", "train"))

    forget_train = load_from_disk(os.path.join(args.data_dir, "forget", "train"))
    retain_train = load_from_disk(os.path.join(args.data_dir, "retain", "train"))

    forget_val = load_from_disk(os.path.join(args.data_dir, "forget", "val"))
    retain_val = load_from_disk(os.path.join(args.data_dir, "retain", "val"))

    print(f"  forget:    {len(forget_train)} train / {len(forget_val)} val")

    adjacent_train, adjacent_val = None, None
    if has_adjacent:
        adjacent_train = load_from_disk(os.path.join(args.data_dir, "adjacent", "train"))
        adjacent_val = load_from_disk(os.path.join(args.data_dir, "adjacent", "val"))
        print(f"  adjacent:  {len(adjacent_train)} train / {len(adjacent_val)} val")
    else:
        print(f"  adjacent:  (not present)")

    print(f"  retain:    {len(retain_train)} train / {len(retain_val)} val")

    # Build training data loaders
    if args.mode == "holdout":
        train_datasets = {"retain": retain_train}
        if has_adjacent:
            train_datasets["adjacent"] = adjacent_train
    else:
        train_datasets = {
            "forget": forget_train,
            "retain": retain_train,
        }
        if has_adjacent:
            train_datasets["adjacent"] = adjacent_train

    pin = args.device != "cpu"
    train_loader = MultiDataLoader(
        train_datasets, batch_size=args.batch_size,
        collate_fn=collator, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
    )

    # Validation loaders
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
    # data_split_order
    # ------------------------------------------------------------------
    n_adjacent = len(adjacent_train) if has_adjacent else 0
    if args.mode == "holdout":
        data_split_order = build_data_split_order(
            n_forget=0,
            n_adjacent=n_adjacent,
            n_ambiguous=0,
            n_retain=len(retain_train),
            upsample_forget=0,
            upsample_adjacent=args.upsample_adjacent if has_adjacent else 0,
            total_steps=args.max_steps,
            seed=args.seed,
        )
    else:
        data_split_order = build_data_split_order(
            n_forget=len(forget_train),
            n_adjacent=n_adjacent,
            n_ambiguous=0,
            n_retain=len(retain_train),
            upsample_forget=args.upsample_forget,
            upsample_adjacent=args.upsample_adjacent if has_adjacent else 0,
            total_steps=args.max_steps,
            seed=args.seed,
        )

    actual_steps = len(data_split_order)
    counts = Counter(data_split_order)
    print(f"\nTraining for {actual_steps} steps (data source schedule):")
    for label in ("forget", "adjacent", "retain"):
        if counts[label]:
            print(f"  {label}: {counts[label]} ({counts[label]/actual_steps*100:.1f}%)")

    if args.mode == "sgtm":
        print(f"\nSGTM mode assignment (probabilistic):")
        print(f"  retain-source steps: {args.retain_retain_perc:.0f}% retain mode, "
              f"{100 - args.retain_retain_perc:.0f}% default mode")
        if has_adjacent:
            print(f"  adjacent-source steps: {args.adjacent_retain_perc:.0f}% retain mode, "
                  f"{100 - args.adjacent_retain_perc:.0f}% default mode")
        print(f"  forget-source steps: {args.forget_retain_perc:.0f}% retain mode, "
              f"{100 - args.forget_retain_perc:.0f}% forget mode")
        print(f"  Embedding masking in forget mode: {args.mask_embeddings}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, actual_steps - args.warmup_steps),
                                eta_min=0)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    history = {"train_loss": [], "val_losses": [], "config": vars(args)}
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        history = ckpt["history"]
        # Restore RNG states
        torch.random.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        random.setstate(ckpt["python_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        print(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TRAINING: {args.mode}")
    print(f"LR={args.lr}, batch={args.batch_size}×{args.grad_accum}={args.batch_size*args.grad_accum}")
    if start_step > 0:
        print(f"RESUMING from step {start_step}")
    print(f"{'='*60}\n")

    model.train()
    running_loss = 0.0
    start_time = time.time()

    for step in tqdm(range(start_step, actual_steps), desc=f"Training ({args.mode})", initial=start_step, total=actual_steps):
        source = data_split_order[step]

        # Map source to SGTM mode, with probabilistic retain/default split
        # matching the original SGTM paper's approach. E.g. with
        # --retain-retain-perc 10, 90% of retain-source steps use default
        # mode (all params update) and 10% use retain mode (forget params
        # masked). This ensures forget params get general training signal.
        if args.mode == "sgtm":
            rng_val = random.random()
            if source == "forget":
                if rng_val < args.forget_retain_perc / 100:
                    sgtm_mode = "retain"
                else:
                    sgtm_mode = "forget"
            elif source == "retain":
                if rng_val < args.retain_retain_perc / 100:
                    sgtm_mode = "retain"
                else:
                    sgtm_mode = "default"
            else:  # "adjacent"
                if rng_val < args.adjacent_retain_perc / 100:
                    sgtm_mode = "retain"
                else:
                    sgtm_mode = "default"
        else:
            sgtm_mode = None

        # Set routing mode before forward pass (gradient routing only)
        if sgtm_mode_ref is not None:
            sgtm_mode_ref[0] = sgtm_mode or "default"

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            batch = train_loader.get_batch(source)
            loss = compute_mlm_loss(model, batch, args.device)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Apply gradient masking ONCE after all microbatches have accumulated.
        # Doing this inside the loop would zero gradients between microbatches,
        # causing only the last microbatch's signal to survive on masked params.
        if sgtm_mode and retain_mask is not None and args.masking_strategy == "gradient_zeroing":
            adjust_gradients(model, retain_mask, forget_mask, sgtm_mode)

        # Zero shared parameter gradients in forget mode so forget data
        # doesn't update token embeddings or layer norms (matches paper)
        if sgtm_mode == "forget" and args.mask_embeddings:
            if model.embed_tokens.weight.grad is not None:
                model.embed_tokens.weight.grad.zero_()
            # Zero all layer norm gradients (shared params, per paper)
            for layer in model.layers:
                for ln in (layer.self_attn_layer_norm, layer.final_layer_norm):
                    if ln.weight.grad is not None:
                        ln.weight.grad.zero_()
                    if ln.bias.grad is not None:
                        ln.bias.grad.zero_()
            # Post-encoder layer norm
            if hasattr(model, "emb_layer_norm_after") and model.emb_layer_norm_after is not None:
                if model.emb_layer_norm_after.weight.grad is not None:
                    model.emb_layer_norm_after.weight.grad.zero_()
                if model.emb_layer_norm_after.bias.grad is not None:
                    model.emb_layer_norm_after.bias.grad.zero_()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        running_loss += accum_loss

        # Logging
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            tqdm.write(
                f"step {step+1}/{actual_steps} | loss={avg_loss:.4f} | "
                f"lr={lr:.2e} | {steps_per_sec:.1f} steps/s | source={source}"
            )
            history["train_loss"].append({"step": step + 1, "loss": avg_loss})
            if use_wandb:
                wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=step + 1)
            running_loss = 0.0

        # Evaluation
        if (step + 1) % args.eval_interval == 0 or step == actual_steps - 1:
            val_results = {}
            for name, loader in val_loaders.items():
                val_results[name] = evaluate(model, loader, args.device)
            eval_msg = "  [eval] " + " ".join(
                f"{k}={v:.4f}" for k, v in val_results.items()
            )
            tqdm.write(eval_msg)

            # PPL for wandb charts (capped at exp(20) to avoid overflow; raw loss is the authoritative metric)
            val_ppl = {k: math.exp(min(v, 20)) for k, v in val_results.items()}

            history["val_losses"].append({"step": step + 1, **val_results})
            if use_wandb:
                log_dict = {}
                for k, v in val_results.items():
                    log_dict[f"val/{k}_loss"] = v
                    log_dict[f"val/{k}_ppl"] = val_ppl[k]
                wandb.log(log_dict, step=step + 1)

            # Ablate-eval-restore: evaluate post-ablation model (SGTM only)
            if args.mode == "sgtm" and forget_mask is not None:
                saved_state = {k: v.clone() for k, v in model.state_dict().items()}
                ablate(model, forget_mask)

                abl_results = {}
                for name, loader in val_loaders.items():
                    abl_results[name] = evaluate(model, loader, args.device)
                # PPL capped for display; raw loss saved to history and wandb as _loss metrics
                abl_ppl = {k: math.exp(min(v, 20)) for k, v in abl_results.items()}

                abl_msg = "  [ablated] " + " ".join(
                    f"{k}={v:.4f} (ppl {abl_ppl[k]:.1f})" for k, v in abl_results.items()
                )
                tqdm.write(abl_msg)

                history["val_losses"][-1].update({
                    f"ablated_{k}": v for k, v in abl_results.items()
                })
                if use_wandb:
                    abl_log = {}
                    for k, v in abl_results.items():
                        abl_log[f"val_ablated/{k}_loss"] = v
                        abl_log[f"val_ablated/{k}_ppl"] = abl_ppl[k]
                    wandb.log(abl_log, step=step + 1)

                # Restore original weights
                model.load_state_dict(saved_state)
                model.train()

            # Save best by retain val loss
            if val_results["retain"] < best_val_loss:
                best_val_loss = val_results["retain"]
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

        # Periodic save (full training state for resumability)
        if (step + 1) % args.save_interval == 0:
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if args.device != "cpu" else None,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            }, os.path.join(run_dir, f"checkpoint_{step+1}.pt"))

    # ------------------------------------------------------------------
    # Save final model + history
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))

    # Also save the masks for SGTM mode
    if forget_mask is not None:
        torch.save({"retain_mask": retain_mask, "forget_mask": forget_mask},
                    os.path.join(run_dir, "masks.pt"))

    history["total_time_s"] = time.time() - start_time
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if use_wandb:
        # Upload model artifacts
        artifact = wandb.Artifact(
            f"sgtm-{cfg.name.lower()}-{run_name}", type="model",
            metadata={"model_size": cfg.name, "mode": args.mode},
        )
        for fname in ("final_model.pt", "masks.pt", "training_history.json"):
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                artifact.add_file(fpath)
        wandb.log_artifact(artifact)
        wandb.finish()

    print(f"\nTraining complete. Saved to {run_dir}")
    print(f"Total time: {history['total_time_s']/3600:.1f} hours")


if __name__ == "__main__":
    main()
