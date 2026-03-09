"""
Poster-ready plots for SGTM family-task experiments.

Reads results JSON files from evaluate_sgtm.py and recovery_finetune.py,
produces publication-quality figures.

Usage:
  python -m sgtm.plot_results --results-dir results/sgtm_p2 --output-dir figures/
  python -m sgtm.plot_results --results-dir results/sgtm_p2 --output-dir figures/ --format pdf
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────
COLORS = {
    "forget": "#d62728",     # red
    "retain": "#2ca02c",     # green
    "adjacent": "#ff7f0e",   # orange
    "holdout": "#7f7f7f",    # gray
    "sgtm": "#1f77b4",       # blue
    "sgtm_ablated": "#9467bd",  # purple
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ── Plot 1: Ablation Ratio Bar Chart ──────────────────────────────────

def plot_ablation_ratios(results_files, output_path, title=None):
    """Bar chart: forget vs retain ablation ratios per condition.

    The money plot — shows whether SGTM achieves selective localization.
    Goal: forget ratio >> 1, retain ratio ≈ 1.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = []
    forget_ratios = []
    retain_ratios = []
    adjacent_ratios = []
    has_adjacent = False

    for path in results_files:
        with open(path) as f:
            data = json.load(f)

        for name, ratios in data.get("ablation_ratios", {}).items():
            conditions.append(_pretty_name(name))
            forget_ratios.append(ratios.get("forget", 1.0))
            retain_ratios.append(ratios.get("retain", 1.0))
            if "adjacent" in ratios:
                adjacent_ratios.append(ratios["adjacent"])
                has_adjacent = True
            else:
                adjacent_ratios.append(None)

    if not conditions:
        print("No ablation ratios found — skipping ablation ratio plot.")
        return

    x = np.arange(len(conditions))
    n_bars = 3 if has_adjacent else 2
    width = 0.7 / n_bars

    bars_f = ax.bar(x - width * (n_bars - 1) / 2, forget_ratios, width,
                    label="Forget", color=COLORS["forget"], edgecolor="white", linewidth=0.5)
    bars_r = ax.bar(x - width * (n_bars - 1) / 2 + width, retain_ratios, width,
                    label="Retain", color=COLORS["retain"], edgecolor="white", linewidth=0.5)
    if has_adjacent:
        adj_vals = [v if v is not None else 0 for v in adjacent_ratios]
        bars_a = ax.bar(x - width * (n_bars - 1) / 2 + 2 * width, adj_vals, width,
                        label="Adjacent", color=COLORS["adjacent"], edgecolor="white", linewidth=0.5)

    # Reference line at ratio = 1
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.5, label="No change")

    # Value labels on bars
    for bars in ([bars_f, bars_r] + ([bars_a] if has_adjacent else [])):
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.0f}×" if h >= 10 else f"{h:.1f}×",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Ablation Ratio (PPL post / PPL pre)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.set_title(title or "Ablation Ratios: Selective Knowledge Localization")

    # Clean up y-axis labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}×" if y >= 1 else f"{y:.2f}×"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 2: Scale Comparison ──────────────────────────────────────────

def plot_scale_comparison(results_files_by_size, output_path):
    """Faceted ablation ratios comparing 8M vs 35M (or more).

    results_files_by_size: dict like {"8M": [path1], "35M": [path2]}
    """
    sizes = sorted(results_files_by_size.keys(), key=lambda s: int(s.rstrip("M")))
    n_sizes = len(sizes)

    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5), sharey=True)
    if n_sizes == 1:
        axes = [axes]

    for ax, size in zip(axes, sizes):
        conditions = []
        forget_ratios = []
        retain_ratios = []

        for path in results_files_by_size[size]:
            with open(path) as f:
                data = json.load(f)
            for name, ratios in data.get("ablation_ratios", {}).items():
                conditions.append(_pretty_name(name))
                forget_ratios.append(ratios.get("forget", 1.0))
                retain_ratios.append(ratios.get("retain", 1.0))

        if not conditions:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(f"ESM-2 {size}")
            continue

        x = np.arange(len(conditions))
        width = 0.3
        ax.bar(x - width / 2, forget_ratios, width, label="Forget", color=COLORS["forget"])
        ax.bar(x + width / 2, retain_ratios, width, label="Retain", color=COLORS["retain"])
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.set_title(f"ESM-2 {size}")
        if ax == axes[0]:
            ax.set_ylabel("Ablation Ratio")
            ax.legend()

    fig.suptitle("SGTM Ablation: Does Scale Help?", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 3: Recovery Fine-Tuning Curve ────────────────────────────────

def plot_recovery_curve(recovery_json_path, output_path):
    """Recovery fine-tuning: how fast does holdout model recover forget capability?

    X = fine-tuning steps, Y = PPL (log scale), lines for forget/retain/adjacent.
    """
    with open(recovery_json_path) as f:
        data = json.load(f)

    history = data["history"]
    baseline = data.get("baseline_ppl", {})

    steps = [h["step"] for h in history]
    splits = [k.replace("_ppl", "") for k in history[0] if k.endswith("_ppl")]

    fig, ax = plt.subplots(figsize=(8, 5))

    for split in splits:
        key = f"{split}_ppl"
        vals = [h[key] for h in history]
        color = COLORS.get(split, "#333333")
        ax.plot(steps, vals, "-o", color=color, label=split.capitalize(),
                markersize=4, linewidth=2)

        # Baseline as dashed horizontal line
        if split in baseline:
            ax.axhline(y=baseline[split], color=color, linestyle=":", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Fine-tuning Steps")
    ax.set_ylabel("Perplexity")
    ax.set_title("Recovery Fine-Tuning: Reversing Data Filtering")
    ax.legend()

    # Annotation
    ax.annotate("Baseline PPL (dotted lines)",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, fontstyle="italic", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 4: Training Loss Curves ─────────────────────────────────────

def plot_training_curves(wandb_history_path, output_path):
    """Training loss over time for forget vs retain splits.

    Expects a JSON file with a list of {step, forget_loss, retain_loss, ...} entries,
    e.g. exported from wandb or logged during training.
    """
    with open(wandb_history_path) as f:
        history = json.load(f)

    if isinstance(history, dict) and "history" in history:
        history = history["history"]

    fig, ax = plt.subplots(figsize=(8, 5))

    steps = [h.get("step", i) for i, h in enumerate(history)]

    for key in ["forget_loss", "retain_loss", "adjacent_loss"]:
        if key not in history[0]:
            continue
        vals = [h[key] for h in history]
        split = key.replace("_loss", "")
        color = COLORS.get(split, "#333333")
        # Smooth with rolling average for readability
        window = max(1, len(vals) // 100)
        smoothed = np.convolve(vals, np.ones(window) / window, mode="valid")
        smooth_steps = steps[:len(smoothed)]
        ax.plot(smooth_steps, smoothed, color=color, label=split.capitalize(), linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("MLM Loss")
    ax.set_title("Training Loss by Data Category")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Helpers ───────────────────────────────────────────────────────────

def _pretty_name(name):
    """Clean up run names for plot labels."""
    name = name.replace("sgtm-family-coronaviridae-", "SGTM ")
    name = name.replace("sgtm-coarse-", "SGTM coarse ")
    name = name.replace("holdout-family-coronaviridae-", "Holdout ")
    name = name.replace("holdout-coarse-", "Holdout coarse ")
    return name.upper() if name.endswith("m") else name


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate poster-ready plots")
    parser.add_argument("--results-dir", default="results/sgtm_p2",
                        help="Directory containing results JSON files")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for plots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output format")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    results_dir = Path(args.results_dir)

    # Find all perplexity results
    ppl_files = sorted(results_dir.glob("**/perplexity_results*.json"))
    if ppl_files:
        print(f"\nFound {len(ppl_files)} perplexity results files")

        # Plot 1: Ablation ratios (all in one)
        plot_ablation_ratios(
            ppl_files,
            os.path.join(args.output_dir, f"ablation_ratios.{ext}"),
        )

        # Plot 2: Scale comparison (group by model size)
        by_size = {}
        for p in ppl_files:
            name = p.stem
            for size in ["8m", "35m", "150m"]:
                if size in str(p).lower():
                    by_size.setdefault(size.upper(), []).append(p)
                    break
            else:
                by_size.setdefault("unknown", []).append(p)

        if len(by_size) > 1:
            plot_scale_comparison(
                by_size,
                os.path.join(args.output_dir, f"scale_comparison.{ext}"),
            )
    else:
        print("No perplexity results found.")

    # Plot 3: Recovery curves
    recovery_files = sorted(results_dir.glob("**/recovery_*.json"))
    for rf in recovery_files:
        size_tag = rf.stem.replace("recovery_", "")
        plot_recovery_curve(
            rf,
            os.path.join(args.output_dir, f"recovery_{size_tag}.{ext}"),
        )

    if not ppl_files and not recovery_files:
        print(f"No results found in {results_dir}. Run experiments first.")
        return

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
