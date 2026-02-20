"""
Visualize SGTM 35M experiment results.

Usage:
  python -m sgtm.visualize_results --results-file results/sgtm/perplexity_results.json
"""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_pre_ablation(results, output_dir):
    """Bar chart of pre-ablation PPL across conditions and splits."""
    conditions = ["holdout", "sgtm-2head-attn-only", "sgtm-2head-zeroing"]
    labels = ["Holdout", "SGTM Attn-Only", "SGTM Attn+MLP"]
    splits = ["forget", "adjacent", "retain"]
    split_labels = ["Forget\n(human-infecting virus)", "Adjacent\n(non-human virus)", "Retain\n(non-viral)"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (split, split_label, color) in enumerate(zip(splits, split_labels, colors)):
        ppls = [results[c][split] for c in conditions]
        bars = ax.bar(x + i * width, ppls, width, label=split_label, color=color, edgecolor="white")
        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{ppl:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Perplexity (MLM)", fontsize=12)
    ax.set_title("ESM-2 35M: Pre-Ablation Perplexity", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, max(results[c][s] for c in conditions for s in splits) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "pre_ablation_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_effect(results, output_dir):
    """Log-scale bar chart showing ablation effect for SGTM conditions."""
    sgtm_conditions = [
        ("sgtm-2head-attn-only", "SGTM Attn-Only"),
        ("sgtm-2head-zeroing", "SGTM Attn+MLP"),
    ]
    splits = ["forget", "adjacent", "retain"]
    split_labels = ["Forget", "Adjacent", "Retain"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (cond, cond_label) in zip(axes, sgtm_conditions):
        pre = [results[cond][s] for s in splits]
        post = [results[f"{cond}_ablated"][s] for s in splits]

        x = np.arange(len(splits))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pre, width, label="Pre-ablation",
                       color="#3498db", edgecolor="white")
        bars2 = ax.bar(x + width / 2, post, width, label="Post-ablation",
                       color="#e74c3c", edgecolor="white")

        # Add value labels
        for bar, val in zip(bars1, pre):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, val in zip(bars2, post):
            exp = int(math.log10(val))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                    f"10^{exp}", ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#c0392b")

        ax.set_yscale("log")
        ax.set_title(cond_label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Perplexity (log scale)" if ax == axes[0] else "", fontsize=12)

    fig.suptitle("ESM-2 35M: Effect of Ablating Forget Parameters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_effect_log.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_8m_vs_35m_comparison(results, output_dir):
    """Side-by-side comparison of 8M and 35M ablation results.

    8M results hardcoded from the completed experiments.
    """
    # 8M results from prior experiments (attention-only ablation)
    results_8m = {
        "pre_ablation": {"forget": 13.93, "adjacent": 14.11, "retain": 14.75},
        "attn_only_ablated": {"forget": 14.85, "adjacent": 14.94, "retain": 14.84},
        "full_ablated": {"forget": 3.87e12, "adjacent": 4.12e12, "retain": 2.95e12},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: attention-only ablation comparison (8M vs 35M)
    ax = axes[0]
    categories = ["Pre-ablation\nRetain PPL", "Post-ablation\nRetain PPL"]
    vals_8m = [results_8m["pre_ablation"]["retain"], results_8m["attn_only_ablated"]["retain"]]
    vals_35m = [
        results.get("sgtm-2head-attn-only", {}).get("retain", 0),
        results.get("sgtm-2head-attn-only_ablated", {}).get("retain", 0),
    ]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width / 2, vals_8m, width, label="8M (3-head attn-only)",
                   color="#3498db", edgecolor="white")
    # For 35M, use log scale friendly values
    bars2_vals = [vals_35m[0], min(vals_35m[1], 1e6)]  # cap for display
    bars2 = ax.bar(x + width / 2, [vals_35m[0], vals_35m[0]], width,
                   label="35M (2-head attn-only)", color="#e67e22", edgecolor="white")

    # Annotate
    for bar, val in zip(bars1, vals_8m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(bars2[0].get_x() + bars2[0].get_width() / 2, bars2[0].get_height() + 0.3,
            f"{vals_35m[0]:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(bars2[1].get_x() + bars2[1].get_width() / 2, bars2[1].get_height() + 0.3,
            f"10^{int(math.log10(vals_35m[1]))}\n(destroyed)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c0392b")

    ax.set_title("Attention-Only Ablation:\nRetain PPL Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylabel("Retain Perplexity", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right panel: summary table as text
    ax = axes[1]
    ax.axis("off")
    table_data = [
        ["", "8M (6 layers)", "35M (12 layers)"],
        ["Pre-ablation retain PPL", "14.75", f"{results['sgtm-2head-attn-only']['retain']:.2f}"],
        ["Attn-only ablated retain", "14.84", f"5.3 x 10^22"],
        ["Attn-only ablation delta", "+0.6%", "CATASTROPHIC"],
        ["Full ablated retain", "2.95 x 10^12", f"7.7 x 10^31"],
        ["Full ablation delta", "CATASTROPHIC", "CATASTROPHIC"],
        ["", "", ""],
        ["Attention-only viable?", "YES", "NO"],
    ]

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style the key result rows
    for i in [3, 7]:
        for j in range(3):
            table[i, j].set_facecolor("#fdebd0")
            table[i, j].set_text_props(fontweight="bold")

    # Color catastrophic cells
    table[2, 2].set_text_props(color="#c0392b", fontweight="bold")
    table[3, 2].set_text_props(color="#c0392b", fontweight="bold")
    table[4, 1].set_text_props(color="#c0392b", fontweight="bold")
    table[4, 2].set_text_props(color="#c0392b", fontweight="bold")
    table[7, 1].set_text_props(color="#27ae60", fontweight="bold")
    table[7, 2].set_text_props(color="#c0392b", fontweight="bold")

    ax.set_title("8M vs 35M Ablation Summary", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "8m_vs_35m_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_holdout_vs_sgtm(results, output_dir):
    """Compare holdout retain PPL vs SGTM pre-ablation retain PPL."""
    conditions = ["holdout", "sgtm-2head-attn-only", "sgtm-2head-zeroing"]
    labels = ["Holdout\n(no viral data)", "SGTM Attn-Only\n(pre-ablation)", "SGTM Attn+MLP\n(pre-ablation)"]
    retain_ppls = [results[c]["retain"] for c in conditions]
    colors = ["#27ae60", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(conditions)), retain_ppls, color=colors, edgecolor="white", width=0.6)

    for bar, ppl in zip(bars, retain_ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Retain Perplexity", fontsize=12)
    ax.set_title("ESM-2 35M: Retain Set Performance (Pre-Ablation)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(retain_ppls) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation
    ax.annotate(
        f"SGTM retain cost: +{retain_ppls[1] - retain_ppls[0]:.1f} PPL\nvs holdout",
        xy=(1, retain_ppls[1]), xytext=(1.5, retain_ppls[0] + 0.5),
        fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "holdout_vs_sgtm_retain.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="results/sgtm/perplexity_results.json")
    parser.add_argument("--output-dir", default="results/sgtm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.results_file)

    plot_pre_ablation(results, args.output_dir)
    plot_ablation_effect(results, args.output_dir)
    plot_holdout_vs_sgtm(results, args.output_dir)
    plot_8m_vs_35m_comparison(results, args.output_dir)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
