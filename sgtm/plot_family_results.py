"""
Comprehensive visualization of Family (Coronaviridae) task results at 35M scale.

Loads results from JSON files and generates publication-quality figures covering:
  1. Pseudo-perplexity comparison (holdout vs SGTM vs SGTM-ablated + pretrained baseline)
  2. Linear probe classification (forget vs adjacent, forget vs retain)
  3. DMS fitness prediction (Spearman rho per dataset)
  4. Recovery fine-tuning curves (SGTM-ablated + holdout if available)
  5. Summary dashboard combining key results

Usage:
  python -m sgtm.plot_family_results
  python -m sgtm.plot_family_results --output-dir figures/family --format pdf
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np


# ── Style ────────────────────────────────────────────────────────────

PAL = {
    "forget":     "#E24A33",
    "retain":     "#348ABD",
    "adjacent":   "#FBC15E",
    "pretrained": "#8EBA42",
    "holdout":    "#988ED5",
    "sgtm":       "#E24A33",
    "sgtm_abl":   "#777777",
    "good":       "#2CA02C",
    "bad":        "#D62728",
}

CONDITION_COLORS = {
    "Pretrained ESM-2":  PAL["pretrained"],
    "Data Filtering":    PAL["holdout"],
    "SGTM":              PAL["sgtm"],
    "SGTM (ablated)":    PAL["sgtm_abl"],
}

BG_LIGHT = "#F7F7F7"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": BG_LIGHT,
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.alpha": 0.7,
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


# ── Data Loading ─────────────────────────────────────────────────────

def load_results(results_dir):
    """Load all JSON result files for the family task."""
    data = {}
    for fname in [
        "perplexity_results_35m.json",
        "perplexity_results_8m.json",
        "linear_probe_results_35m.json",
        "linear_probe_results_8m.json",
        "bioriskeval_mut_results_35m.json",
        "bioriskeval_mut_results_8m.json",
        "pretrained_baseline_35m.json",
        "pretrained_baseline_8m.json",
    ]:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                data[fname.replace(".json", "")] = json.load(f)

    # Recovery files (SGTM-ablated and holdout)
    for size in ["35m", "8m"]:
        rpath = os.path.join(results_dir, f"recovery-{size}", f"recovery_{size}.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                data[f"recovery_{size}"] = json.load(f)
        rpath_ho = os.path.join(results_dir, f"recovery-holdout-{size}", f"recovery_{size}.json")
        if os.path.exists(rpath_ho):
            with open(rpath_ho) as f:
                data[f"recovery_holdout_{size}"] = json.load(f)

    return data


# ── Plot 1: Perplexity Comparison ────────────────────────────────────

def plot_ppl_comparison(data, output_path, scale="35m"):
    """Grouped bar chart: PPL across conditions for forget/adjacent/retain."""
    ppl_data = data[f"perplexity_results_{scale}"]
    pretrained = data.get(f"pretrained_baseline_{scale}", {})

    fig, ax = plt.subplots(figsize=(12, 6))

    # Build condition data
    conditions = []
    ppl_by_cond = {}

    if pretrained:
        conditions.append("Pretrained ESM-2")
        ppl_by_cond["Pretrained ESM-2"] = pretrained.get("ppl", {})

    key_map = {
        f"holdout-{scale}": "Data Filtering",
        f"sgtm-ret25-{scale}": "SGTM",
        f"sgtm-ret25-{scale}_ablated": "SGTM (ablated)",
    }
    for key, label in key_map.items():
        if key in ppl_data.get("absolute_ppl", {}):
            conditions.append(label)
            ppl_by_cond[label] = ppl_data["absolute_ppl"][key]

    splits = ["forget", "adjacent", "retain"]
    split_labels = ["Forget\n(Coronaviridae)", "Adjacent\n(other viral)", "Retain\n(non-viral)"]
    split_colors = [PAL["forget"], PAL["adjacent"], PAL["retain"]]

    x = np.arange(len(conditions))
    n = len(splits)
    width = 0.22

    for i, (split, slabel, scolor) in enumerate(zip(splits, split_labels, split_colors)):
        vals = [ppl_by_cond[c].get(split, 0) for c in conditions]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=slabel, color=scolor,
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel("Pseudo-Perplexity")
    ax.set_title(f"Coronaviridae Task: Pseudo-Perplexity ({scale.upper()})\n"
                 f"Goal: high forget PPL (capability removed) + low retain PPL (preserved)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(max(ppl_by_cond[c].get(s, 0) for s in splits) for c in conditions) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 2: Linear Probe Results ─────────────────────────────────────

def plot_linear_probes(data, output_path, scale="35m"):
    """Grouped bar chart with error bars for linear probe classification."""
    probe_data = data[f"linear_probe_results_{scale}"]
    results = probe_data["results"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    tasks = ["forget_vs_all", "forget_vs_adjacent", "forget_vs_retain"]
    task_titles = [
        "Coronaviridae vs All",
        "Coronaviridae vs\nOther Viral",
        "Coronaviridae vs\nNon-Viral",
    ]

    key_map = {
        f"holdout-{scale}": "Data Filtering",
        f"sgtm-ret25-{scale}": "SGTM",
        f"sgtm-ret25-{scale}_ablated": "SGTM (ablated)",
    }

    for ax, task, title in zip(axes, tasks, task_titles):
        cond_labels = []
        means = []
        stds = []
        colors = []

        for key, label in key_map.items():
            if key in results and task in results[key]:
                cond_labels.append(label)
                means.append(results[key][task]["mean"])
                stds.append(results[key][task]["std"])
                colors.append(CONDITION_COLORS.get(label, "#333"))

        x = np.arange(len(cond_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                      color=colors, edgecolor="white", linewidth=1)

        # Value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Chance line
        ax.axhline(y=0.5, color="#AAAAAA", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(len(cond_labels) - 0.5, 0.51, "chance", fontsize=9, color="#AAAAAA")

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=10, rotation=15, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0.3, 1.05)

    fig.suptitle(f"Linear Probe Classification ({scale.upper()}): Coronaviridae Task\n"
                 f"Can a linear classifier detect Coronaviridae from embeddings?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 3: DMS Fitness Prediction ───────────────────────────────────

def plot_dms_results(data, output_path, scale="35m"):
    """Per-dataset |Spearman rho| comparison across conditions."""
    dms_data = data[f"bioriskeval_mut_results_{scale}"]

    fig, ax = plt.subplots(figsize=(14, 7))

    key_map = {
        f"holdout-{scale}": "Data Filtering",
        f"sgtm-ret25-{scale}": "SGTM",
        f"sgtm-ret25-{scale}_ablated": "SGTM (ablated)",
    }

    # Get dataset names from first condition
    first_key = list(key_map.keys())[0]
    datasets = sorted(dms_data[first_key]["per_dataset"].keys())
    # Shorten dataset names for display
    short_names = [d.replace(".csv", "").split("_")[0] + "\n" + "_".join(d.replace(".csv", "").split("_")[1:3])
                   for d in datasets]

    x = np.arange(len(datasets))
    n_conds = len(key_map)
    width = 0.25

    for i, (key, label) in enumerate(key_map.items()):
        if key not in dms_data:
            continue
        vals = [dms_data[key]["per_dataset"][d]["spearman_abs"] for d in datasets]
        offset = (i - (n_conds - 1) / 2) * width
        color = CONDITION_COLORS.get(label, "#333")
        ax.bar(x + offset, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"BioRiskEval-Mut: DMS Fitness Prediction ({scale.upper()})\n"
                 f"|Spearman ρ| between masked marginal scores and experimental fitness",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.35)

    # Add mean lines
    for key, label in key_map.items():
        if key in dms_data:
            mean_rho = dms_data[key]["mean_abs_spearman"]
            color = CONDITION_COLORS.get(label, "#333")
            ax.axhline(y=mean_rho, color=color, linestyle=":", alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 4: DMS Summary Bar ─────────────────────────────────────────

def plot_dms_summary(data, output_path, scale="35m"):
    """Simple bar chart of mean |Spearman| across conditions."""
    dms_data = data[f"bioriskeval_mut_results_{scale}"]

    fig, ax = plt.subplots(figsize=(8, 5))

    key_map = {
        f"holdout-{scale}": "Data Filtering",
        f"sgtm-ret25-{scale}": "SGTM",
        f"sgtm-ret25-{scale}_ablated": "SGTM (ablated)",
    }

    labels = []
    means = []
    medians = []
    colors = []

    for key, label in key_map.items():
        if key in dms_data:
            labels.append(label)
            means.append(dms_data[key]["mean_abs_spearman"])
            medians.append(dms_data[key]["median_abs_spearman"])
            colors.append(CONDITION_COLORS.get(label, "#333"))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, means, width, label="Mean |ρ|", color=colors,
           edgecolor="white", linewidth=1)
    ax.bar(x + width / 2, medians, width, label="Median |ρ|", color=colors,
           edgecolor="white", linewidth=1, alpha=0.6)

    for i, (m, md) in enumerate(zip(means, medians)):
        ax.text(i - width / 2, m + 0.003, f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + width / 2, md + 0.003, f"{md:.3f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title(f"DMS Fitness Prediction: Aggregate ({scale.upper()})\n"
                 f"Mean and median |Spearman ρ| across 14 viral DMS datasets",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(means) * 1.4)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 5: Recovery Fine-Tuning ─────────────────────────────────────

def plot_recovery(data, output_path, scale="35m"):
    """Recovery curves: PPL over fine-tuning steps.

    Shows SGTM-ablated and holdout recovery side-by-side.
    Adds pretrained ESM-2 baseline as reference lines.
    """
    pretrained = data.get(f"pretrained_baseline_{scale}", {})
    ppl_data = data.get(f"perplexity_results_{scale}", {})

    fig, ax = plt.subplots(figsize=(12, 7))

    y_max = 0

    # Plot SGTM-ablated recovery if available
    sgtm_rec = data.get(f"recovery_{scale}")
    if sgtm_rec and "history" in sgtm_rec:
        history = sgtm_rec["history"]
        steps = [h["step"] for h in history]
        forget_ppl = [h["forget_ppl"] for h in history]
        retain_ppl = [h["retain_ppl"] for h in history]
        ax.plot(steps, forget_ppl, "-o", color=PAL["sgtm"], linewidth=2.5,
                markersize=5, label="SGTM (ablated) → Forget PPL", zorder=5)
        ax.plot(steps, retain_ppl, "-s", color=PAL["sgtm"], linewidth=2.5,
                markersize=5, alpha=0.5, label="SGTM (ablated) → Retain PPL", zorder=4)
        y_max = max(y_max, max(forget_ppl), max(retain_ppl))

    # Plot holdout recovery if available
    holdout_rec = data.get(f"recovery_holdout_{scale}")
    if holdout_rec and "history" in holdout_rec:
        history = holdout_rec["history"]
        steps_h = [h["step"] for h in history]
        forget_h = [h["forget_ppl"] for h in history]
        retain_h = [h["retain_ppl"] for h in history]
        ax.plot(steps_h, forget_h, "-^", color=PAL["holdout"], linewidth=2.5,
                markersize=5, label="Data Filtering → Forget PPL", zorder=5)
        ax.plot(steps_h, retain_h, "-D", color=PAL["holdout"], linewidth=2.5,
                markersize=5, alpha=0.5, label="Data Filtering → Retain PPL", zorder=4)
        y_max = max(y_max, max(forget_h), max(retain_h))

    # Reference lines
    if pretrained and "ppl" in pretrained:
        pt_forget = pretrained["ppl"]["forget"]
        pt_retain = pretrained["ppl"]["retain"]
        ax.axhline(y=pt_forget, color=PAL["pretrained"], linestyle="--",
                   alpha=0.7, linewidth=2, zorder=3, label=f"Pretrained forget = {pt_forget:.1f}")
        ax.axhline(y=pt_retain, color=PAL["pretrained"], linestyle="--",
                   alpha=0.4, linewidth=2, zorder=3, label=f"Pretrained retain = {pt_retain:.1f}")

    # Pre-ablation SGTM reference
    if ppl_data:
        sgtm_key = f"sgtm-ret25-{scale}"
        if sgtm_key in ppl_data.get("absolute_ppl", {}):
            pre_forget = ppl_data["absolute_ppl"][sgtm_key]["forget"]
            ax.axhline(y=pre_forget, color=PAL["forget"], linestyle=":",
                       alpha=0.3, linewidth=1.5)

    max_step = max(
        [h["step"] for h in sgtm_rec["history"]] if sgtm_rec else [2000],
    )

    ax.set_xlabel("Recovery Fine-Tuning Steps")
    ax.set_ylabel("Pseudo-Perplexity")
    ax.set_title(f"Recovery Fine-Tuning ({scale.upper()}): Can Ablation Be Reversed?\n"
                 f"Fine-tuning on Coronaviridae sequences after data filtering / SGTM ablation",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 40)
    ax.set_xlim(-30, max_step * 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 6: Ablation Ratio Summary ───────────────────────────────────

def plot_ablation_ratios(data, output_path, scale="35m"):
    """Bar chart showing ablation ratios: forget vs adjacent vs retain.

    Goal: forget ratio >> retain ratio (selective). Reality: all similar.
    """
    ppl_data = data.get(f"perplexity_results_{scale}", {})
    ratios = ppl_data.get("ablation_ratios", {})

    sgtm_key = f"sgtm-ret25-{scale}"
    if sgtm_key not in ratios:
        print(f"No ablation ratios for {scale}, skipping")
        return

    r = ratios[sgtm_key]
    fig, ax = plt.subplots(figsize=(8, 5))

    splits = ["forget", "adjacent", "retain"]
    split_labels = ["Forget\n(Coronaviridae)", "Adjacent\n(other viral)", "Retain\n(non-viral)"]
    split_colors = [PAL["forget"], PAL["adjacent"], PAL["retain"]]
    vals = [r[s] for s in splits]

    bars = ax.bar(range(len(splits)), vals, width=0.6, color=split_colors,
                  edgecolor="white", linewidth=1.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                f"{v:.2f}×", ha="center", fontsize=13, fontweight="bold")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(2.5, 1.03, "No change", fontsize=10, color="#888", ha="right")

    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(split_labels, fontsize=12)
    ax.set_ylabel("Ablation Ratio (post / pre PPL)")
    ax.set_title(f"Ablation Ratios ({scale.upper()}): Is Forget Selectively Damaged?\n"
                 f"Goal: forget ratio >> retain ratio. Actual: retain damaged MORE than forget.",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Plot 7: Grand Summary Dashboard ─────────────────────────────────

def plot_dashboard(data, output_path, scale="35m"):
    """6-panel dashboard covering the full results taxonomy."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ppl_data = data.get(f"perplexity_results_{scale}", {})
    probe_data = data.get(f"linear_probe_results_{scale}", {})
    dms_data = data.get(f"bioriskeval_mut_results_{scale}", {})
    pretrained = data.get(f"pretrained_baseline_{scale}", {})
    recovery = data.get(f"recovery_{scale}", {})

    key_map = {
        f"holdout-{scale}": "Data\nFiltering",
        f"sgtm-ret25-{scale}": "SGTM",
        f"sgtm-ret25-{scale}_ablated": "SGTM\n(ablated)",
    }

    # ── A: PPL bars ──
    ax_a = fig.add_subplot(gs[0, 0])
    conditions = []
    cond_data = {}
    if pretrained and "ppl" in pretrained:
        conditions.append("Pretrained")
        cond_data["Pretrained"] = pretrained["ppl"]
    for key, label in key_map.items():
        if key in ppl_data.get("absolute_ppl", {}):
            conditions.append(label)
            cond_data[label] = ppl_data["absolute_ppl"][key]

    x_a = np.arange(len(conditions))
    width_a = 0.25
    for i, (split, color) in enumerate(zip(["forget", "retain"],
                                           [PAL["forget"], PAL["retain"]])):
        vals = [cond_data[c].get(split, 0) for c in conditions]
        offset = (i - 0.5) * width_a
        bars = ax_a.bar(x_a + offset, vals, width_a, color=color,
                        edgecolor="white", linewidth=0.8,
                        label="Forget" if split == "forget" else "Retain")
        for bar, v in zip(bars, vals):
            ax_a.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                      f"{v:.1f}", ha="center", fontsize=8, fontweight="bold", color="#444")
    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(conditions, fontsize=9)
    ax_a.set_ylabel("PPL")
    ax_a.set_title("A. Pseudo-Perplexity", fontweight="bold")
    ax_a.legend(fontsize=9, loc="upper left")

    # ── B: Ablation Ratios ──
    ax_b = fig.add_subplot(gs[0, 1])
    ratios = ppl_data.get("ablation_ratios", {})
    sgtm_key = f"sgtm-ret25-{scale}"
    if sgtm_key in ratios:
        r = ratios[sgtm_key]
        splits = ["forget", "adjacent", "retain"]
        vals = [r[s] for s in splits]
        colors_b = [PAL["forget"], PAL["adjacent"], PAL["retain"]]
        bars = ax_b.bar(range(3), vals, width=0.6, color=colors_b, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                      f"{v:.2f}×", ha="center", fontsize=10, fontweight="bold")
        ax_b.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax_b.set_xticks(range(3))
        ax_b.set_xticklabels(["Forget", "Adjacent", "Retain"], fontsize=10)
        ax_b.set_ylabel("Ablation Ratio")
    ax_b.set_title("B. Ablation Ratios (NOT Selective)", fontweight="bold", color=PAL["bad"])

    # ── C: Linear Probes ──
    ax_c = fig.add_subplot(gs[0, 2])
    if probe_data:
        results = probe_data["results"]
        task = "forget_vs_retain"
        cond_labels = []
        means = []
        stds = []
        colors_c = []
        for key, label in key_map.items():
            if key in results and task in results[key]:
                cond_labels.append(label.replace("\n", " "))
                means.append(results[key][task]["mean"])
                stds.append(results[key][task]["std"])
                colors_c.append(CONDITION_COLORS.get(label.replace("\n", " "),
                                CONDITION_COLORS.get(label.replace("\n", "\n"), "#333")))
        x_c = np.arange(len(cond_labels))
        # Use condition colors
        bar_colors = [PAL["holdout"], PAL["sgtm"], PAL["sgtm_abl"]][:len(cond_labels)]
        bars = ax_c.bar(x_c, means, yerr=stds, capsize=5, width=0.55,
                        color=bar_colors, edgecolor="white")
        for bar, m in zip(bars, means):
            ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                      f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax_c.axhline(y=0.5, color="#AAA", linestyle="--", linewidth=1, alpha=0.7)
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(cond_labels, fontsize=9)
        ax_c.set_ylabel("Balanced Accuracy")
        ax_c.set_ylim(0.3, 1.05)
    ax_c.set_title("C. Linear Probe (Corona. vs Non-Viral)", fontweight="bold")

    # ── D: DMS Summary ──
    ax_d = fig.add_subplot(gs[1, 0])
    if dms_data:
        dms_labels = []
        dms_means = []
        dms_colors = []
        for key, label in key_map.items():
            if key in dms_data:
                dms_labels.append(label)
                dms_means.append(dms_data[key]["mean_abs_spearman"])
        dms_colors = [PAL["holdout"], PAL["sgtm"], PAL["sgtm_abl"]][:len(dms_labels)]
        x_d = np.arange(len(dms_labels))
        bars = ax_d.bar(x_d, dms_means, width=0.55, color=dms_colors, edgecolor="white")
        for bar, v in zip(bars, dms_means):
            ax_d.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                      f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax_d.set_xticks(x_d)
        ax_d.set_xticklabels(dms_labels, fontsize=9)
        ax_d.set_ylabel("Mean |Spearman ρ|")
    ax_d.set_title("D. DMS Fitness Prediction", fontweight="bold")

    # ── E: Recovery Curve ──
    ax_e = fig.add_subplot(gs[1, 1])
    if recovery and "history" in recovery:
        history = recovery["history"]
        steps = [h["step"] for h in history]
        forget_ppl = [h["forget_ppl"] for h in history]
        retain_ppl = [h["retain_ppl"] for h in history]
        ax_e.plot(steps, forget_ppl, "-o", color=PAL["forget"], linewidth=2,
                  markersize=4, label="Forget PPL")
        ax_e.plot(steps, retain_ppl, "-s", color=PAL["retain"], linewidth=2,
                  markersize=4, label="Retain PPL")
        if pretrained and "ppl" in pretrained:
            ax_e.axhline(y=pretrained["ppl"]["forget"], color=PAL["pretrained"],
                         linestyle="--", alpha=0.6, linewidth=1.5, label="Pretrained forget")
        ax_e.legend(fontsize=9, loc="upper right")
        ax_e.set_xlabel("Fine-Tuning Steps")
        ax_e.set_ylabel("PPL")
    ax_e.set_title("E. SGTM-Ablated Recovery", fontweight="bold")

    # ── F: Key Findings Text ──
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_axis_off()
    ax_f.set_title("F. Key Findings", fontweight="bold")

    findings = [
        "1. SGTM ablation is NOT selective:",
        "   Retain damaged 2.3× vs forget 1.6×",
        "",
        "2. Data filtering preserves retain PPL",
        "   better than SGTM (7.8 vs 8.5)",
        "",
        "3. Linear probes survive ablation:",
        "   Corona. classification barely changes",
        "   (0.82 → 0.83 after ablation)",
        "",
        "4. DMS: holdout outperforms SGTM",
        f"   (ρ = {dms_data.get(f'holdout-{scale}', {}).get('mean_abs_spearman', 0):.3f}"
        f" vs {dms_data.get(f'sgtm-ret25-{scale}', {}).get('mean_abs_spearman', 0):.3f})",
        "",
        "5. Recovery: SGTM-ablated model recovers",
        "   Coronaviridae capability via fine-tuning",
    ]
    text = "\n".join(findings)
    ax_f.text(0.05, 0.95, text, transform=ax_f.transAxes, fontsize=11,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFF0",
                        edgecolor="#CCCCCC"))

    scale_label = scale.upper()
    fig.suptitle(f"SGTM for ESM-2 {scale_label}: Coronaviridae Forget Task — Complete Results",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot family Coronaviridae results")
    parser.add_argument("--results-dir",
                        default="results/sgtm_p2/family_coronaviridae",
                        help="Directory containing result JSONs")
    parser.add_argument("--output-dir", default="figures/family",
                        help="Output directory for plots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--scale", default="35m", choices=["8m", "35m"],
                        help="Model scale to plot")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format
    scale = args.scale

    data = load_results(args.results_dir)
    print(f"Loaded: {list(data.keys())}\n")

    # Individual plots
    plot_ppl_comparison(data, os.path.join(args.output_dir, f"ppl_comparison_{scale}.{ext}"), scale)
    plot_linear_probes(data, os.path.join(args.output_dir, f"linear_probes_{scale}.{ext}"), scale)
    plot_dms_results(data, os.path.join(args.output_dir, f"dms_per_dataset_{scale}.{ext}"), scale)
    plot_dms_summary(data, os.path.join(args.output_dir, f"dms_summary_{scale}.{ext}"), scale)
    plot_recovery(data, os.path.join(args.output_dir, f"recovery_{scale}.{ext}"), scale)
    plot_ablation_ratios(data, os.path.join(args.output_dir, f"ablation_ratios_{scale}.{ext}"), scale)

    # Dashboard
    plot_dashboard(data, os.path.join(args.output_dir, f"dashboard_{scale}.{ext}"), scale)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
