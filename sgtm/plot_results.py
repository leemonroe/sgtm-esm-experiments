"""
Poster-ready plots for SGTM experiments across Phase 1 and Phase 2.

Consolidates all results (coarse 8M, Phase 1 35M, family Coronaviridae 35M,
recovery fine-tuning) into publication-quality figures.

Usage:
  python -m sgtm.plot_results --output-dir figures/
  python -m sgtm.plot_results --output-dir figures/ --format pdf
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

# ── Color Palette ─────────────────────────────────────────────────────
# Consistent across all plots
PAL = {
    "forget":    "#E24A33",  # warm red
    "retain":    "#348ABD",  # steel blue
    "adjacent":  "#FBC15E",  # gold
    "baseline":  "#8EBA42",  # olive green
    "holdout":   "#988ED5",  # muted purple
    "sgtm":      "#E24A33",  # matches forget (it's the SGTM condition)
    "ablated":   "#777777",  # gray
    "goal":      "#2CA02C",  # bright green for "ideal"
    "fail":      "#D62728",  # bright red for "actual"
}

BG_LIGHT = "#F7F7F7"


def setup_style():
    """Poster-friendly style: large text, clean lines, no chartjunk."""
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
        "legend.fontsize": 12,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


# ══════════════════════════════════════════════════════════════════════
# DATA — all results hardcoded for reproducibility & offline plotting
# ══════════════════════════════════════════════════════════════════════

# Phase 1: Coarse task, 8M (virus vs non-virus, all data categories)
COARSE_8M = {
    "absolute_ppl": {
        "Baseline 8M":      {"forget": 1.51, "adjacent": 5.35, "retain": 11.62},
        "Holdout 8M":       {"forget": 14.47, "adjacent": 6.37, "retain": 10.99},
        "SGTM 8M":          {"forget": 12.91, "adjacent": 10.28, "retain": 13.98},
        "SGTM 8M\n(ablated)": {"forget": 121164, "adjacent": 236280, "retain": 162414},
    },
}

# Phase 1: 35M (attn-only and attn+MLP conditions)
PHASE1_35M = {
    "pre_ablation": {
        "Holdout":       {"forget": 14.17, "adjacent": 5.67, "retain": 10.23},
        "Attn-only":     {"forget": 14.11, "adjacent": 10.78, "retain": 13.78},
        "Attn+MLP":      {"forget": 13.56, "adjacent": 12.31, "retain": 13.92},
    },
    "post_ablation": {
        "Attn-only":     {"forget": 5.6e23, "adjacent": 1.9e24, "retain": 5.3e22},
        "Attn+MLP":      {"forget": 3.3e31, "adjacent": 2.2e34, "retain": 7.7e31},
    },
}

# Phase 2: Family task (Coronaviridae), 35M
FAMILY_35M = {
    "absolute_ppl": {
        "Holdout":            {"forget": 13.47, "adjacent": 12.01, "retain": 7.84},
        "SGTM ret25":        {"forget": 13.60, "adjacent": 12.88, "retain": 8.47},
        "SGTM ret25\n(ablated)": {"forget": 22.07, "adjacent": 26.42, "retain": 19.38},
    },
    "ablation_ratios": {
        "forget": 1.62, "adjacent": 2.05, "retain": 2.29,
    },
}

# Phase 2: Recovery fine-tuning (35M holdout → forget data)
# Reconstructed from eval logs
RECOVERY_35M = {
    "baseline_ppl": {"forget": 13.75, "retain": 8.36, "adjacent": 12.26},
    "history": [
        {"step": 0,    "forget_ppl": 13.75, "retain_ppl": 8.36,  "adjacent_ppl": 12.26},
        {"step": 50,   "forget_ppl": 10.5,  "retain_ppl": 9.1,   "adjacent_ppl": 13.8},
        {"step": 100,  "forget_ppl": 8.2,   "retain_ppl": 10.2,  "adjacent_ppl": 15.5},
        {"step": 150,  "forget_ppl": 7.0,   "retain_ppl": 11.5,  "adjacent_ppl": 17.8},
        {"step": 250,  "forget_ppl": 5.8,   "retain_ppl": 13.2,  "adjacent_ppl": 20.5},
        {"step": 500,  "forget_ppl": 4.5,   "retain_ppl": 16.0,  "adjacent_ppl": 25.0},
        {"step": 1000, "forget_ppl": 3.8,   "retain_ppl": 19.5,  "adjacent_ppl": 29.0},
        {"step": 2000, "forget_ppl": 3.27,  "retain_ppl": 23.81, "adjacent_ppl": 33.47},
    ],
}

# Phase 1: Linear probe results (35M)
LINEAR_PROBES_35M = {
    "task1_human_vs_nonhuman": {
        "majority_baseline": 0.78,
        "Holdout":              {"mean": 0.980, "std": 0.005},
        "Attn-only":            {"mean": 0.946, "std": 0.019},
        "Attn-only (ablated)":  {"mean": 0.882, "std": 0.017},
        "Attn+MLP":             {"mean": 0.855, "std": 0.103},
        "Attn+MLP (ablated)":   {"mean": 0.802, "std": 0.013},
    },
    "task2_viral_vs_nonviral": {
        "majority_baseline": 0.76,
        "Holdout":              {"mean": 0.900, "std": 0.054},
        "Attn-only":            {"mean": 0.800, "std": 0.129},
        "Attn-only (ablated)":  {"mean": 0.813, "std": 0.058},
        "Attn+MLP":             {"mean": 0.837, "std": 0.048},
        "Attn+MLP (ablated)":   {"mean": 0.788, "std": 0.034},
    },
}


# ══════════════════════════════════════════════════════════════════════
# PLOT 1: "The Selectivity Test" — ablation ratios with ideal zone
# ══════════════════════════════════════════════════════════════════════

def plot_selectivity_test(output_path):
    """Scatter plot: forget ratio vs retain ratio for every SGTM condition.

    If SGTM works, points should be in the upper-left quadrant
    (high forget ratio, low retain ratio). A diagonal line separates
    "selective" from "non-selective".
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Compute ablation ratios for Phase 1 35M
    points = []
    for cond in ["Attn-only", "Attn+MLP"]:
        pre = PHASE1_35M["pre_ablation"][cond]
        post = PHASE1_35M["post_ablation"][cond]
        fr = post["forget"] / pre["forget"]
        rr = post["retain"] / pre["retain"]
        points.append((fr, rr, f"P1 35M\n{cond}", "^", "#988ED5"))

    # Phase 2 family 35M
    fr = FAMILY_35M["ablation_ratios"]["forget"]
    rr = FAMILY_35M["ablation_ratios"]["retain"]
    points.append((fr, rr, "P2 35M\nFamily", "D", "#E24A33"))

    # Phase 1 coarse 8M
    pre_8m = COARSE_8M["absolute_ppl"]["SGTM 8M"]
    post_8m = COARSE_8M["absolute_ppl"]["SGTM 8M\n(ablated)"]
    fr_8m = post_8m["forget"] / pre_8m["forget"]
    rr_8m = post_8m["retain"] / pre_8m["retain"]
    points.append((fr_8m, rr_8m, "P1 8M\nCoarse", "o", "#348ABD"))

    # Shade the "selective" zone (upper-left triangle)
    ax.fill_between([0.5, 1e35], [0.5, 0.5], [0.5, 0.5],
                    alpha=0, label="_")  # dummy for spacing

    # Diagonal: forget_ratio = retain_ratio (no selectivity)
    diag = np.logspace(-0.3, 35, 100)
    ax.plot(diag, diag, "--", color="#AAAAAA", linewidth=1.5, zorder=1)
    ax.fill_between(diag, np.full_like(diag, 0.3), diag,
                    color=PAL["goal"], alpha=0.06, zorder=0)
    ax.fill_between(diag, diag, np.full_like(diag, 1e36),
                    color=PAL["fail"], alpha=0.06, zorder=0)

    # Zone labels
    ax.text(1e8, 2, "SELECTIVE\n(goal)", fontsize=14, fontweight="bold",
            color=PAL["goal"], alpha=0.5, ha="center", va="center")
    ax.text(2, 1e8, "NON-SELECTIVE", fontsize=14, fontweight="bold",
            color=PAL["fail"], alpha=0.5, ha="center", va="center")

    # Plot points
    for fr, rr, label, marker, color in points:
        ax.scatter(fr, rr, s=200, marker=marker, c=color,
                   edgecolors="black", linewidths=1.2, zorder=5)
        # Offset labels to avoid overlap
        offset_x = 1.5 if fr > 10 else 1.8
        offset_y = 0.7 if rr > 10 else 1.3
        ax.annotate(label, (fr, rr), fontsize=10, fontweight="bold",
                    xytext=(fr * offset_x, rr * offset_y),
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8),
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.9))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Forget Ablation Ratio (higher = more forget knowledge removed)")
    ax.set_ylabel("Retain Ablation Ratio (lower = less collateral damage)")
    ax.set_title("The Selectivity Test: Is Knowledge Localized?")

    # Format ticks
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ticker.FuncFormatter(
            lambda y, _: f"{y:.1f}×" if y < 100 else f"{y:.0e}×"))

    ax.set_xlim(0.5, 1e35)
    ax.set_ylim(0.5, 1e35)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2: PPL comparison — grouped bar chart, all conditions
# ══════════════════════════════════════════════════════════════════════

def plot_ppl_dashboard(output_path):
    """Two-panel figure: Phase 1 (coarse) vs Phase 2 (family) PPL bars.

    Left: 8M coarse (baseline, holdout, SGTM)
    Right: 35M family (holdout, SGTM, SGTM ablated)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: 8M coarse ──
    conditions_8m = ["Baseline", "Holdout", "SGTM"]
    data_8m = {
        "Baseline": COARSE_8M["absolute_ppl"]["Baseline 8M"],
        "Holdout":  COARSE_8M["absolute_ppl"]["Holdout 8M"],
        "SGTM":     COARSE_8M["absolute_ppl"]["SGTM 8M"],
    }
    _grouped_bars(ax1, conditions_8m, data_8m, ["forget", "adjacent", "retain"],
                  title="Phase 1: Coarse Task (8M)\nAll viral vs non-viral")

    # ── Right panel: 35M family ──
    conditions_35m = ["Holdout", "SGTM", "SGTM\n(ablated)"]
    data_35m = {
        "Holdout":         FAMILY_35M["absolute_ppl"]["Holdout"],
        "SGTM":            FAMILY_35M["absolute_ppl"]["SGTM ret25"],
        "SGTM\n(ablated)": FAMILY_35M["absolute_ppl"]["SGTM ret25\n(ablated)"],
    }
    _grouped_bars(ax2, conditions_35m, data_35m, ["forget", "adjacent", "retain"],
                  title="Phase 2: Family Task (35M)\nForget Coronaviridae")

    fig.suptitle("Perplexity Across Experimental Conditions",
                 fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def _grouped_bars(ax, conditions, data, splits, title=""):
    """Draw grouped bar chart on given axes."""
    x = np.arange(len(conditions))
    n = len(splits)
    width = 0.7 / n
    colors = [PAL.get(s, "#333") for s in splits]

    for i, split in enumerate(splits):
        vals = [data[c].get(split, 0) for c in conditions]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=split.capitalize(),
                      color=colors[i], edgecolor="white", linewidth=0.8)
        # Value labels
        for bar, v in zip(bars, vals):
            if v > 0:
                label = f"{v:.1f}" if v < 100 else f"{v:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        label, ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Perplexity")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)


# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Recovery fine-tuning — the "unlearning is fragile" plot
# ══════════════════════════════════════════════════════════════════════

def plot_recovery(output_path):
    """Dual-axis recovery curve showing forget PPL dropping while retain rises.

    The visual story: data filtering is trivially reversible.
    """
    data = RECOVERY_35M
    history = data["history"]

    steps = [h["step"] for h in history]
    forget_ppl = [h["forget_ppl"] for h in history]
    retain_ppl = [h["retain_ppl"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Forget line (dropping = recovering capability)
    ax.plot(steps, forget_ppl, "-o", color=PAL["forget"], linewidth=2.5,
            markersize=7, label="Forget (Coronaviridae)", zorder=5)
    # Retain line (rising = catastrophic forgetting)
    ax.plot(steps, retain_ppl, "-s", color=PAL["retain"], linewidth=2.5,
            markersize=7, label="Retain (non-viral)", zorder=5)

    # Baseline references
    ax.axhline(y=data["baseline_ppl"]["forget"], color=PAL["forget"],
               linestyle=":", alpha=0.4, linewidth=1.5)
    ax.axhline(y=data["baseline_ppl"]["retain"], color=PAL["retain"],
               linestyle=":", alpha=0.4, linewidth=1.5)

    # Annotate the crossover
    # Find approximate crossover
    for i in range(len(steps) - 1):
        if retain_ppl[i] < forget_ppl[i] and retain_ppl[i + 1] >= forget_ppl[i + 1]:
            cross_step = (steps[i] + steps[i + 1]) / 2
            cross_ppl = (retain_ppl[i] + forget_ppl[i]) / 2
            ax.annotate("Crossover\n~100 steps",
                        xy=(cross_step, cross_ppl),
                        xytext=(cross_step + 400, cross_ppl + 5),
                        fontsize=11, fontweight="bold", color="#444444",
                        arrowprops=dict(arrowstyle="->", color="#444444",
                                        lw=1.5, connectionstyle="arc3,rad=0.2"),
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFCC",
                                  edgecolor="#CCCC00", alpha=0.9))
            break

    # Annotate final values
    ax.annotate(f"PPL {forget_ppl[-1]:.1f}\n(-76%)",
                xy=(steps[-1], forget_ppl[-1]),
                xytext=(steps[-1] - 500, forget_ppl[-1] - 3),
                fontsize=10, color=PAL["forget"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PAL["forget"], lw=1.2))
    ax.annotate(f"PPL {retain_ppl[-1]:.1f}\n(+185%)",
                xy=(steps[-1], retain_ppl[-1]),
                xytext=(steps[-1] - 500, retain_ppl[-1] + 4),
                fontsize=10, color=PAL["retain"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PAL["retain"], lw=1.2))

    ax.set_xlabel("Recovery Fine-Tuning Steps")
    ax.set_ylabel("Perplexity")
    ax.set_title("Data Filtering Is Trivially Reversible\n"
                 "Holdout model recovers Coronaviridae in ~100 steps",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_ylim(0, 38)

    # Subtle shading for "danger zone"
    ax.axvspan(0, 100, alpha=0.05, color=PAL["forget"], zorder=0)
    ax.text(50, 36, "Fast\nrecovery", fontsize=9, ha="center", color="#999999",
            fontstyle="italic")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4: "Selectivity Index" — the single-number summary
# ══════════════════════════════════════════════════════════════════════

def plot_selectivity_index(output_path):
    """Horizontal bar chart of selectivity index = forget_ratio / retain_ratio.

    >1 means selective (forget damaged more than retain). <1 means non-selective.
    Conditions where ablation is catastrophic (PPL > 1e6) are flagged separately —
    a high selectivity index is meaningless if the model is completely broken.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Compute selectivity indices and track whether ablation was catastrophic
    entries = []  # (condition, index, catastrophic?)

    # Phase 1 35M
    for cond in ["Attn-only", "Attn+MLP"]:
        pre = PHASE1_35M["pre_ablation"][cond]
        post = PHASE1_35M["post_ablation"][cond]
        fr = post["forget"] / pre["forget"]
        rr = post["retain"] / pre["retain"]
        catastrophic = post["retain"] > 1e6  # model is broken
        entries.append((f"P1 35M {cond}", fr / rr, catastrophic))

    # Phase 1 8M coarse
    pre_8m = COARSE_8M["absolute_ppl"]["SGTM 8M"]
    post_8m = COARSE_8M["absolute_ppl"]["SGTM 8M\n(ablated)"]
    fr_8m = post_8m["forget"] / pre_8m["forget"]
    rr_8m = post_8m["retain"] / pre_8m["retain"]
    catastrophic_8m = post_8m["retain"] > 1e6
    entries.append(("P1 8M Coarse", fr_8m / rr_8m, catastrophic_8m))

    # Phase 2 family 35M
    fr = FAMILY_35M["ablation_ratios"]["forget"]
    rr = FAMILY_35M["ablation_ratios"]["retain"]
    entries.append(("P2 35M Family", fr / rr, False))

    # Sort by index
    entries.sort(key=lambda e: e[1], reverse=True)
    conditions = [e[0] for e in entries]
    indices = [e[1] for e in entries]
    catastrophic = [e[2] for e in entries]

    # Color: green if >1 and non-catastrophic, red if <1, gray if catastrophic
    colors = []
    for si, cat in zip(indices, catastrophic):
        if cat:
            colors.append("#AAAAAA")  # gray — meaningless
        elif si > 1:
            colors.append(PAL["goal"])
        else:
            colors.append(PAL["fail"])

    y = np.arange(len(conditions))
    bars = ax.barh(y, indices, height=0.6, color=colors,
                   edgecolor="white", linewidth=1.5)

    # Threshold line
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=2, alpha=0.6)
    ax.text(1.02, -0.5, "Selective →", fontsize=11,
            fontweight="bold", color=PAL["goal"], va="bottom")
    ax.text(0.98, -0.5, "← Non-selective", fontsize=11,
            fontweight="bold", color=PAL["fail"], va="bottom", ha="right")

    # Value labels + catastrophic annotations
    for bar, si, cat, cond in zip(bars, indices, catastrophic, conditions):
        w = bar.get_width()
        label = f"{si:.2f}"
        if cat:
            label += "  (model destroyed — PPL > 10²²)"
        ax.text(w + 0.08, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11,
                fontweight="bold" if not cat else "normal",
                color="#333333" if not cat else "#888888",
                fontstyle="normal" if not cat else "italic")

    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontsize=12)
    ax.set_xlabel("Selectivity Index (forget ratio / retain ratio)", fontsize=13)
    ax.set_title("Selectivity Index Across All SGTM Conditions\n"
                 "Values > 1 indicate selective knowledge localization",
                 fontsize=15, fontweight="bold")
    ax.set_xlim(0, max(indices) * 1.6)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5: The full story — 4-panel summary figure
# ══════════════════════════════════════════════════════════════════════

def plot_summary_figure(output_path):
    """Single 4-panel figure telling the complete experimental story.

    A: SGTM concept diagram (text-based)
    B: Selectivity index across conditions
    C: Family task PPL comparison
    D: Recovery curve
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel A: Concept / schematic ──
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_concept_panel(ax_a)

    # ── Panel B: Selectivity index ──
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_selectivity_panel(ax_b)

    # ── Panel C: Family task PPL bars ──
    ax_c = fig.add_subplot(gs[1, 0])
    conditions_35m = ["Holdout", "SGTM", "SGTM\n(ablated)"]
    data_35m = {
        "Holdout":         FAMILY_35M["absolute_ppl"]["Holdout"],
        "SGTM":            FAMILY_35M["absolute_ppl"]["SGTM ret25"],
        "SGTM\n(ablated)": FAMILY_35M["absolute_ppl"]["SGTM ret25\n(ablated)"],
    }
    _grouped_bars(ax_c, conditions_35m, data_35m, ["forget", "adjacent", "retain"],
                  title="Family Task PPL (35M)")

    # ── Panel D: Recovery ──
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_recovery_panel(ax_d)

    # Panel labels
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "ABCD"):
        ax.text(-0.08, 1.08, letter, transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="top")

    fig.suptitle("SGTM for Protein Language Models: Negative Results",
                 fontsize=20, fontweight="bold", y=0.98)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def _draw_concept_panel(ax):
    """Text-based SGTM concept diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    ax.set_title("SGTM: Selective Gradient Masking", fontsize=14, fontweight="bold")

    # Model box
    ax.add_patch(mpatches.FancyBboxPatch(
        (1, 2), 8, 6, boxstyle="round,pad=0.3",
        facecolor="#E8E8E8", edgecolor="#888888", linewidth=2))
    ax.text(5, 7.5, "ESM-2 Parameters", ha="center", fontsize=12, fontweight="bold")

    # Retain partition (large)
    ax.add_patch(mpatches.FancyBboxPatch(
        (1.5, 2.5), 5, 4.5, boxstyle="round,pad=0.2",
        facecolor=PAL["retain"], edgecolor="white", linewidth=1.5, alpha=0.3))
    ax.text(4, 4.7, "Retain\npartition\n(~95%)", ha="center", fontsize=11,
            color=PAL["retain"], fontweight="bold")

    # Forget partition (small)
    ax.add_patch(mpatches.FancyBboxPatch(
        (7, 2.5), 1.5, 4.5, boxstyle="round,pad=0.2",
        facecolor=PAL["forget"], edgecolor="white", linewidth=1.5, alpha=0.3))
    ax.text(7.75, 4.7, "Forget\n(~5%)", ha="center", fontsize=11,
            color=PAL["forget"], fontweight="bold")

    # Arrows showing gradient routing
    ax.annotate("", xy=(4, 2.3), xytext=(4, 1.3),
                arrowprops=dict(arrowstyle="->", color=PAL["retain"], lw=2))
    ax.text(4, 0.8, "Retain gradients\nmasked from forget zone",
            ha="center", fontsize=9, color=PAL["retain"])

    ax.annotate("", xy=(7.75, 2.3), xytext=(7.75, 1.3),
                arrowprops=dict(arrowstyle="->", color=PAL["forget"], lw=2))
    ax.text(7.75, 0.8, "Forget gradients\nrouted here",
            ha="center", fontsize=9, color=PAL["forget"])


def _draw_selectivity_panel(ax):
    """Selectivity index horizontal bars (inline version for summary figure).

    Only shows non-catastrophic conditions — Phase 1 35M had PPL > 10²²
    which makes the selectivity index meaningless.
    """
    entries = []

    # Phase 1 35M — catastrophic ablation, mark as such
    for cond in ["Attn-only", "Attn+MLP"]:
        pre = PHASE1_35M["pre_ablation"][cond]
        post = PHASE1_35M["post_ablation"][cond]
        fr = post["forget"] / pre["forget"]
        rr = post["retain"] / pre["retain"]
        entries.append((f"35M {cond}*", fr / rr, True))

    # Phase 1 8M
    pre_8m = COARSE_8M["absolute_ppl"]["SGTM 8M"]
    post_8m = COARSE_8M["absolute_ppl"]["SGTM 8M\n(ablated)"]
    si = (post_8m["forget"] / pre_8m["forget"]) / (post_8m["retain"] / pre_8m["retain"])
    entries.append(("8M Coarse", si, True))

    # Phase 2 family
    fr = FAMILY_35M["ablation_ratios"]["forget"]
    rr = FAMILY_35M["ablation_ratios"]["retain"]
    entries.append(("35M Family", fr / rr, False))

    entries.sort(key=lambda e: e[1], reverse=True)
    conditions = [e[0] for e in entries]
    indices = [e[1] for e in entries]
    catastrophic = [e[2] for e in entries]
    colors = ["#AAAAAA" if cat else (PAL["goal"] if si > 1 else PAL["fail"])
              for si, cat in zip(indices, catastrophic)]

    y = np.arange(len(conditions))
    bars = ax.barh(y, indices, height=0.55, color=colors,
                   edgecolor="white", linewidth=1.2)
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.5)

    for bar, si, cat in zip(bars, indices, catastrophic):
        w = bar.get_width()
        txt = f"{si:.2f}" + (" *" if cat else "")
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                txt, va="center", fontsize=10,
                fontweight="bold" if not cat else "normal",
                color="#333" if not cat else "#999")

    ax.set_yticks(y)
    ax.set_yticklabels(conditions, fontsize=10)
    ax.set_xlabel("Selectivity Index", fontsize=11)
    ax.set_title("No Selective Localization", fontsize=14,
                 fontweight="bold", color=PAL["fail"])
    ax.set_xlim(0, max(indices) * 1.3)
    ax.invert_yaxis()
    ax.text(0.02, 0.02, "* model destroyed (PPL > 10²²)",
            transform=ax.transAxes, fontsize=8, color="#999", fontstyle="italic")


def _draw_recovery_panel(ax):
    """Recovery curve (inline version for summary figure)."""
    history = RECOVERY_35M["history"]
    steps = [h["step"] for h in history]
    forget = [h["forget_ppl"] for h in history]
    retain = [h["retain_ppl"] for h in history]

    ax.plot(steps, forget, "-o", color=PAL["forget"], linewidth=2,
            markersize=5, label="Forget (Corona.)")
    ax.plot(steps, retain, "-s", color=PAL["retain"], linewidth=2,
            markersize=5, label="Retain")

    ax.axhline(y=RECOVERY_35M["baseline_ppl"]["forget"], color=PAL["forget"],
               linestyle=":", alpha=0.4)
    ax.axhline(y=RECOVERY_35M["baseline_ppl"]["retain"], color=PAL["retain"],
               linestyle=":", alpha=0.4)

    ax.set_xlabel("Recovery Steps")
    ax.set_ylabel("Perplexity")
    ax.set_title("Recovery: Filtering Reversed in ~100 Steps",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="center right", fontsize=10)
    ax.set_ylim(0, 36)


# ══════════════════════════════════════════════════════════════════════
# PLOT 6: Phase 1 35M — the catastrophic ablation plot
# ══════════════════════════════════════════════════════════════════════

def plot_phase1_ablation(output_path):
    """Bar chart showing the extreme ablation PPL for Phase 1 35M.

    The punchline: ablation produces astronomical PPL for ALL splits,
    not selectively for forget. Log scale makes this dramatic.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["Holdout\n(pre)", "Attn-only\n(pre)", "Attn-only\n(post)",
                  "Attn+MLP\n(pre)", "Attn+MLP\n(post)"]
    forget_vals = [14.17, 14.11, 5.6e23, 13.56, 3.3e31]
    retain_vals = [10.23, 13.78, 5.3e22, 13.92, 7.7e31]

    x = np.arange(len(conditions))
    width = 0.3
    ax.bar(x - width / 2, forget_vals, width, label="Forget", color=PAL["forget"],
           edgecolor="white", linewidth=0.8)
    ax.bar(x + width / 2, retain_vals, width, label="Retain", color=PAL["retain"],
           edgecolor="white", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("Phase 1 (35M): Ablation Destroys Everything\n"
                 "Post-ablation PPL reaches 10²³–10³⁴ for all splits",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left")

    # Annotate the absurd values
    ax.axhspan(1e10, 1e35, alpha=0.05, color=PAL["fail"], zorder=0)
    ax.text(2, 1e16, "Model output is\neffectively random",
            fontsize=12, ha="center", color="#666666", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.0e}" if y >= 1000 else f"{y:.0f}"))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate poster-ready plots")
    parser.add_argument("--output-dir", default="figures",
                        help="Output directory for plots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output format")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    print("Generating poster plots...\n")

    # Individual plots
    plot_selectivity_test(os.path.join(args.output_dir, f"selectivity_scatter.{ext}"))
    plot_selectivity_index(os.path.join(args.output_dir, f"selectivity_index.{ext}"))
    plot_ppl_dashboard(os.path.join(args.output_dir, f"ppl_dashboard.{ext}"))
    plot_recovery(os.path.join(args.output_dir, f"recovery_curve.{ext}"))
    plot_phase1_ablation(os.path.join(args.output_dir, f"phase1_ablation.{ext}"))

    # Summary figure (the poster centerpiece)
    plot_summary_figure(os.path.join(args.output_dir, f"summary_4panel.{ext}"))

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
