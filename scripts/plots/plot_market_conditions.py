#!/usr/bin/env python3
"""
Plot: Robustness to Supply-Demand Conditions  (§5.3)

Two-panel figure showing how JFI (tasks) responds to market condition shifts:
  Panel A — Fleet size sweep: JFI vs |W|  (|T|=50k fixed)
             → lines diverge: Greedy collapses, k-NLF/Composite hold up, LAF stays high
  Panel B — Task volume sweep: JFI vs |T|  (|W|=10k fixed)
             → lines converge: all strategies equalise under extreme demand pressure

Input:  results/s53_scalability/scalability_fleet_final.csv
        results/s53_scalability/scalability_tasks_final.csv
Output: results/figures/market_conditions.pdf
        results/figures/market_conditions.png  (preview)

Usage:
    python scripts/plots/plot_market_conditions.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
INPUT_FLEET = os.path.join(PROJECT_ROOT, "results", "s53_scalability",
                           "scalability_fleet_final.csv")
INPUT_TASKS = os.path.join(PROJECT_ROOT, "results", "s53_scalability",
                           "scalability_tasks_final.csv")
OUT_PDF = os.path.join(PROJECT_ROOT, "results", "figures", "market_conditions.pdf")
OUT_PNG = os.path.join(PROJECT_ROOT, "results", "figures", "market_conditions.png")

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

TIMEOUT_S = 900.0

# Strategy display config: internal_name -> (label, color, marker, linestyle)
STRATEGY_STYLE: dict[str, tuple[str, str, str, str]] = {
    "k-NLF (k=15)":       (r"k-NLF ($k{=}15$)$^\dagger$", "#2166ac", "o",  "-"),
    "Composite (static)":  (r"Composite$^\dagger$",         "#e66101", "D",  "--"),
    "Greedy":              ("Greedy (baseline)",             "#888888", "s",  "-"),
}

# ── Load data ──────────────────────────────────────────────────────────────────
df_fleet = pd.read_csv(INPUT_FLEET)
df_tasks = pd.read_csv(INPUT_TASKS)

# Drop timed-out rows
df_fleet = df_fleet[df_fleet["elapsed_s"] < TIMEOUT_S].copy()
df_tasks = df_tasks[df_tasks["elapsed_s"] < TIMEOUT_S].copy()

# ── Figure: two data panels + shared left legend ──────────────────────────────
# Shift both panels right (left=0.24) to leave a clear margin for the legend.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 2.8))
fig.subplots_adjust(left=0.24, right=0.97, wspace=0.42, bottom=0.20, top=0.88)


def _plot_jfi(ax: plt.Axes, df: pd.DataFrame, x_col: str,
              x_tick_vals: list, x_tick_labels: list, title: str,
              xlabel: str) -> None:
    for strat, (label, color, marker, ls) in STRATEGY_STYLE.items():
        sub = df[df["strategy"] == strat].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub["JFI (tasks)"],
                marker=marker, markersize=4, linewidth=1.5,
                color=color, linestyle=ls, label=label, zorder=3)

    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels, rotation=30, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("JFI (tasks) $\\uparrow$")
    ax.set_title(title)
    ax.set_ylim(bottom=0.25, top=1.02)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    # no per-panel legend — shared legend goes in ax_leg


# Panel A: fleet size sweep (supply-rich conditions)
fleet_x_vals   = [10_000, 15_000, 20_000, 25_000, 30_000, 36_799]
fleet_x_labels = ["10k", "15k", "20k", "25k", "30k", "36.8k"]
_plot_jfi(
    ax1, df_fleet, "n_workers",
    fleet_x_vals, fleet_x_labels,
    "(a) Increasing Supply\n" + r"($|\mathcal{T}|{=}50{,}000$ fixed)",
    r"Fleet size $|W|$",
)

# Panel B: task volume sweep (increasing demand pressure)
tasks_x_vals   = [50_000, 100_000, 150_000, 200_000, 224_219]
tasks_x_labels = ["50k", "100k", "150k", "200k", "224k"]
_plot_jfi(
    ax2, df_tasks, "n_tasks",
    tasks_x_vals, tasks_x_labels,
    "(b) Increasing Demand Pressure\n" + r"($|W|{=}10{,}000$ fixed)",
    r"Task volume $|\mathcal{T}|$",
)

# ── Shared left legend ─────────────────────────────────────────────────────────
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center left",
    bbox_to_anchor=(0.01, 0.54),
    bbox_transform=fig.transFigure,
    frameon=False,
    fontsize=8,
)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved → {OUT_PDF}")
print(f"Saved → {OUT_PNG}")
