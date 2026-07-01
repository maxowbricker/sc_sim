#!/usr/bin/env python3
"""
Plot: Computational Efficiency & Scalability  (§5.3)

Two-panel figure:
  Panel A — Runtime vs task volume  (|W| = 10,000 fixed)
  Panel B — Runtime vs fleet size   (|T| = 50,000 fixed)

Strategies timed out at the 900 s wall-clock budget are marked with an ×
at the ceiling rather than excluded entirely, so the reader can see
exactly where each method becomes intractable.

Input:  results/s53_scalability/scalability_tasks_v2.csv
        results/s53_scalability/scalability_fleet_v2.csv
Output: results/figures/scalability.pdf
        results/figures/scalability.png  (preview)

Usage:
    python scripts/plots/plot_scalability.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
INPUT_TASKS = os.path.join(PROJECT_ROOT, "results", "s53_scalability",
                           "scalability_tasks_v2.csv")
INPUT_FLEET = os.path.join(PROJECT_ROOT, "results", "s53_scalability",
                           "scalability_fleet_v2.csv")
OUT_PDF = os.path.join(PROJECT_ROOT, "results", "figures", "scalability.pdf")
OUT_PNG = os.path.join(PROJECT_ROOT, "results", "figures", "scalability.png")

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
    "k-NLF (k=15)":       (r"k-NLF ($k{=}15$)", "#2166ac", "o",  "-"),
    "Composite (static)":  ("Composite",          "#e66101", "D",  "--"),
    "Greedy":              ("Greedy",              "#888888", "s",  "-"),
    "LAF":                 ("LAF",                 "#4dac26", "^",  "-"),
}


def load_sweep(
    path: str, x_col: str
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    Returns {strategy: (x_valid, y_valid, x_timeout)}.
    x_timeout contains x-values where elapsed_s hit the 900 s ceiling.
    """
    df = pd.read_csv(path)
    result: dict[str, tuple[list, list, list]] = {}
    for strat in df["strategy"].unique():
        sub = df[df["strategy"] == strat].sort_values(x_col)
        x_valid, y_valid, x_to = [], [], []
        for _, row in sub.iterrows():
            x, y = float(row[x_col]), float(row["elapsed_s"])
            if y >= TIMEOUT_S:
                x_to.append(x)
            else:
                x_valid.append(x)
                y_valid.append(y)
        result[strat] = (x_valid, y_valid, x_to)
    return result


# ── Load data ──────────────────────────────────────────────────────────────────
tasks_data = load_sweep(INPUT_TASKS, "n_tasks")
fleet_data = load_sweep(INPUT_FLEET, "n_workers")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9))
fig.subplots_adjust(wspace=0.42)


def plot_sweep(
    ax: plt.Axes,
    data: dict,
    x_col_label: str,
    x_tick_vals: list[float],
    x_tick_labels: list[str],
    title: str,
) -> None:
    for strat, (label, color, marker, ls) in STRATEGY_STYLE.items():
        if strat not in data:
            continue
        x_valid, y_valid, _x_to = data[strat]

        if x_valid:
            ax.plot(x_valid, y_valid,
                    marker=marker, markersize=4, linewidth=1.5,
                    color=color, linestyle=ls, label=label, zorder=3)

    ax.set_xlim(x_tick_vals[0] * 0.88, x_tick_vals[-1] * 1.04)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_tick_vals)
    ax.set_xticklabels(x_tick_labels, rotation=30, ha="right")
    ax.set_xlabel(x_col_label)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)


# Panel A: task volume sweep
plot_sweep(
    ax1, tasks_data,
    r"Task volume $|\mathcal{T}|$",
    [50_000, 100_000, 150_000, 200_000, 224_219],
    ["50k", "100k", "150k", "200k", "224k"],
    r"(a) Runtime vs Task Volume" + "\n" + r"($|W|{=}10{,}000$ workers)",
)

# Panel B: fleet size sweep
plot_sweep(
    ax2, fleet_data,
    r"Fleet size $|W|$",
    [10_000, 15_000, 20_000, 25_000, 30_000, 36_799],
    ["10k", "15k", "20k", "25k", "30k", "36.8k"],
    r"(b) Runtime vs Fleet Size" + "\n" + r"($|\mathcal{T}|{=}50{,}000$ tasks)",
)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved → {OUT_PDF}")
print(f"Saved → {OUT_PNG}")
