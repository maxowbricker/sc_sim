#!/usr/bin/env python3
"""
Plot: Composite Fairness Weight (λ_f) Sensitivity  (§5.4.3)

Two-panel figure:
  Panel A — JFI (tasks) [left y] and JFI rate [right y] vs λ_f  (worker fairness)
  Panel B — P95 task wait time vs λ_f  (double benefit: task tail latency drops)

TAR is omitted from the figure (range ±0.0002 across the full sweep) and
reported instead as a single sentence in the prose.

Input:  results/s54_ablation/fairness_weight_sweep_final.csv
Output: results/figures/fairness_weight_sweep.pdf
        results/figures/fairness_weight_sweep.png

Usage:
    python scripts/plots/plot_fairness_weight.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
INPUT   = os.path.join(PROJECT_ROOT, "results", "s54_ablation",
                       "fairness_weight_sweep_final.csv")
OUT_PDF = os.path.join(PROJECT_ROOT, "results", "figures", "fairness_weight_sweep.pdf")
OUT_PNG = os.path.join(PROJECT_ROOT, "results", "figures", "fairness_weight_sweep.png")

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

PAPER_LF     = 1.6
COLOR_RATE   = "#4dac26"   # green  (Composite JFI rate)
COLOR_P95    = "#d6604d"   # red-orange  (Composite P95)
COLOR_GREEDY = "#808080"   # neutral grey
COLOR_KNLF   = "#4B0082"   # dark indigo
COLOR_GUIDE  = "#aaaaaa"

# Reference values from knlf_k_sweep_20161109_v2.csv (k=15 rows, global-scan Greedy anchor)
GREEDY_JFI_RATE = 0.8634
GREEDY_P95      = 10.04
KNLF_JFI_RATE   = 0.8897
KNLF_P95        = 14.08

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT).sort_values("fairness_weight")

lf       = df["fairness_weight"].tolist()
jfi      = df["JFI (tasks)"].tolist()
jfi_rate = df["JFI rate"].tolist()
p95      = df["P95 Wait (m)"].tolist()

# ── Figure: two data panels + shared left legend ──────────────────────────────
# Shift both panels right (left=0.26) to leave clear margin for the 4-entry legend.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0))
fig.subplots_adjust(left=0.26, right=0.97, wspace=0.45, bottom=0.15, top=0.88)

def _add_guide(ax):
    ax.axvline(PAPER_LF, linestyle="--", linewidth=0.8, color=COLOR_GUIDE, zorder=1)
    ymin, ymax = ax.get_ylim()
    y_label = ymin + 0.25 * (ymax - ymin)
    ax.text(PAPER_LF + 0.08, y_label,
            f"$\\lambda_f$={PAPER_LF}", fontsize=7, color="#777777", va="bottom")

# ── Panel A: JFI rate only (hero metric) ─────────────────────────────────────
h_greedy_a, = ax1.plot([], [], linestyle="--", linewidth=1.5, color=COLOR_GREEDY, label="Greedy")
h_knlf_a,   = ax1.plot([], [], linestyle="-.", linewidth=1.5, color=COLOR_KNLF,   label="k-NLF")

ax1.axhline(GREEDY_JFI_RATE, linestyle="--", linewidth=1.5, color=COLOR_GREEDY, zorder=1)
ax1.axhline(KNLF_JFI_RATE,   linestyle="-.", linewidth=1.5, color=COLOR_KNLF,   zorder=1)

h_comp_a, = ax1.plot(lf, jfi_rate, marker="s", markersize=3.5, linewidth=2.5,
                     color=COLOR_RATE, label="Composite", zorder=2)

_add_guide(ax1)

ax1.set_xlabel("Fairness weight $\\lambda_f$")
ax1.set_ylabel("JFI rate")
ax1.set_title("(a) Worker Fairness")

# ── Panel B: P95 task wait ────────────────────────────────────────────────────
h_greedy_b, = ax2.plot([], [], linestyle="--", linewidth=1.5, color=COLOR_GREEDY, label="Greedy")
h_knlf_b,   = ax2.plot([], [], linestyle="-.", linewidth=1.5, color=COLOR_KNLF,   label="k-NLF")

ax2.axhline(GREEDY_P95, linestyle="--", linewidth=1.5, color=COLOR_GREEDY, zorder=1)
ax2.axhline(KNLF_P95,   linestyle="-.", linewidth=1.5, color=COLOR_KNLF,   zorder=1)

h_comp_b, = ax2.plot(lf, p95, marker="s", markersize=3.5, linewidth=2.5,
                     color=COLOR_P95, label="Composite", zorder=2)

ax2.set_ylim(GREEDY_P95 - 0.5, max(p95) + 0.5)
_add_guide(ax2)

ax2.set_xlabel("Fairness weight $\\lambda_f$")
ax2.set_ylabel("P95 task wait time (min)")
ax2.set_title("(b) Task Tail Latency")

# ── Shared left legend ────────────────────────────────────────────────────────
fig.legend(
    handles=[h_greedy_a, h_knlf_a, h_comp_a, h_comp_b],
    labels=[
        "Greedy baseline",
        r"k-NLF ($k{=}15$)",
        r"Composite (JFI $\uparrow$)",
        r"Composite (P95 $\downarrow$)",
    ],
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
