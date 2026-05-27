#!/usr/bin/env python3
"""
Sec 5.4 — Architecture Ablation: Action-Space Symmetry Fix

Produces a grouped bar chart that visually proves the symmetric action space
[−1, 1] → [0, 2] broke the boundary-collapse problem.

Data source: measured max and mean λ₁ values from run logs (no new runs needed).
All values come from weight_outputs.txt files / SPRINT_TRIALS_DETAILED_ANALYSIS.md.

Usage (from project root):
    python scripts/plot_ablation.py

Outputs:
    figures/ablation_action_space.pdf
    figures/ablation_action_space.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Data: (label, max_lambda1, mean_lambda1) ──────────────────────────────────
#
# Ablation B  — Asymmetric [0, 2] + ΔJ FI  (run_20260501_135623, 300k steps)
#   boundary-sticking artefact: agent parks at 0 almost always
#   max ever recorded: 0.357   mean: 0.133
#
# Trial D     — Symmetric [-1,1] + ΔJ FI   (run_20260513_042601, 25k steps sprint)
#   first run with the fix: max 1.199, mean 0.891
#
# Champion    — Symmetric [-1,1] + ΔJ FI   (run_20260513_071355, 300k steps)
#   our final submitted model: max 0.671, mean 0.471   (conservative but stable)
#   Note: max is lower than Trial D because the 300k run converges to
#   a more cautious policy after seeing far more data.

CONFIGS = [
    {
        "label":       "Ablation B\n(Asymmetric [0,2]\n+ ΔJFI, 300k)",
        "max_lambda1": 0.357,
        "mean_lambda1":0.133,
        "color":       "#e74c3c",   # red  — broken
        "hatch":       "//",
    },
    {
        "label":       "Trial D\n(Symmetric [−1,1]\n+ ΔJFI, 25k sprint)",
        "max_lambda1": 1.199,
        "mean_lambda1":0.891,
        "color":       "#f39c12",   # amber — fixed, short
        "hatch":       "..",
    },
    {
        "label":       "Champion\n(Symmetric [−1,1]\n+ ΔJFI, 300k final)",
        "max_lambda1": 0.671,
        "mean_lambda1":0.471,
        "color":       "#27ae60",   # green — proposed, fully trained
        "hatch":       "",
    },
]

labels      = [c["label"]        for c in CONFIGS]
max_vals    = [c["max_lambda1"]  for c in CONFIGS]
mean_vals   = [c["mean_lambda1"] for c in CONFIGS]
colors      = [c["color"]        for c in CONFIGS]
hatches     = [c["hatch"]        for c in CONFIGS]

x      = np.arange(len(labels))
width  = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_max  = ax.bar(x - width / 2, max_vals,  width, label=r"Max $\lambda_1$ explored",
                   color=colors, hatch=[h for h in hatches], edgecolor="black",
                   linewidth=0.8)
bars_mean = ax.bar(x + width / 2, mean_vals, width, label=r"Mean $\lambda_1$",
                   color=colors, alpha=0.55, hatch=[h for h in hatches],
                   edgecolor="black", linewidth=0.8)

# Value labels on top of each bar
for bar in bars_max:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
for bar in bars_mean:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            fontsize=10)

# Reference line at λ₁ = 1.0 (Optuna-optimal static weight)
ax.axhline(1.0, color="steelblue", linestyle="--", linewidth=1.4,
           label=r"Optuna-optimal static $\lambda_1 = 1.0$")

# Boundary collapse annotation
ax.annotate("Boundary\nCollapse",
            xy=(0, max_vals[0] + 0.02), xytext=(0.25, 0.55),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
            fontsize=9, color="#c0392b", ha="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r"$\lambda_1$ (Fairness Weight) Value", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.45)
ax.set_title(
    r"Ablation Study: Impact of Symmetric Action Space on $\lambda_1$ Exploration",
    fontsize=13, pad=14,
)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.45)

fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(FIGURES_DIR, f"ablation_action_space.{ext}")
    plt.savefig(out, format=ext, dpi=300, bbox_inches="tight")
    print(f"✅  Saved {out}")

plt.close()
