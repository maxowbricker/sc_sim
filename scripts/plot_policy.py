#!/usr/bin/env python3
"""
Sec 5.3 — Policy Visualization: Weight Trajectory Plot

Reads baseline_best_model_weight_outputs.txt from the best 300k run and produces
a dual-axis plot showing how the Governor adapts λ₁ and λ₂ across the 8-hour episode.
Includes actual time-of-day context (e.g., "6:00 AM - 2:00 PM").

Usage (from project root):
    python scripts/plot_policy.py

Outputs:
    figures/policy_visualization.pdf   — LaTeX-ready vector figure
    figures/policy_visualization.png   — Quick-preview PNG
"""

import os
import re
import sys
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── path to the weight-outputs file ──────────────────────────────────────────
WEIGHTS_TXT = os.path.join(
    PROJECT_ROOT,
    "rl_logs_sb3", "run_20260513_071355",
    "baseline_best_model_weight_outputs.txt",
)

# ── parse ─────────────────────────────────────────────────────────────────────
STEP_RE = re.compile(
    r"Step\s+\d+:\s+λ1=([\d.]+),\s+λ2=([\d.]+),\s+λ3=[\d.]+\s+\(reward=([-\d.]+),"
    r"\s+sim_time=([\d.]+)"
)

steps, lambda1, lambda2, rewards, sim_times = [], [], [], [], []

with open(WEIGHTS_TXT, "r", encoding="utf-8") as fh:
    for line in fh:
        m = STEP_RE.search(line)
        if m:
            idx = len(steps)
            steps.append(idx * 5 / 60)        # 5-min steps → hours
            lambda1.append(float(m.group(1)))
            lambda2.append(float(m.group(2)))
            rewards.append(float(m.group(3)))
            sim_times.append(float(m.group(4)))

steps = np.array(steps)
lambda1 = np.array(lambda1)
lambda2 = np.array(lambda2)
rewards = np.array(rewards)

# Extract start and end time from sim_times
start_dt = datetime.datetime.fromtimestamp(sim_times[0])
end_dt = datetime.datetime.fromtimestamp(sim_times[-1])
time_str = f"{start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}"

print(f"Parsed {len(steps)} steps  |  "
      f"λ₁ ∈ [{lambda1.min():.3f}, {lambda1.max():.3f}]  |  "
      f"λ₂ ∈ [{lambda2.min():.3f}, {lambda2.max():.3f}]  |  "
      f"Time: {time_str}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 5))

# Background shading: green = positive reward, red = negative reward
for i in range(len(steps) - 1):
    color = "#d4edda" if rewards[i] >= 0 else "#f8d7da"   # light green / light red
    ax1.axvspan(steps[i], steps[i + 1], alpha=0.35, color=color, linewidth=0)

# λ₁ — left axis
C1 = "#c0392b"   # deep red
ax1.plot(steps, lambda1, color=C1, linewidth=2.5, label=r"$\lambda_1$ (Fairness weight)")
ax1.set_xlabel("Hours into Shift", fontsize=12, fontweight="bold")
ax1.set_ylabel(r"Fairness Weight $\lambda_1$", color=C1, fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=C1)
ax1.set_xlim(steps[0], steps[-1])
l1_pad = 0.05
ax1.set_ylim(lambda1.min() - l1_pad, lambda1.max() + l1_pad)

# λ₂ — right axis
ax2 = ax1.twinx()
C2 = "#e67e22"   # orange
ax2.plot(steps, lambda2, color=C2, linewidth=2.0, linestyle="--",
         label=r"$\lambda_2$ (Starvation weight)")
ax2.set_ylabel(r"Starvation Weight $\lambda_2$", color=C2, fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=C2)
l2_pad = 0.01
ax2.set_ylim(lambda2.min() - l2_pad, lambda2.max() + l2_pad)

# Legend — combine both axes
patch_green = mpatches.Patch(color="#d4edda", label="Positive reward step")
patch_red   = mpatches.Patch(color="#f8d7da", label="Negative reward step")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + [patch_green, patch_red],
           labels1 + labels2 + ["Positive reward step", "Negative reward step"],
           loc="upper right", fontsize=9, framealpha=0.9)

plt.title(
    "DRL Governor: Dynamic Weight Adaptation over 8-Hour Shift\n"
    f"({time_str}, Nov 28 evaluation day, Δ JFI reward)",
    fontsize=13, pad=12,
)
ax1.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = os.path.join(FIGURES_DIR, f"policy_visualization.{ext}")
    plt.savefig(out, format=ext, dpi=300, bbox_inches="tight")
    print(f"✅  Saved {out}")

plt.close()
