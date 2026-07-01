#!/usr/bin/env python3
"""
Plot: Impact of k on JFI and Wait Time  (§5.4.1)

Two-panel figure:
  Panel A — JFI (tasks) vs k, with Greedy and LAF reference lines
  Panel B — Avg Wait (m) vs k, with Greedy reference line

Input:  results/s54_ablation/knlf_k_sweep_final.csv
Output: results/figures/k_sweep.pdf
        results/figures/k_sweep.png  (preview)

Usage:
    python scripts/plots/plot_k_sweep.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
INPUT  = os.path.join(PROJECT_ROOT, "results", "s54_ablation",
                      "knlf_k_sweep_final.csv")
OUT_PDF = os.path.join(PROJECT_ROOT, "results", "figures", "k_sweep.pdf")
OUT_PNG = os.path.join(PROJECT_ROOT, "results", "figures", "k_sweep.png")

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
# Use Times New Roman to match LNCS/Springer body font.
# If usetex=True is preferred (requires a LaTeX installation), swap the
# font.family lines for:  plt.rcParams["text.usetex"] = True
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

PAPER_K       = 15
COLOR_KNLF    = "#2166ac"   # blue  — consistent across all three panels
COLOR_COMP    = "#e66101"   # orange
COLOR_GREEDY  = "#888888"
COLOR_LAF     = "#4dac26"

# Aliases kept for readability; all three panels use COLOR_KNLF for k-NLF lines
COLOR_JFI  = COLOR_KNLF
COLOR_WAIT = COLOR_KNLF
COLOR_TIME = COLOR_KNLF

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)

# Separate anchors from the k-NLF sweep
greedy_row  = df[df["strategy"] == "greedy"].iloc[0]
_laf_rows   = df[df["strategy"] == "laf"]
laf_row     = _laf_rows.iloc[0] if len(_laf_rows) > 0 else None
has_laf     = laf_row is not None
sweep_knlf  = df[df["strategy"] == "knlf"].copy()
sweep_comp = df[df["strategy"] == "composite"].copy()

for s in (sweep_knlf, sweep_comp):
    s["k"] = s["k"].astype(int)

sweep_knlf = sweep_knlf.sort_values("k")
sweep_comp = sweep_comp.sort_values("k")

k_vals  = sweep_knlf["k"].tolist()
jfi     = sweep_knlf["JFI (tasks)"].tolist()
wait    = sweep_knlf["Avg Wait (m)"].tolist()
runtime = sweep_knlf["elapsed_s"].tolist()

has_composite = len(sweep_comp) > 0
if has_composite:
    k_vals_comp  = sweep_comp["k"].tolist()
    jfi_comp     = sweep_comp["JFI (tasks)"].tolist()
    wait_comp    = sweep_comp["Avg Wait (m)"].tolist()
    runtime_comp = sweep_comp["elapsed_s"].tolist()

greedy_jfi  = greedy_row["JFI (tasks)"]
greedy_wait = greedy_row["Avg Wait (m)"]
laf_jfi     = laf_row["JFI (tasks)"] if has_laf else None

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.0, 2.9))
fig.subplots_adjust(wspace=0.40, bottom=0.30)

# ── Panel A: JFI (tasks) vs k ─────────────────────────────────────────────────
h_knlf, = ax1.plot(k_vals, jfi, marker="o", markersize=4, linewidth=1.6,
                   color=COLOR_KNLF)
if has_composite:
    h_comp, = ax1.plot(k_vals_comp, jfi_comp, marker="D", markersize=4,
                       linewidth=1.4, linestyle="--", color=COLOR_COMP)

h_greedy, = ax1.plot([], [], linestyle="--", linewidth=1.0, color=COLOR_GREEDY)
ax1.axhline(greedy_jfi, linestyle="--", linewidth=1.0, color=COLOR_GREEDY)
if has_laf:
    h_laf, = ax1.plot([], [], linestyle=":", linewidth=1.0, color=COLOR_LAF)
    ax1.axhline(laf_jfi, linestyle=":", linewidth=1.0, color=COLOR_LAF)

ax1.axvline(PAPER_K, linestyle="--", linewidth=0.8, color="#aaaaaa")
ax1.text(PAPER_K + 1, min(jfi) - 0.005,
         f"k={PAPER_K}", fontsize=7, color="#888888", va="bottom")

ax1.set_xlabel("Candidate pool size $k$")
ax1.set_ylabel("JFI (tasks)")
ax1.set_title("(a) Fairness vs $k$")
ax1.set_xscale("log")
ax1.set_xticks(k_vals)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

# ── Panel B: Avg Wait vs k ────────────────────────────────────────────────────
ax2.plot(k_vals, wait, marker="s", markersize=4, linewidth=1.6, color=COLOR_KNLF)
if has_composite:
    ax2.plot(k_vals_comp, wait_comp, marker="D", markersize=4, linewidth=1.4,
             linestyle="--", color=COLOR_COMP)

ax2.axhline(greedy_wait, linestyle="--", linewidth=1.0, color=COLOR_GREEDY)

ax2.axvline(PAPER_K, linestyle="--", linewidth=0.8, color="#aaaaaa")
ax2.text(PAPER_K + 1, min(wait) - 0.05,
         f"k={PAPER_K}", fontsize=7, color="#888888", va="bottom")

ax2.set_xlabel("Candidate pool size $k$")
ax2.set_ylabel("Avg wait time (min)")
ax2.set_title("(b) Wait Time vs $k$")
ax2.set_xscale("log")
ax2.set_xticks(k_vals)
ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

# ── Panel C: Runtime vs k ─────────────────────────────────────────────────────
ax3.plot(k_vals, runtime, marker="^", markersize=4, linewidth=1.6, color=COLOR_KNLF)
if has_composite:
    ax3.plot(k_vals_comp, runtime_comp, marker="D", markersize=4, linewidth=1.4,
             linestyle="--", color=COLOR_COMP)

# Greedy is not k-parameterised (O(|W|) global scan) so no reference line here.
if has_laf:
    ax3.axhline(laf_row["elapsed_s"], linestyle=":", linewidth=1.0, color=COLOR_LAF)

ax3.axvline(PAPER_K, linestyle="--", linewidth=0.8, color="#aaaaaa")
ax3.text(PAPER_K + 1, min(runtime) - 1.5,
         f"k={PAPER_K}", fontsize=7, color="#888888", va="bottom")

ax3.set_xlabel("Candidate pool size $k$")
ax3.set_ylabel("Wall-clock time (s)")
ax3.set_title("(c) Runtime vs $k$")
ax3.set_xscale("log")
ax3.set_xticks(k_vals)
ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

# ── Shared bottom legend ───────────────────────────────────────────────────────
_leg_handles = [h_knlf]
_leg_labels  = [r"k-NLF ($k{=}15$)$^\dagger$"]
if has_composite:
    _leg_handles.append(h_comp)
    _leg_labels.append(r"Composite$^\dagger$")
_leg_handles.append(h_greedy)
_leg_labels.append("Greedy (baseline)")
if has_laf:
    _leg_handles.append(h_laf)
    _leg_labels.append("LAF")

fig.legend(
    _leg_handles, _leg_labels,
    loc="lower center",
    ncol=len(_leg_handles),
    frameon=False,
    fontsize=7.5,
    bbox_to_anchor=(0.5, 0.01),
)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved → {OUT_PDF}")
print(f"Saved → {OUT_PNG}")
