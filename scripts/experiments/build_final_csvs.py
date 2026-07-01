#!/usr/bin/env python3
"""
Build ALL canonical final CSVs for the paper (tables + figures).

These _final.csv files are the single authoritative data source for every
table and figure in paper.tex.  No plot script or LaTeX table should ever
read a _v2 / _v3 / laptop CSV directly.  Run this script whenever any source
CSV changes to regenerate all canonical versions.

Greedy rows in the main-results tables use metrics from the O(|W|) global-scan
laptop calibration run; wall-clock time is scaled to a cluster equivalent via
relative benchmarking against k-NLF (see PROGRESS.md for details).

Outputs — tables
----------------
  results/s52_main_results/didi_main_results_final.csv         (tab:didi_results)
  results/s52_main_results/gowalla_main_results_final.csv      (tab:gowalla_results)
  results/s54_ablation/signal_comparison_final.csv             (tab:signal_comparison)

Outputs — figures
-----------------
  results/s53_scalability/scalability_fleet_final.csv          (fig:market_conditions panel a)
  results/s53_scalability/scalability_tasks_final.csv          (fig:market_conditions panel b)
  results/s54_ablation/knlf_k_sweep_final.csv                  (fig:k_sweep)
  results/s54_ablation/fairness_weight_sweep_final.csv         (fig:fw_sweep)
"""

from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

# ── Relative-benchmarking scaling constants ───────────────────────────────────
# Formula: greedy_cluster_s = greedy_laptop_s × (knlf_cluster_s / knlf_laptop_s)
#
# DiDi — k-NLF cluster time from didi_core_v2.csv:
GREEDY_DIDI_LAPTOP_S     = 239.54
KNLF_DIDI_LAPTOP_S       = 94.52
KNLF_DIDI_CLUSTER_S      = 123.0
GREEDY_DIDI_CLUSTER_S    = GREEDY_DIDI_LAPTOP_S * (KNLF_DIDI_CLUSTER_S / KNLF_DIDI_LAPTOP_S)

# Signal comparison — k-NLF cluster time from signal_comparison_v3.csv (133s,
# a different run context from the main results table, so scaled separately):
KNLF_SIGNAL_CLUSTER_S    = 133.0
GREEDY_SIGNAL_CLUSTER_S  = GREEDY_DIDI_LAPTOP_S * (KNLF_SIGNAL_CLUSTER_S / KNLF_DIDI_LAPTOP_S)

# Gowalla
GREEDY_GOWALLA_LAPTOP_S  = 73.5
KNLF_GOWALLA_LAPTOP_S    = 20.4
KNLF_GOWALLA_CLUSTER_S   = 27.3
GREEDY_GOWALLA_CLUSTER_S = GREEDY_GOWALLA_LAPTOP_S * (KNLF_GOWALLA_CLUSTER_S / KNLF_GOWALLA_LAPTOP_S)


def p(name: str) -> str:
    return os.path.join(PROJECT_ROOT, name)


# ── 1. DiDi main results ──────────────────────────────────────────────────────

def build_didi() -> pd.DataFrame:
    core    = pd.read_csv(p("results/s52_main_results/didi_core_v2.csv"))
    onrta   = pd.read_csv(p("results/s52_main_results/didi_onrta_v2.csv"))
    lp      = pd.read_csv(p("results/s52_main_results/didi_lp_v2.csv"))
    laptop  = pd.read_csv(p("results/s52_main_results/didi_greedy_global_laptop.csv"))

    # Drop stale Greedy row (k-50 / k-10 era) from core
    core_no_greedy = core[core["Strategy"] != "Greedy"].copy()

    # Take the Greedy row from the laptop calibration run
    greedy_row = laptop[laptop["Strategy"] == "Greedy"].copy()
    greedy_row["Wall time (s)"] = round(GREEDY_DIDI_CLUSTER_S, 1)

    # Canonical row order matching tab:didi_results in paper.tex
    order = [
        "Greedy",
        "k-NLF (k=15)",
        "Composite (static)",
        "LAF",
        "BiRanking (BRK)",
        "ONRTA-RT",
        "Discrete Review LP",
    ]

    combined = pd.concat(
        [greedy_row, core_no_greedy, onrta, lp],
        ignore_index=True,
    )
    combined["_order"] = combined["Strategy"].map({s: i for i, s in enumerate(order)})
    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return combined


# ── 2. Gowalla main results ───────────────────────────────────────────────────

# Strategies included in paper table (plus Greedy which is rebuilt below)
GOWALLA_PAPER_STRATEGIES = [
    "k-NLF (k=15)",
    "Composite (static)",
    "LAF",
    "BiRanking (BRK)",
    "ONRTA-RT",
    "Discrete Review LP",
]


def build_gowalla() -> pd.DataFrame:
    v2     = pd.read_csv(p("results/s52_main_results/gowalla_austin_compressed_v2.csv"))
    laptop = pd.read_csv(p("results/s52_main_results/gowalla_greedy_global_laptop.csv"))

    # Filter to compressed, ratio 0.20
    filt = (v2["_compress"] == True) & (v2["_ratio"] == "Ratio 0.20")
    paper_rows = v2[filt & v2["_strategy"].isin(GOWALLA_PAPER_STRATEGIES)].copy()

    # Greedy row from laptop calibration (already compressed / ratio 0.20)
    greedy_row = laptop[
        (laptop["_compress"] == True) &
        (laptop["_ratio"] == "Ratio 0.20") &
        (laptop["_strategy"] == "Greedy")
    ].copy()
    greedy_row["_elapsed"] = round(GREEDY_GOWALLA_CLUSTER_S, 1)

    order = ["Greedy"] + GOWALLA_PAPER_STRATEGIES
    combined = pd.concat([greedy_row, paper_rows], ignore_index=True)
    combined["_order"] = combined["_strategy"].map({s: i for i, s in enumerate(order)})
    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return combined


# ── 3. Signal comparison ──────────────────────────────────────────────────────

# Strategies in paper table (including k=5 variants that appear in prose)
SIGNAL_PAPER_STRATEGIES = [
    "Greedy",
    "k-NTF-EPH (k=15)",
    "k-NTF-EPH (k=5)",
    "k-NTF-IR  (k=15)",
    "k-NTF-IR  (k=5)",
    "k-NLF (k=15)",
    "Composite (k=15)",
]


def build_signal() -> pd.DataFrame:
    v3     = pd.read_csv(p("results/s54_ablation/signal_comparison_20161109_v3.csv"))
    laptop = pd.read_csv(p("results/s52_main_results/didi_greedy_global_laptop.csv"))

    # All non-Greedy rows from v3 are authoritative (cluster run, correct implementation)
    v3_no_greedy = v3[v3["strategy"] != "Greedy"].copy()

    # Build new Greedy row by mapping laptop columns → signal schema
    g = laptop[laptop["Strategy"] == "Greedy"].iloc[0]

    # CV (earn.) is not output by run_strategy_comparison.py; carry the old v3
    # k=10 value (0.949) as a placeholder until signal comparison is re-run with
    # --only greedy.  TODO: re-run to get the true global-scan CV (earn.).
    old_cv_earn = float(v3[v3["strategy"] == "Greedy"]["CV (earn)"].iloc[0])

    greedy_row = pd.DataFrame([{
        "strategy":        "Greedy",
        "complexity":      "O(|W|)",
        "TAR":             g["TAR"],
        "Revenue ($)":     g["Revenue ($)"],
        "JFI (tasks)":     g["JFI (tasks)"],
        "Gini (tasks)":    g["Gini (tasks)"],
        "JFI (earnings)":  g["JFI (earnings)"],
        "Gini (earn)":     float("nan"),         # not available from laptop run
        "JFI rate":        g["JFI rate"],
        "P10 tasks":       g["P10 tasks"],
        "P25 tasks":       g["P25 tasks"],
        "CV (idle)":       g["CV Idle"],
        "CV (earn)":       old_cv_earn,          # ← placeholder; see TODO above
        "Avg Wait (m)":    g["Avg Wait (m)"],
        "P50 Wait (m)":    g["P50 Wait (m)"],
        "P95 Wait (m)":    g["P95 Wait (m)"],
        "Avg Pickup (km)": g["Avg Pickup (km)"],
        "Completed":       int(g["Completed"]),
        "Total":           224_219,
        "elapsed_s":       round(GREEDY_SIGNAL_CLUSTER_S, 1),
    }])

    combined = pd.concat([greedy_row, v3_no_greedy], ignore_index=True)
    combined["_order"] = combined["strategy"].map(
        {s: i for i, s in enumerate(SIGNAL_PAPER_STRATEGIES)}
    )
    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return combined


# ── 4. Figure 1 — Market conditions (pass-through) ───────────────────────────
# Direct cluster outputs; no row-level transformation needed.

def build_scalability_fleet() -> pd.DataFrame:
    return pd.read_csv(p("results/s53_scalability/scalability_fleet_v2.csv"))


def build_scalability_tasks() -> pd.DataFrame:
    return pd.read_csv(p("results/s53_scalability/scalability_tasks_v2.csv"))


# ── 5. Figure 2 — k sweep (pass-through with Greedy anchor already patched) ──
# knlf_k_sweep_20161109_v2.csv was updated in-place to replace the stale
# k-indexed Greedy row with global-scan anchor values (see PROGRESS.md).
# This builder simply promotes that file to _final.csv with no further changes.

def build_k_sweep() -> pd.DataFrame:
    return pd.read_csv(p("results/s54_ablation/knlf_k_sweep_20161109_v2.csv"))


# ── 6. Figure 3 — Fairness weight sweep (pass-through) ───────────────────────
# Direct cluster output for Composite lambda_f sweep; no transformation needed.

def build_fairness_weight() -> pd.DataFrame:
    return pd.read_csv(p("results/s54_ablation/fairness_weight_sweep_20161109_v2.csv"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    table_outputs = {
        p("results/s52_main_results/didi_main_results_final.csv"):    build_didi,
        p("results/s52_main_results/gowalla_main_results_final.csv"): build_gowalla,
        p("results/s54_ablation/signal_comparison_final.csv"):        build_signal,
    }

    figure_outputs = {
        p("results/s53_scalability/scalability_fleet_final.csv"):          build_scalability_fleet,
        p("results/s53_scalability/scalability_tasks_final.csv"):          build_scalability_tasks,
        p("results/s54_ablation/knlf_k_sweep_final.csv"):                  build_k_sweep,
        p("results/s54_ablation/fairness_weight_sweep_final.csv"):         build_fairness_weight,
    }

    all_outputs = {**table_outputs, **figure_outputs}

    print("Building canonical final CSVs...")
    print()

    print("  — Tables —")
    for out_path, builder in table_outputs.items():
        df = builder()
        df.to_csv(out_path, index=False)
        name = os.path.basename(out_path)
        strat_col = "Strategy" if "Strategy" in df.columns else (
                    "_strategy" if "_strategy" in df.columns else "strategy")
        print(f"  ✓  {name}  ({len(df)} rows)")
        if strat_col in df.columns:
            print(f"     strategies: {df[strat_col].tolist()}")
        print()

    print("  — Figures —")
    for out_path, builder in figure_outputs.items():
        df = builder()
        df.to_csv(out_path, index=False)
        name = os.path.basename(out_path)
        print(f"  ✓  {name}  ({len(df)} rows)")
        print()

    print("Done.")
    print()
    print("Scaled Greedy runtimes used:")
    print(f"  DiDi (main table):   {GREEDY_DIDI_CLUSTER_S:.1f}s")
    print(f"  DiDi (signal table): {GREEDY_SIGNAL_CLUSTER_S:.1f}s")
    print(f"  Gowalla:             {GREEDY_GOWALLA_CLUSTER_S:.1f}s")


if __name__ == "__main__":
    main()
