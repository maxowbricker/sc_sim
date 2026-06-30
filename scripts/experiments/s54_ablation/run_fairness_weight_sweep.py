#!/usr/bin/env python3
"""
Section 5.4.3 — Weighted Scorer Sensitivity (λ_f sweep)
Composite Fairness Weight Sweep — Didi 20161109

Varies the fairness weight λ_f (fairness_weight) in the Composite Scorer while
holding all other parameters fixed at the paper-final configuration.  Shows how
an operator can "dial in" the desired fairness–efficiency tradeoff by adjusting
a single scalar parameter.

The paper default (λ_f = 1.6) should appear on the curve and be highlighted.

If the surface is flat over a wide range (e.g. 1.0–2.0), that proves robustness
— the operator does not need to hand-tune the weight precisely.  If the curve is
steep, it demonstrates a meaningful tuning lever for platform operators.

Fixed params (paper-final Composite config):
    starvation_weight = 0.0   (disabled — ablation showed sw hurts idle-time equity)
    utility_weight    = 1.0   (anchors the action space)
    gamma             = 0.1   (EWMA smoothing factor — confirmed stable plateau)
    k                 = 15    (candidate pool size — Pareto-optimal from k-sweep)
    soft_threshold    = 0.0   (disabled — negligible effect found in sensitivity test)

Output: results/s3_ablation/fairness_weight_sweep_20161109.csv

Usage:
    python scripts/experiments/s3_ablation/run_fairness_weight_sweep.py
    python scripts/experiments/s3_ablation/run_fairness_weight_sweep.py --output path/to/out.csv
"""

from __future__ import annotations

import argparse
import copy
import csv as _csv
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DATA_ROOT   = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
TARGET_DAY  = "496528674@qq.com_20161109"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "s54_ablation")

# Paper-final fixed params — everything except the weight being swept
FIXED = dict(
    starvation_weight=0.0,
    utility_weight=1.0,
    gamma=0.1,
    k=15,
    soft_threshold=0.0,
)

# λ_f values to sweep — covers 0 (pure utility/spatial) through 2.0 (strong fairness)
# Paper default 1.6 is included and will be flagged in output.
WEIGHT_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

PAPER_DEFAULT_WEIGHT = 1.6

TIMEOUT_SEC = 300

FIELDNAMES = [
    "fairness_weight",
    "starvation_weight", "utility_weight", "gamma", "k", "soft_threshold",
    "TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate",
    "Gini (tasks)",
    "Avg Wait (m)", "P95 Wait (m)", "Avg Pickup (km)",
    "Completed", "Total", "elapsed_s",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_metrics(stats: Dict[str, Any], workers) -> Dict[str, Any]:
    completed  = stats.get("completed_tasks", 0)
    total      = stats.get("total_tasks", 1)
    wait_times = stats.get("wait_times", [])

    tar      = completed / total if total else 0.0
    jfi      = stats.get("final_jains_fairness_index", 0.0)
    gini     = stats.get("final_gini_coefficient", 0.0)
    jfi_earn = stats.get("final_jfi_earnings", 0.0)

    worker_task_counts = [w.completed_tasks for w in workers]
    jfi_rate = sum(1 for c in worker_task_counts if c > 0) / max(len(worker_task_counts), 1)

    avg_wait = float(np.mean(wait_times))            if wait_times else 0.0
    p95_wait = float(np.percentile(wait_times, 95))  if wait_times else 0.0

    return {
        "TAR":             tar,
        "JFI (tasks)":     jfi,
        "JFI (earnings)":  jfi_earn,
        "JFI rate":        jfi_rate,
        "Gini (tasks)":    gini,
        "Avg Wait (m)":    avg_wait,
        "P95 Wait (m)":    p95_wait,
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
        "Completed":       completed,
        "Total":           total,
    }


def _sim_thread(sim, exc_holder):
    try:
        sim.step(duration_seconds=None)
    except Exception as exc:
        exc_holder["exc"] = exc


def run_config(
    workers_template, tasks_template,
    fairness_weight: float,
    timeout_sec: float = TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    params = {**FIXED, "fairness_weight": fairness_weight}
    cfg = create_composite_config(assignment_strategy="composite", **params)
    t0  = time.time()
    exc_holder: dict = {}

    try:
        from simulator.simulation import EventSimulator
        sim = EventSimulator(
            copy.deepcopy(workers_template),
            copy.deepcopy(tasks_template),
            cfg,
        )
        sim.reset()
        thread = threading.Thread(target=_sim_thread, args=(sim, exc_holder), daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            return None
        if "exc" in exc_holder:
            raise exc_holder["exc"]

        stats   = sim.get_final_results()
        workers = list(sim.state.all_workers_map.values())

    except Exception as exc:
        print(f"FAILED: {exc}")
        return None

    m = extract_metrics(stats, workers)
    m["elapsed_s"] = time.time() - t0
    return m


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

COL_W  = 20
STAT_W = 13
DISPLAY_KEYS = ["TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate", "Gini (tasks)", "Avg Wait (m)", "P95 Wait (m)"]


def _fmt(v, key):
    if isinstance(v, float):
        if key in ("TAR", "JFI (tasks)", "JFI (earnings)", "JFI rate", "Gini (tasks)"):
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def _header():
    return f"{'λ_f (fw)':<{COL_W}}" + "".join(f" {k:>{STAT_W}}" for k in DISPLAY_KEYS) + f"  {'Time (s)':>8}"


def _sep():
    return "─" * (COL_W + (STAT_W + 1) * len(DISPLAY_KEYS) + 10)


def _row(fw, m):
    tag = "  ← paper default" if fw == PAPER_DEFAULT_WEIGHT else ""
    label = f"λ_f={fw:.1f}{tag}"
    cells = "".join(f" {_fmt(m.get(key, 0), key):>{STAT_W}}" for key in DISPLAY_KEYS)
    return f"{label:<{COL_W}}{cells}  {m.get('elapsed_s', 0):>8.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",  type=str,   default=None)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(RESULTS_DIR, "fairness_weight_sweep_20161109.csv")

    est_min = len(WEIGHT_VALUES) * 100 // 60

    print("=" * 78)
    print("  Section 5.4.3 — Composite Fairness Weight (λ_f) Sensitivity Sweep")
    print(f"  Day:        {TARGET_DAY}")
    print(f"  Fixed:      sw=0.0  uw=1.0  gamma=0.1  k=15  st=0.0")
    print(f"  λ_f values: {WEIGHT_VALUES}")
    print(f"  Paper default: λ_f={PAPER_DEFAULT_WEIGHT}")
    print(f"  Total runs: {len(WEIGHT_VALUES)}")
    print(f"  Est. time:  ~{est_min} min")
    print(f"  Output:     {output_path}")
    print("=" * 78)

    day_path = os.path.join(DATA_ROOT, TARGET_DAY)
    print(f"\n  Loading {TARGET_DAY} ...")
    workers_template, tasks_template = load_workers_tasks("didi", root_path=day_path)
    print(f"  {len(workers_template):,} workers | {len(tasks_template):,} tasks\n")

    results: List[Dict] = []

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for fw in WEIGHT_VALUES:
            tag   = "  ← paper default" if fw == PAPER_DEFAULT_WEIGHT else ""
            label = f"λ_f={fw:.1f}{tag}"
            print(f"  Running {label:<26}", end="  ", flush=True)

            m = run_config(workers_template, tasks_template, fw, args.timeout)
            if m is None:
                print("TIMEOUT / FAILED")
                continue

            print(
                f"TAR={m['TAR']:.4f}  "
                f"JFI={m['JFI (tasks)']:.4f}  "
                f"JFI(e)={m['JFI (earnings)']:.4f}  "
                f"wait={m['Avg Wait (m)']:.2f}m  "
                f"[{m['elapsed_s']:.1f}s]"
            )
            row = {"fairness_weight": fw, **FIXED, **m}
            writer.writerow(row)
            f.flush()
            m["fairness_weight"] = fw
            results.append(m)

    # Summary table
    print(f"\n\n{'=' * 78}")
    print("  RESULTS — Composite λ_f Sensitivity (Didi 20161109)")
    print(f"  Fixed: sw=0.0  uw=1.0  gamma=0.1  k=15  st=0.0")
    print(f"{'=' * 78}\n")
    print(_sep())
    print(_header())
    print(_sep())
    for r in results:
        print(_row(r["fairness_weight"], r))
    print(_sep())

    if len(results) >= 2:
        jfis  = [r["JFI (tasks)"]  for r in results]
        waits = [r["Avg Wait (m)"] for r in results]
        tars  = [r["TAR"]          for r in results]

        print(f"\n  Range across all λ_f values:")
        print(f"    Δ JFI  (tasks): {max(jfis)  - min(jfis):.4f}  "
              f"(min={min(jfis):.4f} at λ_f={results[jfis.index(min(jfis))]['fairness_weight']:.1f}, "
              f"max={max(jfis):.4f} at λ_f={results[jfis.index(max(jfis))]['fairness_weight']:.1f})")
        print(f"    Δ Wait:         {max(waits) - min(waits):.3f} m  "
              f"(min={min(waits):.3f}m at λ_f={results[waits.index(min(waits))]['fairness_weight']:.1f}, "
              f"max={max(waits):.3f}m at λ_f={results[waits.index(max(waits))]['fairness_weight']:.1f})")
        print(f"    Δ TAR:          {max(tars)  - min(tars):.4f}")

        # Highlight what happens at the paper default vs λ_f=0 (pure utility)
        base = next((r for r in results if r["fairness_weight"] == 0.0), None)
        paper = next((r for r in results if r["fairness_weight"] == PAPER_DEFAULT_WEIGHT), None)
        if base and paper:
            dj = paper["JFI (tasks)"] - base["JFI (tasks)"]
            dw = paper["Avg Wait (m)"] - base["Avg Wait (m)"]
            print(f"\n  Paper default (λ_f={PAPER_DEFAULT_WEIGHT}) vs λ_f=0.0 (pure utility):")
            print(f"    ΔJFI  = {dj:+.4f}  ({'+' if dj >= 0 else ''}{dj/max(abs(base['JFI (tasks)']), 1e-9)*100:.1f}%)")
            print(f"    ΔWait = {dw:+.3f} m")

    print(f"\n  Results saved → {output_path}")


if __name__ == "__main__":
    main()
