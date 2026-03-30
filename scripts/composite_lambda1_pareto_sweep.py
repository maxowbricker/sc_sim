#!/usr/bin/env python3
"""
Sweep static composite λ1 (fairness_weight) on the full loaded day: EventSimulator runs
until the event queue is exhausted (no greedy warmup, no 8 h window — entire dataset).

For each day folder (sorted order): load data once, run every λ1 in the grid, collate,
then move to the next day. That keeps JFI vs wait comparable within a day (same scenario).

Stratified sampling follows config.py DATA_SAMPLING when use_stratified_sampling=True
(same pattern as scripts/greedy_baseline_sweep.py).

Outputs:
  composite_lambda1_pareto_<timestamp>.csv
      — one row per (day, λ1); pareto_efficient_within_day; **ΔJFI stats** from 5-min stepping
        (same cadence as rl/gym_environment.py): consecutive snapshot deltas match obs/reward
        momentum; delta_jfi_obs_style_first ≈ first obs delta if prev_jfi=1 at reset
  composite_lambda1_pareto_<timestamp>_across_days.csv
      — per λ1: mean/std/min/max of JFI and wait; **pooled ΔJFI magnitude** across days
  composite_lambda1_pareto_<timestamp>_delta_steps.csv  (only with --save-delta-steps)
      — one row per snapshot: jfi, consecutive ΔJFI, step avg wait, consecutive Δwait

Usage:
  python scripts/composite_lambda1_pareto_sweep.py
  python scripts/composite_lambda1_pareto_sweep.py --days foo,bar --stratified false
  python scripts/composite_lambda1_pareto_sweep.py --max-days 2 --save-delta-steps   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

# Same default as rl/gym_environment.AdaptiveSpatialCrowdsourcingEnv (step_duration_minutes=5)
DEFAULT_STEP_MINUTES = 5.0
# Gym reset() sets prev_jfi = 1.0 for the first observation delta channel
GYM_OBS_PREV_JFI = 1.0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as config_module
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator


def discover_day_folders(data_root: str) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    return sorted(
        d
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")
    )


def parse_lambda1_grid(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive grid from lo to hi by step (e.g. 0.1 … 2.0 step 0.1 → 20 points)."""
    if step <= 0:
        raise ValueError("lambda1-step must be positive")
    vals: list[float] = []
    x = lo
    while x <= hi + 1e-9:
        vals.append(round(float(x), 10))
        x += step
    if not vals:
        vals = [lo]
    if vals[-1] > hi + 1e-9:
        vals[-1] = round(hi, 10)
    return vals


def pareto_mask_min_wait_max_jfi(waits: np.ndarray, jfis: np.ndarray) -> np.ndarray:
    """Minimize wait, maximize JFI; non-dominated points."""
    n = len(waits)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        wi, ji = waits[i], jfis[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            wj, jj = waits[j], jfis[j]
            if (wj <= wi and jj >= ji) and (wj < wi or jj > ji):
                dominated = True
                break
        out[i] = not dominated
    return out


def _empty_delta_stats() -> dict[str, Any]:
    keys = (
        "n_snapshots",
        "n_delta_steps",
        "jfi_first_snapshot",
        "delta_jfi_obs_style_first",
        "delta_jfi_mean",
        "delta_jfi_std",
        "delta_jfi_min",
        "delta_jfi_max",
        "delta_jfi_abs_mean",
        "delta_jfi_abs_p50",
        "delta_jfi_abs_p90",
        "delta_jfi_abs_p99",
        "delta_jfi_abs_max",
        "delta_avg_wait_mean",
        "delta_avg_wait_std",
        "delta_avg_wait_abs_p90",
        "delta_avg_wait_abs_p99",
        "delta_avg_wait_abs_max",
    )
    return {k: "" for k in keys}


def compute_delta_series_stats(
    jfi_series: list[float],
    wait_series: list[float],
    obs_prev_jfi: float = GYM_OBS_PREV_JFI,
) -> dict[str, Any]:
    """
    Consecutive ΔJFI / Δwait between 5-min snapshots (matches gym obs after first step).
    delta_jfi_obs_style_first = jfi[0] - obs_prev_jfi (first obs delta if prev_jfi fixed at reset).
    """
    out = _empty_delta_stats()
    if not jfi_series:
        return out

    jf = np.asarray(jfi_series, dtype=float)
    wt = np.asarray(wait_series, dtype=float)
    out["n_snapshots"] = len(jf)
    out["jfi_first_snapshot"] = float(jf[0])
    out["delta_jfi_obs_style_first"] = float(jf[0] - obs_prev_jfi)

    dj = np.diff(jf)
    dw = np.diff(wt)
    out["n_delta_steps"] = int(len(dj))
    if len(dj) == 0:
        return out

    ad = np.abs(dj)
    aw = np.abs(dw)
    out["delta_jfi_mean"] = float(np.mean(dj))
    out["delta_jfi_std"] = float(np.std(dj, ddof=1)) if len(dj) > 1 else 0.0
    out["delta_jfi_min"] = float(np.min(dj))
    out["delta_jfi_max"] = float(np.max(dj))
    out["delta_jfi_abs_mean"] = float(np.mean(ad))
    out["delta_jfi_abs_p50"] = float(np.percentile(ad, 50))
    out["delta_jfi_abs_p90"] = float(np.percentile(ad, 90))
    out["delta_jfi_abs_p99"] = float(np.percentile(ad, 99))
    out["delta_jfi_abs_max"] = float(np.max(ad))
    out["delta_avg_wait_mean"] = float(np.mean(dw))
    out["delta_avg_wait_std"] = float(np.std(dw, ddof=1)) if len(dw) > 1 else 0.0
    out["delta_avg_wait_abs_p90"] = float(np.percentile(aw, 90))
    out["delta_avg_wait_abs_p99"] = float(np.percentile(aw, 99))
    out["delta_avg_wait_abs_max"] = float(np.max(aw))
    return out


def build_delta_step_trace_rows(
    day: str,
    lambda1: float,
    lambda2: float,
    jfi_series: list[float],
    wait_series: list[float],
    sim_times: list[float],
    obs_prev_jfi: float = GYM_OBS_PREV_JFI,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for i in range(len(jfi_series)):
        jfi = jfi_series[i]
        w = wait_series[i]
        t_end = sim_times[i] if i < len(sim_times) else float("nan")
        if i == 0:
            rows_out.append(
                {
                    "day_folder": day,
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "step_index": i,
                    "sim_time_end": t_end,
                    "jfi": float(jfi),
                    "delta_jfi_obs_style_first": float(jfi - obs_prev_jfi),
                    "delta_jfi_consecutive": "",
                    "avg_wait_minutes": float(w),
                    "delta_avg_wait_consecutive": "",
                }
            )
        else:
            rows_out.append(
                {
                    "day_folder": day,
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "step_index": i,
                    "sim_time_end": t_end,
                    "jfi": float(jfi),
                    "delta_jfi_obs_style_first": "",
                    "delta_jfi_consecutive": float(jfi - jfi_series[i - 1]),
                    "avg_wait_minutes": float(w),
                    "delta_avg_wait_consecutive": float(w - wait_series[i - 1]),
                }
            )
    return rows_out


def run_composite_full_day(
    day: str,
    data_root: str,
    lambda1: float,
    lambda2: float,
    step_minutes: float = DEFAULT_STEP_MINUTES,
    collect_delta_trace: bool = False,
) -> tuple[
    dict[str, Any] | None,
    float | None,
    str | None,
    dict[str, Any],
    list[dict[str, Any]] | None,
]:
    """
    Full-day composite run using fixed-duration steps (default 5 min, same as gym).
    Records JFI and step avg wait at each snapshot to summarize ΔJFI / Δwait distributions.
    Returns (stats, elapsed, err, delta_stats_dict, delta_trace_or_none).
    """
    day_path = os.path.join(data_root, day)
    cfg0 = get_simulation_config()
    dataset = cfg0["dataset"]

    try:
        workers, tasks = load_workers_tasks(dataset=dataset, root_path=day_path)
    except Exception as e:
        return None, None, f"load failed: {e}", _empty_delta_stats(), None

    if not tasks:
        return None, None, "no tasks", _empty_delta_stats(), None

    cfg = get_simulation_config()
    cfg["assignment_strategy"] = "composite"
    sp = dict(get_strategy_params("composite"))
    sp["normalize_scores"] = True
    sp["fairness_weight"] = float(lambda1)
    sp["starvation_weight"] = float(lambda2)
    cfg["strategy_params"] = sp

    step_sec = float(step_minutes) * 60.0
    jfi_series: list[float] = []
    wait_series: list[float] = []
    sim_times: list[float] = []

    t0 = time.time()
    try:
        sim = EventSimulator(workers, tasks, cfg)
        sim.reset()
        while True:
            done = sim.step(duration_seconds=step_sec)
            st = sim.metrics.current_step_stats
            jfi_series.append(float(st.get("jfi", 1.0)))
            wait_series.append(float(st.get("avg_wait", 0.0)))
            sim_times.append(float(sim.current_time))
            if done:
                break
        stats = sim.get_final_results()
    except Exception as e:
        return None, None, f"sim failed: {e}", _empty_delta_stats(), None

    delta_stats = compute_delta_series_stats(jfi_series, wait_series)
    trace = None
    if collect_delta_trace and jfi_series:
        trace = build_delta_step_trace_rows(
            day, lambda1, lambda2, jfi_series, wait_series, sim_times
        )

    return stats, time.time() - t0, None, delta_stats, trace


def assign_within_day_pareto(rows: list[dict[str, Any]]) -> None:
    """Set pareto_efficient_within_day per row, grouped by day_folder."""
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_day[str(r["day_folder"])].append(i)

    for _day, indices in by_day.items():
        waits = np.array([float(rows[i]["avg_wait_time_minutes"]) for i in indices])
        jfis = np.array([float(rows[i]["final_jains_fairness_index"]) for i in indices])
        mask = pareto_mask_min_wait_max_jfi(waits, jfis)
        for k, idx in enumerate(indices):
            rows[idx]["pareto_efficient_within_day"] = bool(mask[k])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-day static composite: λ1 sweep per day, then next day (JFI vs wait tradeoff)"
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
        help="Directory containing one folder per day",
    )
    parser.add_argument(
        "--days",
        default=None,
        help="Comma-separated day folders (default: all folders under data-root, sorted)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Process only the first N sorted days (for testing)",
    )
    parser.add_argument("--lambda1-min", type=float, default=0.1)
    parser.add_argument("--lambda1-max", type=float, default=2.0)
    parser.add_argument("--lambda1-step", type=float, default=0.1)
    parser.add_argument(
        "--lambda2",
        type=float,
        default=None,
        help="starvation_weight; default = config composite",
    )
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() == "true",
        default=True,
        help="true: DATA_SAMPLING stratified load; false: full raw day",
    )
    parser.add_argument(
        "--output-prefix",
        default="composite_lambda1_pareto",
        help="Output CSV prefix under project root",
    )
    parser.add_argument(
        "--step-minutes",
        type=float,
        default=DEFAULT_STEP_MINUTES,
        help="Wall-clock sim duration per step (must match gym step_duration_minutes for ΔJFI comparability)",
    )
    parser.add_argument(
        "--save-delta-steps",
        action="store_true",
        help="Write per-snapshot rows (jfi, ΔJFI, wait, Δwait) to *_delta_steps_<ts>.csv (large file)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root.lstrip("./"))

    all_days = discover_day_folders(data_root)
    if not all_days:
        print(f"No day folders under {data_root}")
        return 1

    if args.days:
        days = [d.strip() for d in args.days.split(",") if d.strip()]
    else:
        days = list(all_days)
        if args.max_days is not None:
            days = days[: args.max_days]

    lambda_grid = parse_lambda1_grid(args.lambda1_min, args.lambda1_max, args.lambda1_step)
    lambda2_val = args.lambda2
    if lambda2_val is None:
        lambda2_val = float(get_strategy_params("composite")["starvation_weight"])

    orig_strat = config_module.DATA_SAMPLING["use_stratified_sampling"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}.csv")
    across_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}_across_days.csv")
    delta_steps_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_delta_steps_{ts}.csv")

    rows: list[dict[str, Any]] = []
    delta_step_rows: list[dict[str, Any]] = []
    empty_delta = _empty_delta_stats()
    print(
        f"Days ({len(days)}), sorted / explicit order: first={days[0] if days else '—'} …\n"
        f"λ1 grid: {lambda_grid[0]} … {lambda_grid[-1]} ({len(lambda_grid)} sims per day), "
        f"λ2={lambda2_val}, stratified={args.stratified}, step_minutes={args.step_minutes}\n"
    )

    total = len(days) * len(lambda_grid)
    run_idx = 0

    try:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified
        for day in days:
            print(f"--- Day: {day} ({len(lambda_grid)} λ1 runs) ---", flush=True)
            for lam1 in lambda_grid:
                run_idx += 1
                print(f"  [{run_idx}/{total}] λ1={lam1:.4f} ...", flush=True)
                stats, elapsed, err, delta_stats, trace = run_composite_full_day(
                    day=day,
                    data_root=data_root,
                    lambda1=lam1,
                    lambda2=lambda2_val,
                    step_minutes=args.step_minutes,
                    collect_delta_trace=args.save_delta_steps,
                )
                base = {
                    "day_folder": day,
                    "lambda1": lam1,
                    "lambda2": lambda2_val,
                    "lambda3": float(get_strategy_params("composite")["utility_weight"]),
                }
                if err:
                    rows.append(
                        {
                            **base,
                            "final_jains_fairness_index": "",
                            "avg_wait_time_minutes": "",
                            "completed_tasks": "",
                            "n_expired": "",
                            "runtime_sec": "",
                            "error": err,
                            "stratified": args.stratified,
                            "pareto_efficient_within_day": "",
                            **empty_delta,
                        }
                    )
                    continue

                assert stats is not None and elapsed is not None
                if args.save_delta_steps and trace:
                    delta_step_rows.extend(trace)

                jfi = float(stats.get("final_jains_fairness_index", 0.0))
                wait = float(stats.get("avg_wait_time_minutes", 0.0))
                completed = int(stats.get("completed_tasks", 0))
                expired = len(stats.get("expired_tasks", []) or [])
                rows.append(
                    {
                        **base,
                        "final_jains_fairness_index": jfi,
                        "avg_wait_time_minutes": wait,
                        "completed_tasks": completed,
                        "n_expired": expired,
                        "runtime_sec": round(elapsed, 2),
                        "error": "",
                        "stratified": args.stratified,
                        "pareto_efficient_within_day": False,
                        **delta_stats,
                    }
                )
    finally:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = orig_strat

    ok_rows = [r for r in rows if r.get("error") in ("", None)]
    assign_within_day_pareto(ok_rows)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Across days: per λ1, stats over days (only successful rows)
    by_l1: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for r in ok_rows:
        by_l1[float(r["lambda1"])].append(
            (float(r["avg_wait_time_minutes"]), float(r["final_jains_fairness_index"]))
        )

    across_rows: list[dict[str, Any]] = []
    for l1 in sorted(by_l1.keys()):
        pts = by_l1[l1]
        waits = [p[0] for p in pts]
        jfis = [p[1] for p in pts]
        wa = np.array(waits)
        ja = np.array(jfis)
        across_rows.append(
            {
                "lambda1": l1,
                "n_days": len(pts),
                "mean_jfi": float(np.mean(ja)),
                "std_jfi": float(np.std(ja, ddof=1)) if len(ja) > 1 else 0.0,
                "min_jfi": float(np.min(ja)),
                "max_jfi": float(np.max(ja)),
                "mean_avg_wait_minutes": float(np.mean(wa)),
                "std_avg_wait_minutes": float(np.std(wa, ddof=1)) if len(wa) > 1 else 0.0,
                "min_avg_wait_minutes": float(np.min(wa)),
                "max_avg_wait_minutes": float(np.max(wa)),
            }
        )

    if across_rows:
        # Pool ΔJFI magnitudes across days at each λ1 (for obs max_abs_jfi_delta / reward scale)
        by_l1_dabs: dict[float, list[float]] = defaultdict(list)
        by_l1_d99: dict[float, list[float]] = defaultdict(list)
        for r in ok_rows:
            l1 = float(r["lambda1"])
            v = r.get("delta_jfi_abs_max")
            if v != "" and v is not None:
                by_l1_dabs[l1].append(float(v))
            v99 = r.get("delta_jfi_abs_p99")
            if v99 != "" and v99 is not None:
                by_l1_d99[l1].append(float(v99))

        for s in across_rows:
            l1 = float(s["lambda1"])
            xs = by_l1_d99.get(l1, [])
            xm = by_l1_dabs.get(l1, [])
            s["delta_jfi_abs_p99_mean_across_days"] = float(np.mean(xs)) if xs else ""
            s["delta_jfi_abs_p99_std_across_days"] = (
                float(np.std(xs, ddof=1)) if len(xs) > 1 else (0.0 if xs else "")
            )
            s["delta_jfi_abs_max_mean_across_days"] = float(np.mean(xm)) if xm else ""
            s["delta_jfi_abs_max_std_across_days"] = (
                float(np.std(xm, ddof=1)) if len(xm) > 1 else (0.0 if xm else "")
            )

        with open(across_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(across_rows[0].keys()))
            w.writeheader()
            for s in across_rows:
                w.writerow(s)

    if delta_step_rows:
        with open(delta_steps_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_step_rows[0].keys()))
            w.writeheader()
            for dr in delta_step_rows:
                w.writerow(dr)

    print()
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {across_path}")
    if delta_step_rows:
        print(f"Wrote: {delta_steps_path}")
    print(
        "Within each day, pareto_efficient_within_day = non-dominated (min wait, max JFI) over λ1. "
        "ΔJFI columns: consecutive 5-min snapshot deltas (align with gym obs/reward). "
        "Use delta_jfi_abs_p99 / max to tune config OBSERVATION_STATIC_SCALING max_abs_jfi_delta "
        "and fairness term scale vs latency."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
