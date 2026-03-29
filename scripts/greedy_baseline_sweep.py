#!/usr/bin/env python3
"""
Run greedy assignment on every day folder under the DiDi GAIA root and collate metrics.

By default runs **two** passes per day:
  1. **stratified** — uses `DATA_SAMPLING` in config.py (e.g. target_tasks, bins, random_state).
  2. **full** — `use_stratified_sampling=False` (full raw day after loader).

Use `--mode` to run only one of them.

Usage:
    python scripts/greedy_baseline_sweep.py
    python scripts/greedy_baseline_sweep.py --mode full
    python scripts/greedy_baseline_sweep.py --data-root data/didi/full_didi_gaia
    python scripts/greedy_baseline_sweep.py --max-days 3   # smoke test

Writes:
    greedy_baselines_report_<timestamp>.txt   (two sections when mode=both)
    greedy_baselines_report_<timestamp>.csv   (column sampling_mode = stratified | full)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as config_module
from config import get_simulation_config
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


def run_greedy_day(
    day_name: str, data_root: str, sampling_mode: str
) -> tuple[dict | None, str | None]:
    """Returns (metrics dict, error message or None). sampling_mode: 'stratified' | 'full'."""
    day_path = os.path.join(data_root, day_name)
    try:
        workers, tasks = load_workers_tasks("didi", root_path=day_path)
    except Exception as e:
        return None, f"load failed: {e}"

    if not tasks:
        return None, "no tasks"

    cfg = get_simulation_config()
    cfg["assignment_strategy"] = "greedy"
    cfg["strategy_params"] = {"enable_deferral_tracking": False}

    t0 = time.time()
    try:
        sim = EventSimulator(workers, tasks, cfg)
        sim.reset()
        sim.step()
        elapsed = time.time() - t0
        results = sim.get_final_results()
        results["_runtime_sec"] = elapsed
        results["_day"] = day_name
        results["_sampling_mode"] = sampling_mode
        results["_n_workers"] = len(workers)
        results["_n_tasks"] = len(tasks)
        return results, None
    except Exception as e:
        return None, f"sim failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="Greedy baseline sweep across all day folders")
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
        help="Directory containing one folder per day",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "stratified", "full"),
        default="both",
        help="both = stratified (config.py) then full raw day; stratified or full = one pass only",
    )
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Legacy: true/false overrides use_stratified_sampling for a single pass (ignores --mode)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Process at most this many days (after sorting); for testing",
    )
    parser.add_argument(
        "--output-prefix",
        default="greedy_baselines_report",
        help="Output files: <prefix>_<timestamp>.txt and .csv",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root.lstrip("./"))

    if args.stratified is not None:
        modes: list[str] = ["stratified"] if args.stratified else ["full"]
    elif args.mode == "both":
        modes = ["stratified", "full"]
    else:
        modes = [args.mode]

    days = discover_day_folders(data_root)
    if args.max_days is not None:
        days = days[: args.max_days]

    if not days:
        print(f"No day folders found under {data_root}")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}.txt")
    csv_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}.csv")

    rows_out: list[dict] = []
    lines: list[str] = []
    lines.append("GREEDY BASELINE SWEEP (greedy until completion per day folder)")
    lines.append("=" * 100)
    lines.append(f"data_root: {data_root}")
    lines.append(f"days: {len(days)}")
    lines.append(f"passes: {', '.join(modes)}")
    lines.append(f"generated: {datetime.now().isoformat()}")
    lines.append("=" * 100)
    lines.append("")

    hdr = (
        f"{'day_folder':<42} | {'workers':>7} | {'tasks':>7} | {'completed':>9} | "
        f"{'compl%':>7} | {'JFI':>7} | {'peak_BL':>8} | {'avg_wait_m':>10} | {'expired':>8} | {'sec':>7} | status"
    )

    orig_stratified = config_module.DATA_SAMPLING["use_stratified_sampling"]
    try:
        for mode in modes:
            use_strat = mode == "stratified"
            config_module.DATA_SAMPLING["use_stratified_sampling"] = use_strat
            if args.stratified is not None:
                print(
                    f"DATA_SAMPLING.use_stratified_sampling = {use_strat} (legacy --stratified)\n"
                )
            else:
                print(
                    f"--- pass: {mode} (use_stratified_sampling={use_strat}) ---\n"
                )
            if mode == "stratified":
                ds = config_module.DATA_SAMPLING
                lines.append("")
                lines.append(
                    f"PASS: STRATIFIED (config.py DATA_SAMPLING: target_tasks={ds.get('target_tasks')}, "
                    f"target_workers={ds.get('target_workers')}, bins={ds.get('stratified_sampling_bins')}, "
                    f"random_state={ds.get('random_state')})"
                )
            else:
                lines.append("")
                lines.append("PASS: FULL (no stratified sampling — full raw day from loader)")

            lines.append("")
            lines.append(hdr)
            lines.append("-" * 100)

            for i, day in enumerate(days, 1):
                print(f"[{mode} {i}/{len(days)}] {day} ...", flush=True)
                res, err = run_greedy_day(day, data_root, mode)
                if err:
                    line = (
                        f"{day:<42} | {'—':>7} | {'—':>7} | {'—':>9} | {'—':>7} | {'—':>7} | "
                        f"{'—':>8} | {'—':>10} | {'—':>8} | {'—':>7} | {err}"
                    )
                    lines.append(line)
                    rows_out.append(
                        {
                            "day_folder": day,
                            "sampling_mode": mode,
                            "error": err,
                        }
                    )
                    continue

                completed = res.get("completed_tasks", 0)
                total = res.get("total_tasks", res.get("_n_tasks", 0))
                pct = (completed / total * 100.0) if total else 0.0
                jfi = res.get("final_jains_fairness_index", 0.0)
                bl = res.get("backlog_peak", 0)
                wait = res.get("avg_wait_time_minutes", 0.0)
                expired = len(res.get("expired_tasks", []))
                sec = res.get("_runtime_sec", 0.0)

                line = (
                    f"{day:<42} | {res['_n_workers']:7d} | {total:7d} | {completed:9d} | "
                    f"{pct:6.1f}% | {jfi:7.4f} | {bl:8.0f} | {wait:10.2f} | {expired:8d} | {sec:7.1f} | ok"
                )
                lines.append(line)
                rows_out.append(
                    {
                        "day_folder": day,
                        "sampling_mode": mode,
                        "n_workers": res["_n_workers"],
                        "n_tasks": total,
                        "completed_tasks": completed,
                        "completion_pct": round(pct, 2),
                        "jfi": jfi,
                        "backlog_peak": bl,
                        "avg_wait_minutes": wait,
                        "n_expired": expired,
                        "runtime_sec": round(sec, 2),
                        "total_travel_km": res.get("total_travel_km", ""),
                        "task_assignment_ratio": res.get("task_assignment_ratio", ""),
                    }
                )
    finally:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = orig_stratified

    lines.append("")
    lines.append("Notes:")
    lines.append("  stratified = loader uses DATA_SAMPLING in config.py; full = use_stratified_sampling=False.")
    lines.append("  JFI = Jain's fairness on per-worker completed task counts at end of day (0–1).")
    lines.append("  peak_BL = peak backlog during sim. avg_wait_m = mean wait over completed tasks.")
    lines.append("  expired = count of tasks in expired_tasks list.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if rows_out:
        all_keys = set()
        for r in rows_out:
            all_keys.update(r.keys())
        rest = sorted(
            k for k in all_keys if k not in ("day_folder", "sampling_mode")
        )
        fieldnames = ["day_folder", "sampling_mode"] + rest
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows_out:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    print()
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
