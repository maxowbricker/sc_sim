#!/usr/bin/env python3
"""
Sample greedy warmup JFI many times with the same randomness protocol as the RL gym env:
  random day from data root → random drop-in start_time → 30 min greedy sim → read JFI.

Use this to see what greedy_baseline_jfi would typically be after warmup (active-worker JFI
from metrics.manager snapshot_step).

Usage:
    python scripts/warmup_jfi_baseline_sample.py --n-runs 30 --stratified false
    python scripts/warmup_jfi_baseline_sample.py --n-runs 10 --base-seed 0 --max-days 5   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from statistics import mean, pstdev

import numpy as np

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


def run_warmup_jfi_once(
    *,
    seed: int,
    day_folders: list[str],
    data_root: str,
    warmup_seconds: float,
    episode_duration_seconds: float,
    dataset: str = "didi",
) -> tuple[float, str, float, str | None]:
    """
    Mirrors rl/gym_environment.py reset() warmup only (no composite / RL phase).
    Returns (jfi_after_warmup, selected_day, start_time, error_or_none).
    """
    random.seed(seed)
    np.random.seed(seed)

    selected_day = random.choice(day_folders)
    day_path = os.path.join(data_root, selected_day)

    try:
        workers, tasks = load_workers_tasks(dataset, root_path=day_path)
    except Exception as e:
        return float("nan"), selected_day, float("nan"), f"load: {e}"

    if not tasks:
        return float("nan"), selected_day, float("nan"), "no tasks"

    earliest = min(t.release_time for t in tasks)
    latest = max(t.release_time for t in tasks)
    # Match gym reset(): random drop-in must leave room for warmup + RL phase after start.
    total_needed = warmup_seconds + episode_duration_seconds
    max_start = latest - total_needed

    if max_start < earliest:
        start_time = float(earliest)
    else:
        start_time = float(random.uniform(earliest, max_start))

    warmup_config = get_simulation_config()
    warmup_config["assignment_strategy"] = "greedy"
    warmup_config["strategy_params"] = {"enable_deferral_tracking": False}

    try:
        sim = EventSimulator(workers, tasks, sim_config=warmup_config)
        sim.reset(start_time=start_time)
        sim.step(duration_seconds=warmup_seconds)
        jfi = float(sim.metrics.current_step_stats.get("jfi", 0.5))
        return jfi, selected_day, start_time, None
    except Exception as e:
        return float("nan"), selected_day, start_time, f"sim: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample post-warmup greedy JFI (same protocol as gym reset warmup)"
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
        help="Directory containing one folder per day",
    )
    parser.add_argument("--n-runs", type=int, default=30, help="Number of random warmup samples")
    parser.add_argument("--base-seed", type=int, default=42, help="Seeds used: base, base+1, ...")
    parser.add_argument(
        "--warmup-minutes",
        type=float,
        default=30.0,
        help="Greedy warmup duration (minutes), default matches gym",
    )
    parser.add_argument(
        "--episode-duration-hours",
        type=float,
        default=8.0,
        help="RL phase length used only for feasible drop-in window (must match training env)",
    )
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Override DATA_SAMPLING.use_stratified_sampling (true/false); omit = use config.py",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Use at most this many day folders (after sort), for testing",
    )
    parser.add_argument(
        "--output-prefix",
        default="warmup_jfi_baseline_sample",
        help="Writes <prefix>_<timestamp>.txt and .csv",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root.lstrip("./"))

    day_folders = discover_day_folders(data_root)
    if args.max_days is not None:
        day_folders = day_folders[: args.max_days]

    if not day_folders:
        print(f"No day folders under {data_root}")
        return 1

    warmup_sec = args.warmup_minutes * 60.0
    episode_sec = args.episode_duration_hours * 3600.0
    orig_strat = config_module.DATA_SAMPLING["use_stratified_sampling"]
    if args.stratified is not None:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified

    rows: list[dict] = []
    t0 = time.time()
    try:
        for i in range(args.n_runs):
            seed = args.base_seed + i
            jfi, day, st, err = run_warmup_jfi_once(
                seed=seed,
                day_folders=day_folders,
                data_root=data_root,
                warmup_seconds=warmup_sec,
                episode_duration_seconds=episode_sec,
            )
            rows.append(
                {
                    "run_index": i + 1,
                    "seed": seed,
                    "day_folder": day,
                    "start_time": st,
                    "warmup_jfi": jfi,
                    "error": err or "",
                }
            )
            status = err or "ok"
            print(f"[{i + 1}/{args.n_runs}] seed={seed} day={day} warmup_jfi={jfi:.6f} ({status})")
    finally:
        config_module.DATA_SAMPLING["use_stratified_sampling"] = orig_strat

    ok_jfis = [
        r["warmup_jfi"]
        for r in rows
        if r["error"] == "" and not math.isnan(r["warmup_jfi"])
    ]
    elapsed = time.time() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}.txt")
    csv_path = os.path.join(PROJECT_ROOT, f"{args.output_prefix}_{ts}.csv")

    lines = [
        "WARMUP JFI BASELINE SAMPLE (greedy only, post-warmup snapshot — matches gym greedy_baseline_jfi input)",
        "=" * 88,
        f"data_root: {data_root}",
        f"day_folders used: {len(day_folders)}",
        f"n_runs: {args.n_runs}  base_seed: {args.base_seed}  warmup_minutes: {args.warmup_minutes}  "
        f"episode_duration_hours (for drop-in window): {args.episode_duration_hours}",
        f"stratified override: {args.stratified if args.stratified is not None else '(config.py)'}",
        f"successful_runs: {len(ok_jfis)} / {args.n_runs}",
        f"elapsed_sec: {elapsed:.1f}",
        "=" * 88,
    ]
    if ok_jfis:
        lines.extend(
            [
                "",
                f"mean JFI: {mean(ok_jfis):.6f}",
                f"std JFI:  {pstdev(ok_jfis) if len(ok_jfis) > 1 else 0.0:.6f}",
                f"min JFI:  {min(ok_jfis):.6f}",
                f"max JFI:  {max(ok_jfis):.6f}",
            ]
        )
    lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    print()
    if ok_jfis:
        print(
            f"Summary: mean={mean(ok_jfis):.6f}  std={pstdev(ok_jfis) if len(ok_jfis) > 1 else 0.0:.6f}  "
            f"min={min(ok_jfis):.6f}  max={max(ok_jfis):.6f}"
        )
    print(f"Wrote: {txt_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
