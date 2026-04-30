"""
Calibrate SLA_WAIT_TIME for the piecewise reward using the static baseline.

Runs the same protocol as compare_model_to_baseline.run_static_baseline (greedy warmup →
composite with config λ). After each env step, records stats['latency'] from
get_reward_stats — the same quantity used in AdaptiveSpatialCrowdsourcingEnv._calculate_reward.

Usage (repo root):
    python3 scripts/calibrate_sla.py
    python3 scripts/calibrate_sla.py --day YOUR_DAY --eval-seed 42 --percentile 75

Then set the printed suggested value as the default sla_wait_time_minutes in
rl/gym_environment.py (or pass it when constructing the env).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import get_strategy_params
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv


def collect_baseline_step_latencies(
    day: str,
    data_root: str,
    eval_seed: int,
) -> tuple[List[float], dict]:
    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=data_root, day_folders=[day])
    _, _ = env.reset(seed=eval_seed)

    sp = get_strategy_params("composite")
    sp["normalize_scores"] = True
    sp["enable_deferral_tracking"] = True
    env.simulator.switch_strategy("composite", sp)

    action = np.array(
        [float(sp["fairness_weight"]), float(sp["starvation_weight"])],
        dtype=np.float32,
    )

    latencies: List[float] = []
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(action)
        stats = env.simulator.metrics.get_reward_stats(env.simulator.current_time)
        latencies.append(float(stats["latency"]))
        done = terminated or truncated

    final = env.simulator.get_final_results()
    meta = {
        "day": day,
        "eval_seed": eval_seed,
        "data_root": data_root,
        "lambda1": float(sp["fairness_weight"]),
        "lambda2": float(sp["starvation_weight"]),
        "lambda3": float(sp.get("utility_weight", 1.0)),
        "steps": len(latencies),
        "episode_avg_wait_minutes": float(final.get("avg_wait_time_minutes", 0.0)),
    }
    return latencies, meta


def main() -> None:
    default_data = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
    parser = argparse.ArgumentParser(
        description="Per-step latency distribution under static baseline (for SLA calibration)."
    )
    parser.add_argument(
        "--day",
        type=str,
        default="496528674@qq.com_20161128",
        help="Day folder name (must exist under --data-root)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data,
        help="Path to DiDi day folders",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="RNG seed (match train_sb3 / compare_model_to_baseline)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=75.0,
        help="Percentile for suggested SLA (e.g. 75 or 90)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write latencies + summary JSON (project-root relative OK)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(PROJECT_ROOT, data_root.lstrip("./"))

    t0 = time.time()
    print(f"SLA calibration: static baseline, day={args.day}, eval_seed={args.eval_seed}")
    latencies, meta = collect_baseline_step_latencies(
        args.day, data_root, args.eval_seed
    )
    elapsed = time.time() - t0

    arr = np.asarray(latencies, dtype=np.float64)
    p = float(np.percentile(arr, args.percentile))
    summary = {
        **meta,
        "latencies_minutes": latencies,
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "suggested_sla_minutes": p,
        "suggested_percentile": args.percentile,
    }

    print(f"\nDone in {elapsed:.2f}s, {len(latencies)} steps.")
    print("--- Step latency (minutes), same signal as reward ---")
    print(f"  mean:   {summary['mean']:.4f}")
    print(f"  p50:    {summary['p50']:.4f}")
    print(f"  p75:    {summary['p75']:.4f}")
    print(f"  p90:    {summary['p90']:.4f}")
    print(f"  max:    {summary['max']:.4f}")
    print(f"  episode avg wait (final results): {meta['episode_avg_wait_minutes']:.4f}")
    print(
        f"\nSuggested SLA (p{args.percentile:g}): {p:.4f} minutes\n"
        f"  → Set sla_wait_time_minutes={p:.4f} in rl/gym_environment.py __init__ default,\n"
        f"    or pass sla_wait_time_minutes when building the env."
    )

    if args.json_out:
        out = args.json_out
        if not os.path.isabs(out):
            out = os.path.join(PROJECT_ROOT, out.lstrip("./"))
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
