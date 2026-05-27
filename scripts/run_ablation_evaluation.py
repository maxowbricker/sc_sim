#!/usr/bin/env python3
"""
Sec 5.4 — Ablation Study: Composite Strategy Design Choices

Runs three ablation variants of the composite strategy across all 6 eval days
and reports 6-day averages.  Results are saved to a timestamped CSV so they
never overwrite the main 6-day evaluation results.

Ablation variants:
  Ablation: No Brain     — EWMA fairness, standard weights (λ₁=1.0, λ₂=0.2)
  Ablation: No Structure — Uniform weights (λ₁=0.6, λ₂=0.6, equal emphasis)
  Ablation: No Metric    — Global marginal JFI instead of EWMA, same weights
  Proposed DRL Governor  — Best trained RL model (for comparison)

Usage (from project root):
    python scripts/run_ablation_evaluation.py
"""

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from stable_baselines3 import PPO
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

EVAL_DAYS = [
    "496528674@qq.com_20161109",
    "496528674@qq.com_20161110",
    "496528674@qq.com_20161116",
    "496528674@qq.com_20161118",
    "496528674@qq.com_20161124",
    "496528674@qq.com_20161128",
]

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")

RL_BEST = os.path.join(
    PROJECT_ROOT,
    "rl_logs_sb3", "run_20260513_071355", "best_model", "best_model",
)

# ── Ablation strategy definitions ─────────────────────────────────────────────
STRATEGIES = {
    "Ablation: No Brain": ("composite", {
        "fairness_weight":    1.0,
        "starvation_weight":  0.2,
        "utility_weight":     1.0,
        "gamma":              0.1,
        "k":                  15,
        "soft_threshold":     0.05,
        "use_global_jfi":     False,   # proposed EWMA metric, standard weights
    }),
    "Ablation: No Structure": ("composite", {
        "fairness_weight":    0.6,
        "starvation_weight":  0.6,     # equal emphasis — no asymmetric structure
        "utility_weight":     1.0,
        "gamma":              0.1,
        "k":                  15,
        "soft_threshold":     0.05,
        "use_global_jfi":     False,
    }),
    "Ablation: No Metric": ("composite", {
        "fairness_weight":    1.0,
        "starvation_weight":  0.2,
        "utility_weight":     1.0,
        "gamma":              0.1,
        "k":                  15,
        "soft_threshold":     0.05,
        "use_global_jfi":     True,    # global marginal JFI instead of EWMA
    }),
    "Proposed DRL Governor": ("rl", RL_BEST),
}


def extract_metrics(stats: dict) -> dict:
    wait_times = stats.get("wait_times", [])
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0
    return {
        "TAR":            stats.get("task_assignment_ratio", 0.0),
        "JFI":            stats.get("final_jains_fairness_index", 0.0),
        "Gini":           stats.get("final_gini_coefficient", 0.0),
        "Mean Wait (m)":  stats.get("avg_wait_time_minutes", 0.0),
        "P95 Wait (m)":   p95_wait,
        "Peak Backlog":   stats.get("backlog_peak", 0),
        "Avg Pickup (km)":stats.get("avg_pickup_distance_km", 0.0),
    }


def run_heuristic(day: str, strategy_name: str, params: dict) -> dict:
    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[day])
    env.reset(seed=42)
    env.simulator.switch_strategy(strategy_name, params)

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated

    return extract_metrics(env.simulator.get_final_results())


def run_rl(day: str, model: PPO) -> dict:
    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[day])
    obs, _ = env.reset(seed=42)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(np.ravel(action))
        done = terminated or truncated

    return extract_metrics(env.simulator.get_final_results())


def main():
    print(f"🔬 Ablation Study — {len(EVAL_DAYS)} eval days × {len(STRATEGIES)} strategies")

    print("🔄 Loading RL model …")
    rl_model = PPO.load(RL_BEST)

    all_results = []

    for day in EVAL_DAYS:
        print(f"\n📅 {day}")
        for name, (strat_type, config) in STRATEGIES.items():
            t0 = time.time()
            try:
                if strat_type == "rl":
                    metrics = run_rl(day, rl_model)
                else:
                    metrics = run_heuristic(day, strat_type, config)

                metrics["Day"] = day
                metrics["Strategy"] = name
                all_results.append(metrics)

                print(f"  ✔️ {name:<28} | Wait: {metrics['Mean Wait (m)']:.2f}m "
                      f"| JFI: {metrics['JFI']:.4f} | [{time.time()-t0:.1f}s]")
            except Exception as exc:
                print(f"  ❌ {name:<28} FAILED: {exc}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🔬 ABLATION STUDY — 6-DAY AVERAGES")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    summary = df.groupby("Strategy").mean(numeric_only=True).round(4)

    cols = ["TAR", "JFI", "Gini", "Mean Wait (m)", "P95 Wait (m)", "Peak Backlog", "Avg Pickup (km)"]
    summary = summary[cols]

    sort_order = ["Ablation: No Brain", "Ablation: No Structure",
                  "Ablation: No Metric", "Proposed DRL Governor"]
    summary = summary.reindex([x for x in sort_order if x in summary.index])

    # Timestamped filename — never overwrites previous runs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"ablation_results_{ts}.csv")
    summary.to_csv(csv_path)
    print(f"\n✅ Saved to '{csv_path}'")

    print(summary.to_markdown())


if __name__ == "__main__":
    main()
