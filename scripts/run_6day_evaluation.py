#!/usr/bin/env python3
"""
Extended 6-day evaluation of an RL model across multiple held-out test days.

After `rl/train_sb3.py` completes, `compare_model_to_baseline.py` runs a single-day
baseline comparison (e.g., JFI, peak backlog, avg wait). That result goes in
`baseline_best_model_metrics.txt` and `eval_weights_best_steps.txt` — useful for
a quick sanity check, but only one day's data.

This script extends the evaluation across 6 held-out days to confirm the policy
generalizes and doesn't overfit to the training/validation set. It compares:
  - Static composite (fixed λ1=1.0, λ2=0.2) across all 6 days
  - RL agent (policy-chosen λ each step) across all 6 days

Output: A CSV with aggregated metrics across days, confirming whether the learned
policy beats or matches the static baseline on multiple test scenarios.

Usage:
  python scripts/run_6day_evaluation.py rl_logs_sb3/run_YYYYMMDD_HHMMSS/ppo_sc_final
  python scripts/run_6day_evaluation.py rl_logs_sb3/run_YYYYMMDD_HHMMSS/best_model/best_model

Results are saved to: run_YYYYMMDD_HHMMSS/6day_eval_results.csv
"""
import sys
import os
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv

EVAL_DAYS = [
    "496528674@qq.com_20161109",
    "496528674@qq.com_20161110",
    "496528674@qq.com_20161116",
    "496528674@qq.com_20161118",
    "496528674@qq.com_20161124",
    "496528674@qq.com_20161128"
]

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")

STATIC_COMPOSITE_PARAMS = {
    'fairness_weight': 1.0,
    'starvation_weight': 0.2,
    'utility_weight': 1.0,
    'gamma': 0.1,
    'k': 15,
    'soft_threshold': 0.05,
    'enable_diagnostics': False,
    'enable_deferral_tracking': False,
}

def extract_metrics(stats):
    """Pulls exactly the metrics defined in your LaTeX document."""
    wait_times = stats.get('wait_times', [])
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0
    
    return {
        'TAR': stats.get('task_assignment_ratio', 0.0),
        'JFI': stats.get('final_jains_fairness_index', 0.0),
        'Gini': stats.get('final_gini_coefficient', 0.0),
        'Mean Wait (m)': stats.get('avg_wait_time_minutes', 0.0),
        'P95 Wait (m)': p95_wait,
        'Peak Backlog': stats.get('backlog_peak', 0),
        'Avg Pickup (km)': stats.get('avg_pickup_distance_km', 0.0)
    }

def run_heuristic(day, strategy_name, params):
    """Runs a non-RL strategy using the exact same environment and warmup."""
    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[day])
    env.reset(seed=42)
    
    # Safe injection for FATP-ANN state requirements
    if strategy_name == "fatp_ann":
        from simulator.strategies.fatp_ann import FairnessCapTracker
        tracker = FairnessCapTracker()
        tracker.initialize(env.simulator.state.all_workers_map.values())
        params['fairness_cap_tracker'] = tracker

    env.simulator.switch_strategy(strategy_name, params)
    
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated
        
    return extract_metrics(env.simulator.get_final_results())

def run_rl(day, model_path):
    """Runs the trained PPO model."""
    load_path = model_path[:-4] if model_path.endswith(".zip") else model_path
    model = PPO.load(load_path)
    
    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[day])
    obs, _ = env.reset(seed=42)
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(np.ravel(action))
        done = terminated or truncated
        
    return extract_metrics(env.simulator.get_final_results())

def main():
    print(f"🚀 Starting 6-Day Evaluation Sweep on {len(EVAL_DAYS)} days...")
    
    # The models to test
    RL_BEST = "rl_logs_sb3/run_20260513_071355/best_model/best_model.zip"
    RL_FINAL = "rl_logs_sb3/run_20260513_071355/ppo_sc_final.zip"
    
    strategies = {
        "Greedy": ("greedy", {}),
        "Random": ("random_assign", {}),
        "LAF": ("laf", {}),
        "FATP-ANN": ("fatp_ann", {'k': 15}),
        "Static-Composite": ("composite", STATIC_COMPOSITE_PARAMS),
        "RL-Best": ("rl", RL_BEST),
        "RL-Final": ("rl", RL_FINAL)
    }
    
    all_results = []
    
    for day in EVAL_DAYS:
        print(f"\n📅 Evaluating Day: {day}")
        for name, (strat_type, config) in strategies.items():
            t0 = time.time()
            try:
                if strat_type == "rl":
                    metrics = run_rl(day, os.path.join(PROJECT_ROOT, config))
                else:
                    metrics = run_heuristic(day, strat_type, config)
                
                metrics['Day'] = day
                metrics['Strategy'] = name
                all_results.append(metrics)
                
                print(f"  ✔️ {name:<16} | Wait: {metrics['Mean Wait (m)']:.2f}m | JFI: {metrics['JFI']:.4f} | Time: {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  ❌ {name:<16} FAILED: {str(e)}")

    # Aggregate and Print Final Table
    print("\n" + "="*80)
    print("🏆 FINAL 6-DAY AVERAGE RESULTS (Ready for LaTeX)")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    # Group by Strategy and calculate the mean for the 6 days
    summary = df.groupby('Strategy').mean(numeric_only=True).round(4)
    
    # Reorder columns to match your LaTeX narrative
    cols = ['TAR', 'JFI', 'Gini', 'Mean Wait (m)', 'P95 Wait (m)', 'Peak Backlog', 'Avg Pickup (km)']
    summary = summary[cols]
    
    # Sort for visual progression from greedy -> fairness -> balanced
    sort_order = ['Greedy', 'Random', 'Static-Composite', 'RL-Best', 'RL-Final', 'FATP-ANN', 'LAF']
    summary = summary.reindex([x for x in sort_order if x in summary.index])

    # 1. SAVE FIRST (The Safety Net)
    summary.to_csv("final_6day_eval_results.csv")
    print("\n✅ Saved to 'final_6day_eval_results.csv'")

    # 2. PRINT SECOND
    print(summary.to_markdown())

if __name__ == "__main__":
    main()