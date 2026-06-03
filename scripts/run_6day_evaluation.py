#!/usr/bin/env python3
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

# Evaluation runs the FULL day (queue drain), not the 8h PPO training window.
# This makes TAR a real completion rate comparable to the conference-ready 6-day eval.
EVAL_EPISODE_HOURS = 48.0

STATIC_COMPOSITE_PARAMS = {
    'fairness_weight': 1.0,
    'starvation_weight': 0.2,
    'utility_weight': 1.0,
    'gamma': 0.1,
    'k': 15,
    'soft_threshold': 0.05,
    'enable_diagnostics': True,
    'enable_deferral_tracking': False,
}

def _enable_tier1_fairness_diagnostics(simulator) -> None:
    simulator.metrics.enable_tier1_fairness_diagnostics()

def extract_metrics(stats):
    """Pulls efficiency + Tier-1 fairness metrics for LaTeX tables."""
    wait_times = stats.get('wait_times', [])
    p95_wait = float(np.percentile(wait_times, 95)) if wait_times else 0.0
    
    return {
        'TAR': stats.get('task_assignment_ratio', 0.0),
        'TAR_released': stats.get('completion_rate_released', 0.0),
        'Tasks Completed': stats.get('completed_tasks', 0),
        'Tasks Released': stats.get('tasks_released', 0),
        'Never Released': stats.get('tasks_never_released', 0),
        'JFI': stats.get('final_jains_fairness_index', 0.0),
        'JFI_rate': stats.get('final_jfi_rate', 0.0),
        'JFI_opportunity': stats.get('final_jfi_opportunity', np.nan),
        'Gini': stats.get('final_gini_coefficient', 0.0),
        'Gini_rate': stats.get('final_gini_rate', 0.0),
        'Gini_opportunity': stats.get('final_gini_opportunity', np.nan),
        'Mean Wait (m)': stats.get('avg_wait_time_minutes', 0.0),
        'P95 Wait (m)': p95_wait,
        'Peak Backlog': stats.get('backlog_peak', 0),
        'Avg Pickup (km)': stats.get('avg_pickup_distance_km', 0.0)
    }

def _run_simulator_episode(env, strategy: str, strategy_params=None):
    """
    Post-warmup phase via EventSimulator directly (no oracle twin).

    Required for pure greedy on oracle-approach: gym env.step() always runs
    greedy-twin + composite, even after switch_strategy('greedy').
    """
    if strategy_params is None:
        strategy_params = {}
    env.simulator.switch_strategy(strategy, strategy_params)
    done = False
    while not done:
        sim_done = env.simulator.step(duration_seconds=env.step_duration)
        env.current_step_idx += 1
        if env.episode_end_time and env.simulator.current_time >= env.episode_end_time:
            done = True
        elif sim_done:
            done = True
    return env.simulator.get_final_results()


def _day_short(day: str) -> str:
    return day.split("_")[-1]


def run_heuristic(day, strategy_name, params):
    """Runs a non-RL strategy using the same warmup protocol as RL eval."""
    env = AdaptiveSpatialCrowdsourcingEnv(
        data_root=DATA_ROOT, day_folders=[day], episode_duration_hours=EVAL_EPISODE_HOURS
    )
    env.reset(seed=42)
    _enable_tier1_fairness_diagnostics(env.simulator)

    if strategy_name == "greedy":
        stats = _run_simulator_episode(env, "greedy", {})
        return extract_metrics(stats)

    if strategy_name == "fatp_ann":
        from simulator.strategies.fatp_ann import FairnessCapTracker
        tracker = FairnessCapTracker()
        tracker.initialize(env.simulator.state.all_workers_map.values())
        params = dict(params)
        params["fairness_cap_tracker"] = tracker

    if strategy_name == "composite":
        # Match compare_model_to_baseline static protocol (fixed λ each step).
        sp = dict(params)
        sp["normalize_scores"] = True
        env.simulator.switch_strategy("composite", sp)
        action = np.array(
            [float(sp["fairness_weight"]), float(sp["starvation_weight"])],
            dtype=np.float32,
        )
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        return extract_metrics(env.simulator.get_final_results())

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
    
    env = AdaptiveSpatialCrowdsourcingEnv(
        data_root=DATA_ROOT, day_folders=[day], episode_duration_hours=EVAL_EPISODE_HOURS
    )
    obs, _ = env.reset(seed=42)
    _enable_tier1_fairness_diagnostics(env.simulator)
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(np.ravel(action))
        done = terminated or truncated
        
    return extract_metrics(env.simulator.get_final_results())

def main():
    print(f"🚀 Starting 6-Day Evaluation Sweep on {len(EVAL_DAYS)} days...")
    
    # Oracle Δ count-JFI run (62k steps, resumed from run_20260522)
    RL_BEST = "rl_logs_sb3/run_20260529_092006/best_model/best_model.zip"
    RL_FINAL = "rl_logs_sb3/run_20260529_092006/ppo_sc_final.zip"
    out_csv = os.path.join(
        PROJECT_ROOT, "rl_logs_sb3", "run_20260529_092006", "final_6day_eval_results.csv"
    )
    per_day_csv = out_csv.replace(".csv", "_per_day.csv")
    
    # Focused sweep: greedy + static composite + RL only.
    # Re-enable Random / FATP-ANN / LAF for full paper baseline table.
    strategies = {
        "Greedy": ("greedy", {}),
        # "Random": ("random_assign", {}),
        # "LAF": ("laf", {}),
        # "FATP-ANN": ("fatp_ann", {'k': 15}),
        "Static-Composite": ("composite", STATIC_COMPOSITE_PARAMS),
        "RL-Best": ("rl", RL_BEST),
        "RL-Final": ("rl", RL_FINAL),
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
                
                print(
                    f"  ✔️ {name:<16} | Wait: {metrics['Mean Wait (m)']:.2f}m | "
                    f"JFI: {metrics['JFI']:.4f} | JFI_rate: {metrics['JFI_rate']:.4f} | "
                    f"Time: {time.time()-t0:.1f}s"
                )
            except Exception as e:
                print(f"  ❌ {name:<16} FAILED: {str(e)}")

    # Aggregate and Print Final Table
    print("\n" + "="*80)
    print("🏆 FINAL 6-DAY AVERAGE RESULTS (Ready for LaTeX)")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    metric_cols = [
        'TAR', 'TAR_released', 'Tasks Completed', 'Tasks Released', 'Never Released',
        'JFI', 'JFI_rate', 'JFI_opportunity',
        'Gini', 'Gini_rate', 'Gini_opportunity',
        'Mean Wait (m)', 'P95 Wait (m)', 'Peak Backlog', 'Avg Pickup (km)',
    ]

    # Per-day rows (all strategies × all days)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df[['Day', 'Strategy'] + metric_cols].round(4).to_csv(per_day_csv, index=False)
    print(f"\n✅ Per-day rows saved to '{per_day_csv}'")

    # Per-day console breakdown (key strategies)
    print("\n" + "=" * 80)
    print("📅 PER-DAY BREAKDOWN (Tasks Completed | Mean Wait | JFI | JFI_rate | TAR_released)")
    print("=" * 80)
    key_strats = [s for s in ['Greedy', 'Static-Composite', 'RL-Best', 'RL-Final'] if s in df['Strategy'].unique()]
    for day in EVAL_DAYS:
        print(f"\n--- {_day_short(day)} ---")
        sub = (
            df[df['Day'] == day]
            .set_index('Strategy')
            .reindex(key_strats)[['Tasks Completed', 'Mean Wait (m)', 'JFI', 'JFI_rate', 'TAR_released']]
            .round(4)
        )
        print(sub.to_markdown())

    # Group by Strategy and calculate the mean for the 6 days
    summary = df.groupby('Strategy').mean(numeric_only=True).round(4)
    
    # Reorder columns to match your LaTeX narrative
    cols = metric_cols
    summary = summary[cols]
    
    # Sort for visual progression from greedy -> fairness -> balanced
    sort_order = ['Greedy', 'Static-Composite', 'RL-Best', 'RL-Final']
    # Full paper order: ['Greedy', 'Random', 'Static-Composite', 'RL-Best', 'RL-Final', 'FATP-ANN', 'LAF']
    summary = summary.reindex([x for x in sort_order if x in summary.index])

    # 1. SAVE FIRST (The Safety Net)
    summary.to_csv(out_csv)
    print(f"\n✅ 6-day means saved to '{out_csv}'")

    # 2. PRINT SECOND
    print(summary.to_markdown())

if __name__ == "__main__":
    main()