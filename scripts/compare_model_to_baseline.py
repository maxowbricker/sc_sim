"""
Compare RL Agent vs. Static Baseline (FATP/Composite)
"""
import argparse
import sys
import os
import time
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from simulator.simulation import EventSimulator
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks

def run_static_baseline(day, data_root):
    print(f"\n[1/2] 🏃 Running Static Baseline (Composite Strategy) for {day}...")
    start_time = time.time()

    # Setup Static Config directly from config.py defaults
    config = get_simulation_config()
    config['data_root'] = data_root
    config['assignment_strategy'] = 'composite'
    config['strategy_params'] = get_strategy_params('composite')

    w_fair = config['strategy_params'].get('fairness_weight', 'N/A')
    w_starv = config['strategy_params'].get('starvation_weight', 'N/A')
    w_util = config['strategy_params'].get('utility_weight', 'N/A')
    print(f"      ⚖️  Using static config.py weights: Fairness={w_fair}, Starvation={w_starv}, Utility={w_util}")

    day_path = os.path.join(config['data_root'], day)
    workers, tasks = load_workers_tasks(dataset=config['dataset'], root_path=day_path)
    sim = EventSimulator(workers, tasks, config)
    sim.reset()
    sim.step()

    # NEW: Get the true final results for the whole day
    stats = sim.get_final_results()
    print(f"      ✅ Finished in {time.time() - start_time:.2f} seconds.")
    return stats

def run_rl_agent(model_path, day, data_root, quiet=False):
    print(f"\n[2/2] 🧠 Running RL Agent ({model_path}) for {day}...")
    start_time = time.time()

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=data_root, day_folders=[day])
    # SB3 appends .zip internally; strip it to avoid "model.zip.zip" FileNotFoundError
    load_path = model_path[:-4] if model_path.endswith('.zip') else model_path
    model = PPO.load(load_path, env=env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        a = np.ravel(action)  # Handle (2,) or (1,2) from DummyVecEnv
        if not quiet:
            print(f"      🤖 Agent chose weights: λ1 (Fairness)={a[0]:.2f}, λ2 (Starvation)={a[1]:.2f}")
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

    # NEW: Get the true final results for the whole day
    stats = env.simulator.get_final_results()
    print(f"      ✅ Finished in {time.time() - start_time:.2f} seconds.")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compare RL model against Static Baseline")
    parser.add_argument("--model", type=str, required=True, help="Path to the PPO .zip model")
    parser.add_argument("--day", type=str, default="496528674@qq.com_20161128", help="Test day folder")
    parser.add_argument("--data-root", type=str, default="./data/didi/full_didi_gaia", help="Path to data")
    parser.add_argument("--stratified", type=lambda x: x.lower() == "true", default=None,
                        help="Override stratified sampling: true or false (default: use config.py)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step agent action prints")
    args = parser.parse_args()

    # Override config before any data loading
    if args.stratified is not None:
        import config as config_module
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified
        print(f"   📋 Stratified sampling: {'ON' if args.stratified else 'OFF'} (override)")

    print("="*60)
    print(f"🧪 EVALUATION BENCHMARK: {args.day}")
    print("="*60)

    static_stats = run_static_baseline(args.day, args.data_root)
    rl_stats = run_rl_agent(args.model, args.day, args.data_root, quiet=args.quiet)

    # Use the correct dictionary keys from get_final_results()
    jfi_delta = rl_stats['final_jains_fairness_index'] - static_stats['final_jains_fairness_index']
    backlog_delta = rl_stats['backlog_peak'] - static_stats['backlog_peak']
    wait_delta = rl_stats['avg_wait_time_minutes'] - static_stats['avg_wait_time_minutes']

    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Static Baseline':<15} | {'RL Agent':<15} | {'Improvement'}")
    print("-" * 60)

    jfi_trend = "🟢" if jfi_delta > 0 else "🔴"
    print(f"{'JFI (Fairness)':<20} | {static_stats['final_jains_fairness_index']:<15.4f} | {rl_stats['final_jains_fairness_index']:<15.4f} | {jfi_trend} {jfi_delta:+.4f}")

    backlog_trend = "🟢" if backlog_delta < 0 else "🔴"
    print(f"{'Peak Backlog':<20} | {static_stats['backlog_peak']:<15.0f} | {rl_stats['backlog_peak']:<15.0f} | {backlog_trend} {backlog_delta:+.0f}")

    wait_trend = "🟢" if wait_delta < 0 else "🔴"
    print(f"{'Avg Wait Time (m)':<20} | {static_stats['avg_wait_time_minutes']:<15.2f} | {rl_stats['avg_wait_time_minutes']:<15.2f} | {wait_trend} {wait_delta:+.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
