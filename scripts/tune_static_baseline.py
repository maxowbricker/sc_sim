"""
Optuna script to find the absolute best static weights for the Composite strategy.
Runs on a full day to ensure the baseline is highly competitive.
"""
import optuna
import sys
import os
import time
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks

# --- CONFIGURATION ---
TEST_DAY = "496528674@qq.com_20161130"  # Wednesday
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
day_path = os.path.join(DATA_ROOT, TEST_DAY)

if not os.path.exists(day_path):
    print(f"❌ Data not found: {day_path}")
    print(f"   Expected: {day_path}/gps.txt and {day_path}/order.txt")
    print(f"   Download Didi GAIA Chengdu data and place day folders under: {DATA_ROOT}")
    sys.exit(1)

print(f"==================================================")
print(f"💿 LOADING DATA TO RAM FOR: {TEST_DAY}")
print(f"==================================================")
base_config = get_simulation_config()
WORKERS, TASKS = load_workers_tasks(dataset=base_config['dataset'], root_path=day_path)
print(f"✅ Loaded {len(WORKERS):,} workers and {len(TASKS):,} tasks into memory.\n")

print(f"==================================================")
print(f"🎯 CALCULATING DYNAMIC GREEDY BASELINE...")
print(f"==================================================")
greedy_config = get_simulation_config()
greedy_config['assignment_strategy'] = 'greedy'
sim_greedy = EventSimulator(WORKERS, TASKS, greedy_config)
sim_greedy.reset()
sim_greedy.step() # Run full day
greedy_stats = sim_greedy.get_final_results()
GREEDY_BASELINE_JFI = greedy_stats.get('final_jains_fairness_index', 0.5)
print(f"🎯 Target Baseline JFI Locked In: {GREEDY_BASELINE_JFI:.4f}\n")


def objective(trial):
    start_time = time.time()
    
    # 1. Let Optuna suggest the parameters
    w_fair = trial.suggest_float("fairness_weight", 0.0, 2.0, step=0.05)
    w_starv = trial.suggest_float("starvation_weight", 0.0, 0.5, step=0.05)
    soft_thresh = trial.suggest_float("soft_threshold", 0.0, 1.0, step=0.05)
    
    # 2. Build the config
    sim_config = get_simulation_config()
    sim_config['assignment_strategy'] = 'composite'
    
    params = get_strategy_params('composite')
    params['fairness_weight'] = w_fair
    params['starvation_weight'] = w_starv
    params['soft_threshold'] = soft_thresh
    params['utility_weight'] = 1.0  # Fixed unit anchor
    params['k'] = 15               # Lock in the high-performance neighborhood size
    sim_config['strategy_params'] = params
    
    # 3. Run Simulation
    sim = EventSimulator(WORKERS, TASKS, sim_config)
    sim.reset()
    sim.step()
    
    # 4. Get Results
    stats = sim.get_final_results()
    jfi = stats.get('final_jains_fairness_index', 0.0)
    wait = stats.get('avg_wait_time_minutes', 0.0)
    # 5. Calculate Score (Apples-to-Apples with RL Environment)
    
    # Pillar 1: Dynamic Fairness Anchor
    jfi_improvement = jfi - GREEDY_BASELINE_JFI
    if jfi_improvement >= 0:
        score_fairness = jfi_improvement * 20.0
    else:
        score_fairness = -10.0 * (np.exp(abs(jfi_improvement) * 5.0) - 1.0)
        
    # Pillar 2: Efficiency Anchor
    score_latency = -wait / 5.0
    
    # Pillar 3: Expirations Anchor (Relaxed & Capped for the full day)
    # We cap at -10.0 so that even a "bad" day doesn't skew the Optuna results
    # to only care about starvation.
    total_expirations = len(stats.get('expired_tasks', []))
    score_starvation = -min(10.0, total_expirations / 100.0)
    
    total_score = score_fairness + score_latency + score_starvation
    
    print(f"Trial {trial.number} in {time.time() - start_time:.1f}s | "
          f"JFI: {jfi:.3f} (Δ {jfi_improvement:+.3f}), Wait: {wait:.2f}m, Died: {total_expirations} | "
          f"Score: {total_score:.2f}")
    
    return total_score

if __name__ == "__main__":
    print("🚀 Starting Optuna Study for Static Baseline...")
    study = optuna.create_study(direction="maximize")
    
    # You can set this to 30-50 for a solid local run
    study.optimize(objective, n_trials=40)

    print("\n==================================================")
    print("🏆 BEST STATIC BASELINE PARAMETERS FOUND 🏆")
    print("==================================================")
    trial = study.best_trial
    print(f"Highest Score: {trial.value:.2f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value:.4f}")