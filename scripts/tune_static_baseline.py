"""
Optuna script to find the absolute best static weights for the Composite strategy.
Runs on a full day to ensure the baseline is highly competitive.
"""
import optuna
import sys
import os
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks

# --- CONFIGURATION ---
# We use the exact same evaluation day to keep the baseline perfectly aligned
# 20161130 = Wednesday Nov 30, 2016 (original 20161128 was Monday)
TEST_DAY = "496528674@qq.com_20161130"  
DATA_ROOT = "./data/didi/full_didi_gaia"

print(f"==================================================")
print(f"💿 LOADING DATA TO RAM FOR: {TEST_DAY}")
print(f"==================================================")
# Load once outside the loop to save hours of disk I/O time
day_path = os.path.join(DATA_ROOT, TEST_DAY)
base_config = get_simulation_config()
WORKERS, TASKS = load_workers_tasks(dataset=base_config['dataset'], root_path=day_path)
print(f"✅ Loaded {len(WORKERS):,} workers and {len(TASKS):,} tasks into memory.\n")

def objective(trial):
    start_time = time.time()
    
    # 1. Let Optuna suggest the parameters
    w_fair = trial.suggest_float("fairness_weight", 0.0, 5.0)
    w_starv = trial.suggest_float("starvation_weight", 0.0, 2.0)
    soft_thresh = trial.suggest_float("soft_threshold", 0.0, 1.0)
    
    # 2. Build the config
    sim_config = get_simulation_config()
    sim_config['assignment_strategy'] = 'composite'
    
    params = get_strategy_params('composite')
    params['fairness_weight'] = w_fair
    params['starvation_weight'] = w_starv
    params['soft_threshold'] = soft_thresh
    params['utility_weight'] = 1.0  # Fixed unit anchor
    sim_config['strategy_params'] = params
    
    # 3. Run Simulation (Using the pre-loaded RAM objects)
    sim = EventSimulator(WORKERS, TASKS, sim_config)
    sim.reset()
    sim.step()
    
    # 4. Get Results
    stats = sim.get_final_results()
    jfi = stats.get('final_jains_fairness_index', 0.0)
    backlog = stats.get('backlog_peak', 0)
    wait = stats.get('avg_wait_time_minutes', 0.0)
    
    # 5. Calculate Score (Using the EXACT same rubric as the RL agent!)
    score_fairness = (jfi - 0.5) * 10.0
    if jfi < 0.75:
        score_fairness -= 20.0  # The JFI Cliff
        
    score_throughput = -backlog / 100.0
    score_latency = -wait / 5.0
    
    total_score = score_fairness + score_throughput + score_latency
    
    print(f"Trial {trial.number} finished in {time.time() - start_time:.1f}s | "
          f"JFI: {jfi:.3f}, Backlog: {backlog}, Wait: {wait:.2f}m | Score: {total_score:.2f}")
    print(f"    (λ1: {w_fair:.2f}, λ2: {w_starv:.2f}, Thresh: {soft_thresh:.2f})")
    
    return total_score

if __name__ == "__main__":
    print("🚀 Starting Optuna Study for Static Baseline...")
    # TPE algorithm will maximize the score
    study = optuna.create_study(direction="maximize")
    
    # 50 trials on a full day will likely take a few hours. 
    study.optimize(objective, n_trials=50)

    print("\n==================================================")
    print("🏆 BEST STATIC BASELINE PARAMETERS FOUND 🏆")
    print("==================================================")
    trial = study.best_trial
    print(f"Highest Score: {trial.value:.2f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value:.4f}")