"""
scripts/tune_physics_full.py
Optimizes Gamma, K, and Soft Threshold on the FULL DATASET.
Uses discrete steps to avoid over-optimizing decimal points.

This script tests physics parameters (gamma, k, soft_threshold) while keeping
lambda weights fixed at known good values from exp_021.

Usage:
    python scripts/tune_physics_full.py [--trials N] [--data-path PATH]
    
Examples:
    # Quick test (10 trials)
    python scripts/tune_physics_full.py --trials 10
    
    # Full tuning run (30 trials, overnight)
    python scripts/tune_physics_full.py --trials 30
"""

import optuna
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.simulation import EventSimulator
from config import get_simulation_config
from data.loader import load_workers_tasks

# --- DEFAULT CONFIGURATION ---
DEFAULT_DATASET_PATH = "./data/didi/full_didi_gaia"
DEFAULT_N_TRIALS = 50  # Maximum 50 full simulations
# Fixed lambda values
FIXED_LAMBDA1 = 4.0  # Fairness weight (fixed)
FIXED_LAMBDA3 = 4.0  # Utility weight (fixed)
# -----------------------------

def objective(trial, all_workers, all_tasks):
    """
    Optuna Objective: Returns a score for a specific set of parameters.
    
    Optimizes: λ2 (starvation), k, soft_threshold, gamma
    Fixed: λ1=4.0, λ3=4.0, normalize_scores=True, consistent dataset
    
    Args:
        trial: Optuna trial object
        all_workers: Pre-loaded list of workers (loaded once in main() to save time)
        all_tasks: Pre-loaded list of tasks (loaded once in main() to save time)
    """
    # 1. PARAMETER SEARCH (Discrete steps to avoid over-optimization)
    
    # Gamma: EWMA smoothing factor (0.1 to 0.9, step 0.05)
    gamma = trial.suggest_float("gamma", 0.1, 0.9, step=0.05)
    
    # K: Number of nearest workers/tasks to consider (5 to 50, step 5)
    k = trial.suggest_int("k", 5, 50, step=5)
    
    # Soft threshold: Minimum score for assignment (0.0 to 1.5, step 0.1)
    # Note: 0.0 means threshold is disabled (code handles this automatically)
    threshold = trial.suggest_float("soft_threshold", 0.0, 1.5, step=0.1)
    
    # Lambda2: Starvation weight (0.0 to 1.0, step 0.1)
    lambda2 = trial.suggest_float("λ2", 0.0, 1.0, step=0.1)
    
    # Fixed lambda values
    lambda1 = FIXED_LAMBDA1  # Fixed at 4.0
    lambda3 = FIXED_LAMBDA3  # Fixed at 4.0
    
    lambda_weights = {"λ1": lambda1, "λ2": lambda2, "λ3": lambda3}
    print(f"\n🔄 Trial {trial.number}: λ=[{lambda1:.1f}, {lambda2:.1f}*, {lambda3:.1f}], Gamma={gamma}, K={k}, Threshold={threshold}")
    print(f"   (* λ2 is being optimized, λ1 and λ3 are fixed at 4.0)")

    # 2. Setup Simulation Config
    # Map λ1, λ2, λ3 to the parameter names expected by composite strategy
    # λ1 = fairness_weight, λ2 = starvation_weight, λ3 = utility_weight
    
    config = get_simulation_config()
    config['assignment_strategy'] = 'composite'
    config['strategy_params'] = {
        # Lambda weights: map to composite strategy parameter names
        'fairness_weight': lambda_weights['λ1'],    # λ1: Weight for fairness (EWMA idle time)
        'starvation_weight': lambda_weights['λ2'],  # λ2: Weight for starvation (task age)
        'utility_weight': lambda_weights['λ3'],      # λ3: Weight for utility (spatial efficiency)
        # Also store as λ1, λ2, λ3 for compatibility with update_weights() method
        'λ1': lambda_weights['λ1'],
        'λ2': lambda_weights['λ2'],
        'λ3': lambda_weights['λ3'],
        # Physics parameters
        'k': k,
        'soft_threshold': threshold,
        'gamma': gamma,  # Now passed via strategy_params (standardized)
        # Performance flags
        'normalize_scores': True,  # Enable normalization (middle path)
        'enable_deferral_tracking': False,  # Disable for speedup
        'enable_diagnostics': False  # Disable for speedup
    }

    # 3. Run FULL Simulation
    # Create fresh simulator with loaded data
    sim = EventSimulator(all_workers, all_tasks, sim_config=config)
    
    # Reset to start of day
    sim.reset()
    
    # Run to completion (None duration = run until all events processed)
    try:
        sim.step(duration_seconds=None)
    except Exception as e:
        print(f"   ❌ Simulation failed: {e}")
        return 0.0  # Return 0 score for failed trials
    
    # 4. Calculate Score
    results = sim.get_final_results()
    
    # Extract metrics (using correct keys from get_final_results)
    jfi = results.get('final_jains_fairness_index', 0.0)
    tar = results.get('task_assignment_ratio', 0.0)  # Task Assignment Ratio (TAR)
    avg_wait = results.get('avg_wait_time_minutes', 0.0)
    completed_tasks = results.get('completed_tasks', 0)
    total_tasks = results.get('total_tasks', 0)
    
    # Store detailed metrics as user attributes for later analysis
    trial.set_user_attr("jfi", jfi)
    trial.set_user_attr("task_assignment_ratio", tar)  # TAR: proportion of tasks assigned
    trial.set_user_attr("avg_wait_minutes", avg_wait)
    trial.set_user_attr("completed_tasks", completed_tasks)  # Number of tasks completed
    trial.set_user_attr("total_tasks", total_tasks)  # Total tasks in simulation
    
    print(f"   ➡️ Result: JFI={jfi:.4f}, TAR={tar:.4f}, Wait={avg_wait:.2f}m, Completed={completed_tasks}/{total_tasks}")

    # Hard Constraint: If wait time > 15 mins, it's a failed config (Score = 0)
    if avg_wait > 15.0:
        print(f"   ⚠️  Rejected: Wait time {avg_wait:.2f}m exceeds 15min threshold")
        return 0.0
        
    # Balanced Objective: 
    # We want to maximize Fairness * Task Assignment Ratio while minimizing wait time
    # Score = JFI × TAR × Wait_Penalty
    # Wait_Penalty: 1.0 at 0min wait, decreases linearly to 0.0 at 15min wait
    # This encourages lower wait times while still rewarding high fairness and completion
    wait_penalty = max(0.0, 1.0 - (avg_wait / 15.0))  # Linear penalty from 0-15 minutes
    score = jfi * tar * wait_penalty
    
    # No pruning - all simulations run to completion for meaningful EWMA signals
    return score

def main():
    parser = argparse.ArgumentParser(description="Tune physics parameters (gamma, k, soft_threshold) on full dataset")
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS,
                       help=f"Number of trials to run (default: {DEFAULT_N_TRIALS})")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATASET_PATH,
                       help=f"Path to dataset (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--output", type=str, default="best_physics_params.json",
                       help="Output file for best parameters (default: best_physics_params.json)")
    parser.add_argument("--study-db", type=str, default=None,
                       help="SQLite database path to save full Optuna study (default: study_physics_tuning.db)")
    parser.add_argument("--export-csv", type=str, default=None,
                       help="CSV file path to export all trial results (default: trial_results.csv)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 PHYSICS PARAMETER TUNING")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Optimizing: λ2 (starvation), k, soft_threshold, gamma")
    print(f"Fixed: λ1={FIXED_LAMBDA1:.1f}, λ3={FIXED_LAMBDA3:.1f}, normalize_scores=True")
    print(f"Max trials: {args.trials}")
    print(f"Output files:")
    print(f"  Best params: {args.output}")
    print(f"  Study DB: {args.study_db or 'study_physics_tuning.db'}")
    print(f"  Trial CSV: {args.export_csv or 'trial_results.csv'}")
    print("=" * 80)
    print()
    
    # Verify dataset path exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Dataset path not found: {args.data_path}")
        print("   Please ensure the full_didi_gaia folder exists.")
        return 1
    
    # Load data ONCE globally to save loading time between trials
    print("⏳ Loading Full Dataset (this happens once)...")
    try:
        # Get available day folders
        day_folders = sorted([d for d in os.listdir(args.data_path) 
                            if os.path.isdir(os.path.join(args.data_path, d))])
        
        if not day_folders:
            print(f"❌ Error: No day folders found in {args.data_path}")
            return 1
        
        # Use first day for tuning (or you could average across multiple days)
        first_day = day_folders[0]
        day_path = os.path.join(args.data_path, first_day)
        
        print(f"   Using day: {first_day}")
        all_workers, all_tasks = load_workers_tasks("didi", root_path=day_path)
        print(f"✅ Loaded {len(all_workers):,} workers and {len(all_tasks):,} tasks")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Determine study storage (SQLite database for persistence)
    study_db_path = args.study_db or "study_physics_tuning.db"
    storage = f"sqlite:///{study_db_path}"
    
    # Create study with persistent storage
    # NOTE: No pruner - we want ALL simulations to run to completion
    # EWMA fairness signal needs full simulation to be meaningful
    study = optuna.create_study(
        direction="maximize",
        study_name="physics_tuning",
        storage=storage,
        load_if_exists=True,  # Resume if study already exists
        pruner=None  # Disable pruning - run all trials to completion
    )
    
    # Create objective function with fixed arguments
    def objective_with_args(trial):
        return objective(trial, all_workers, all_tasks)
    
    try:
        print("\n🔬 Starting optimization...")
        study.optimize(objective_with_args, n_trials=args.trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted by user. Saving current best...")
    except Exception as e:
        print(f"\n❌ Tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show results
    print("\n" + "=" * 80)
    print("🏆 OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    if len(study.trials) == 0:
        print("❌ No trials completed successfully!")
        return 1
    
    print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print()
    
    if study.best_trial:
        print(f"Best Score: {study.best_value:.4f}")
        print("Best Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Add lambda config info to saved params
        best_params = study.best_params.copy()
        best_params['lambda_weights'] = {
            'λ1': FIXED_LAMBDA1,  # Fixed at 4.0
            'λ2': best_params.get('λ2', 0.5),  # Optimized
            'λ3': FIXED_LAMBDA3  # Fixed at 4.0
        }
        # Remove λ2 from top level (it's in lambda_weights now)
        best_params.pop('λ2', None)
        best_params['note'] = f"λ1 and λ3 are fixed at {FIXED_LAMBDA1:.1f}, normalize_scores=True, consistent dataset"
        
        # Save to file
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"\n✅ Saved best parameters to: {output_path}")
        
        # Save full study database (already persisted, but confirm location)
        print(f"✅ Full study saved to: {study_db_path}")
        print(f"   (You can load this later with: study = optuna.load_study(study_name='physics_tuning', storage='sqlite:///{study_db_path}'))")
        
        # Export all trial results to CSV for easy analysis
        csv_path = args.export_csv or "trial_results.csv"
        try:
            import pandas as pd
            trials_data = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
                    row = {
                        'trial_number': trial.number,
                        'score': trial.value,
                        'state': trial.state.name
                    }
                    # Add all parameters
                    row.update(trial.params)
                    # Add user attributes if any
                    row.update(trial.user_attrs)
                    trials_data.append(row)
            
            if trials_data:
                df = pd.DataFrame(trials_data)
                df.to_csv(csv_path, index=False)
                print(f"✅ Exported {len(trials_data)} trial results to: {csv_path}")
        except ImportError:
            print(f"⚠️  pandas not available, skipping CSV export")
        except Exception as e:
            print(f"⚠️  Failed to export CSV: {e}")
        
        # Also print summary statistics
        if len(study.trials) > 1:
            scores = [t.value for t in study.trials if t.value is not None]
            if scores:
                print(f"\n📊 Score Statistics:")
                print(f"   Mean: {np.mean(scores):.4f}")
                print(f"   Std:  {np.std(scores):.4f}")
                print(f"   Min:  {np.min(scores):.4f}")
                print(f"   Max:  {np.max(scores):.4f}")
    else:
        print("❌ No successful trials to report!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
