#!/usr/bin/env python3
"""
Simple experiment runner for the root of the repo
Usage: python run_experiment.py <experiment_number>
Example: python run_experiment.py 7
"""

import sys
import subprocess
import os
from pathlib import Path

def run_experiment(exp_number):
    """Run a specific experiment by number."""
    
    # Map experiment numbers to directories
    experiment_map = {
        1: "exp_001_rq1_1_fairness_weights",
        2: "exp_002_comprehensive_parameter_sweep", 
        3: "exp_003_custom_parameter_sweep",
        4: "exp_004_focused_parameter_sweep",
        5: "exp_005_bottleneck_analysis",
        6: "exp_006_comparative_parameter_sweep",
        7: "exp_007_ewma_gamma_sensitivity"
    }
    
    if exp_number not in experiment_map:
        print(f"[ERROR] Error: Experiment {exp_number} not found")
        print(f"[INFO] Available experiments: {list(experiment_map.keys())}")
        return False
    
    exp_dir = f"experiments_analysis/{experiment_map[exp_number]}"
    run_script = f"{exp_dir}/run_experiment.py"
    
    if not os.path.exists(run_script):
        print(f"[ERROR] Error: {run_script} not found")
        return False
    
    print(f"[RUNNING] Running Experiment {exp_number}: {experiment_map[exp_number]}")
    print(f"📁 Directory: {exp_dir}")
    print(f"[INFO] This will run in the background...")
    print()
    
    # Change to experiment directory and run
    os.chdir(exp_dir)
    
    # Run the experiment
    try:
        result = subprocess.run([sys.executable, "run_experiment.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Error running experiment: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <experiment_number>")
        print("Example: python run_experiment.py 7")
        print("\nAvailable experiments:")
        print("  1 - RQ1.1 Fairness Weights")
        print("  2 - Comprehensive Parameter Sweep")
        print("  3 - Custom Parameter Sweep") 
        print("  4 - Focused Parameter Sweep")
        print("  5 - Bottleneck Analysis")
        print("  6 - Comparative Parameter Sweep")
        print("  7 - EWMA Gamma Sensitivity (NEW)")
        sys.exit(1)
    
    try:
        exp_num = int(sys.argv[1])
        success = run_experiment(exp_num)
        if success:
            print("[SUCCESS] Experiment completed successfully!")
        else:
            print("[FAILED] Experiment failed!")
            sys.exit(1)
    except ValueError:
        print("[ERROR] Error: Experiment number must be an integer")
        sys.exit(1)
