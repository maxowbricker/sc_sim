#!/usr/bin/env python3
"""
Full Dataset Experiment Runner
Strategically designed for 220K task experiments on powerful PC hardware.
"""

import sys
import os
import json
import time
import itertools
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def get_targeted_parameter_ranges(strategy="validation"):
    """Get parameter ranges optimized for full dataset experiments."""
    
    if strategy == "validation":
        # Single best-guess configuration for initial validation
        return {
            "fairness_weight": [1.5],      # Based on your previous findings
            "starvation_weight": [1.0],    # Standard baseline
            "utility_weight": [0.8],       # Slightly efficiency-focused
            "soft_threshold": [0.5],       # Standard threshold
            "dataset_size": {"max_tasks": None, "max_workers": None},  # FULL DATASET
            "num_runs": 1,
            "description": "Single config validation with full dataset"
        }
    
    elif strategy == "targeted_sweep":
        # Small but focused parameter grid around promising regions
        return {
            "fairness_weight": [1.2, 1.5, 1.8],       # 3 values around optimal
            "starvation_weight": [0.8, 1.0, 1.2],     # 3 values  
            "utility_weight": [0.6, 0.8, 1.0],        # 3 values
            "soft_threshold": [0.4, 0.5, 0.6],        # 3 values
            "dataset_size": {"max_tasks": None, "max_workers": None},  # FULL DATASET
            "num_runs": 1,  # 3^4 = 81 experiments (~40 days - still too much!)
            "description": "Targeted sweep with full dataset"
        }
    
    elif strategy == "champion_validation":
        # Run multiple promising configs identified from smaller experiments
        return {
            "configurations": [
                {"fairness_weight": 1.5, "starvation_weight": 1.0, "utility_weight": 0.8, "soft_threshold": 0.5},
                {"fairness_weight": 1.8, "starvation_weight": 0.8, "utility_weight": 0.6, "soft_threshold": 0.6},
                {"fairness_weight": 1.2, "starvation_weight": 1.2, "utility_weight": 1.0, "soft_threshold": 0.4},
            ],
            "dataset_size": {"max_tasks": None, "max_workers": None},  # FULL DATASET
            "num_runs": 1,
            "description": "Champion configurations with full dataset validation"
        }

def run_single_full_dataset_experiment(config_params, experiment_name="full_dataset_test"):
    """Run a single experiment with the full dataset."""
    
    print(f"🚀 STARTING FULL DATASET EXPERIMENT: {experiment_name}")
    print("=" * 70)
    print(f"📊 Parameters:")
    for key, value in config_params.items():
        if key != 'dataset_size':
            print(f"   {key}: {value}")
    print("   dataset_size: FULL DiDi Dataset (220,139 tasks)")
    print()
    
    start_time = time.time()
    
    # Load FULL dataset
    print("🚀 Loading FULL DiDi dataset...")
    workers_df, tasks_df = load_data(
        'didi',
        max_workers=None,  # Load all workers
        max_tasks=None     # Load all tasks
    )
    
    print(f"✅ Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    load_time = time.time() - start_time
    print(f"⏱️  Loading time: {load_time:.1f} seconds")
    print()
    
    # Create simulation configuration
    sim_config = create_composite_config(
        fairness_weight=config_params["fairness_weight"],
        starvation_weight=config_params["starvation_weight"],
        utility_weight=config_params["utility_weight"],
        soft_threshold=config_params["soft_threshold"],
        assignment_strategy="composite"
    )
    
    print("🧪 Starting simulation...")
    sim_start_time = time.time()
    
    # Run simulation with timeout protection (4 hour max)
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Simulation exceeded 4 hour limit")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(4 * 3600)  # 4 hour timeout
    
    try:
        sim = Simulation(sim_config, workers_df, tasks_df)
        sim_results = sim.run()
        signal.alarm(0)  # Cancel timeout
        
        simulation_time = time.time() - sim_start_time
        print(f"✅ Simulation completed in {simulation_time/3600:.2f} hours")
        
    except TimeoutError as e:
        signal.alarm(0)
        print(f"⏰ TIMEOUT: {str(e)}")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"❌ ERROR: {str(e)}")
        return None
    
    # Compile results
    total_time = time.time() - start_time
    
    result_entry = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'total_runtime_hours': total_time / 3600,
        'simulation_runtime_hours': simulation_time / 3600,
        'dataset_loading_seconds': load_time,
        
        # Parameters
        'fairness_weight': config_params["fairness_weight"],
        'starvation_weight': config_params["starvation_weight"], 
        'utility_weight': config_params["utility_weight"],
        'soft_threshold': config_params["soft_threshold"],
        
        # Dataset info
        'total_workers': len(workers_df),
        'total_tasks': len(tasks_df),
        'dataset_scale': 'full',
        
        # Results
        'jfi': float(sim_results.get('jfi', 0.0)),
        'task_assignment_ratio': float(sim_results.get('task_assignment_ratio', 0.0)),
        'avg_wait_time_hours': float(sim_results.get('avg_wait_time_minutes', 0.0) / 60),  # Convert to hours
        'avg_pickup_distance_km': float(sim_results.get('avg_pickup_distance_km', 0.0)),
        'total_travel_km': float(sim_results.get('total_travel_km', 0.0)),
        'ewma_cv': float(sim_results.get('ewma_cv', 1.0)),
        'utility_difference': float(sim_results.get('utility_difference', 0.0)),
        'completed_tasks': int(sim_results.get('completed_tasks', 0)),
        'success': True
    }
    
    # Save individual result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"../../../results/full_dataset_experiment_{experiment_name}_{timestamp}.json"
    
    os.makedirs('results', exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result_entry, f, indent=2)
    
    print(f"\n📊 FULL DATASET EXPERIMENT RESULTS:")
    print(f"   JFI: {result_entry['jfi']:.3f}")
    print(f"   TAR: {result_entry['task_assignment_ratio']*100:.1f}%")
    print(f"   Avg wait time: {result_entry['avg_wait_time_hours']*60:.1f} minutes")
    print(f"   Completed tasks: {result_entry['completed_tasks']:,} / {len(tasks_df):,}")
    print(f"   Total runtime: {result_entry['total_runtime_hours']:.2f} hours")
    print(f"   Results saved to: {result_file}")
    
    return result_entry

def main():
    """Main execution function."""
    
    print("🌟 FULL DATASET EXPERIMENT SUITE")
    print("=" * 50)
    print("🎯 Designed for 220K task DiDi dataset on powerful PC hardware")
    print("⚡ Optimized for Intel i7-10700K with 16GB RAM")
    print()
    
    # Strategy selection
    strategy = input("Select strategy:\n1. Single validation experiment (~12 hours)\n2. Champion config validation (3 configs, ~36 hours)\nChoice (1 or 2): ").strip()
    
    if strategy == "1":
        config = get_targeted_parameter_ranges("validation")
        result = run_single_full_dataset_experiment(
            {
                "fairness_weight": config["fairness_weight"][0],
                "starvation_weight": config["starvation_weight"][0], 
                "utility_weight": config["utility_weight"][0],
                "soft_threshold": config["soft_threshold"][0]
            },
            "validation"
        )
        
    elif strategy == "2":
        config = get_targeted_parameter_ranges("champion_validation")
        results = []
        
        for i, params in enumerate(config["configurations"], 1):
            print(f"\n🏆 RUNNING CHAMPION CONFIG {i}/{len(config['configurations'])}")
            result = run_single_full_dataset_experiment(
                params, 
                f"champion_{i}"
            )
            if result:
                results.append(result)
        
        # Summary comparison
        if results:
            print(f"\n📊 CHAMPION COMPARISON SUMMARY:")
            print("-" * 50)
            for i, result in enumerate(results, 1):
                print(f"Champion {i}: JFI={result['jfi']:.3f}, TAR={result['task_assignment_ratio']*100:.1f}%, Wait={result['avg_wait_time_hours']*60:.1f}min")
    
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print(f"\n🎉 FULL DATASET EXPERIMENTS COMPLETE!")

if __name__ == "__main__":
    main()
