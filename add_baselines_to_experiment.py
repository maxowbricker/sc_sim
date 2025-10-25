#!/usr/bin/env python3
"""
Baseline Supplementation Tool for Experiments
==============================================

This script adds missing baseline strategies to completed experiments,
allowing for comprehensive comparisons across all strategies.

Usage:
    python add_baselines_to_experiment.py --exp exp_013_fairness_efficiency_tradeoff --baselines laf ewma_only fatp_ann

Features:
- Automatically detects experiment data configuration (worker/task count, dataset)
- Loads the same data that was used in the original experiment
- Runs only the missing baseline strategies
- Appends results to existing CSV files
- Preserves experiment numbering and structure
"""

import sys
import json
import copy
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

# ============================================================================
# BASELINE STRATEGY CONFIGURATIONS
# ============================================================================

BASELINE_STRATEGIES = {
    'greedy': {
        'name': 'Greedy_Baseline',
        'assignment_strategy': 'greedy',
        'strategy_params': {}
    },
    'laf': {
        'name': 'LAF_Baseline',
        'assignment_strategy': 'laf',
        'strategy_params': {}
    },
    'ewma_only': {
        'name': 'EWMA_Only',
        'assignment_strategy': 'ewma_only',
        'strategy_params': {'gamma': 0.5}
    },
    'random_assign': {
        'name': 'Random_Baseline',
        'assignment_strategy': 'random_assign',
        'strategy_params': {'k': 15}
    },
    'fatp_ann': {
        'name': 'FATP_ANN_Baseline',
        'assignment_strategy': 'fatp_ann',
        'strategy_params': {
            'mu': 0.5,
            'alpha_scale': 0.5,
            'use_k_nearest': False,
            'k': 15
        }
    }
}

# ============================================================================
# EXPERIMENT ANALYSIS & DATA LOADING
# ============================================================================

def analyze_experiment(exp_dir):
    """
    Analyze an experiment directory to understand its configuration.
    
    Returns:
        dict: Experiment metadata including data size, existing strategies, etc.
    """
    exp_path = project_root / "experiments_analysis" / exp_dir
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")
    
    print(f"📊 Analyzing experiment: {exp_dir}")
    print()
    
    # Find CSV file
    csv_files = list(exp_path.glob("**/*aggregate*.csv")) + list(exp_path.glob("**/*results*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No results CSV found in {exp_dir}")
    
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    
    print(f"   Results CSV: {csv_path.relative_to(exp_path)}")
    print(f"   Existing runs: {len(df)}")
    
    # Detect existing strategies
    strategy_col = None
    for col in ['strategy', 'exp_name', 'name']:
        if col in df.columns:
            strategy_col = col
            break
    
    existing_strategies = []
    if strategy_col:
        existing_strategies = df[strategy_col].unique().tolist()
        print(f"   Existing strategies: {', '.join(str(s) for s in existing_strategies[:5])}")
        if len(existing_strategies) > 5:
            print(f"                        ... and {len(existing_strategies) - 5} more")
    
    # Try to infer data configuration from JSONs
    json_files = list(exp_path.glob("**/*summary*.json"))
    data_config = {'workers': None, 'tasks': None, 'dataset': 'didi'}
    
    if json_files:
        with open(json_files[0]) as f:
            sample_data = json.load(f)
            if 'total_tasks' in sample_data:
                data_config['tasks'] = sample_data.get('total_tasks')
            # Try to infer worker count from other metrics
            if 'workers' in sample_data:
                data_config['workers'] = len(sample_data['workers'])
    
    print()
    print(f"   📦 Inferred data config:")
    print(f"      Dataset: {data_config['dataset']}")
    if data_config['workers']:
        print(f"      Workers: {data_config['workers']}")
    if data_config['tasks']:
        print(f"      Tasks: {data_config['tasks']}")
    print()
    
    return {
        'exp_dir': exp_dir,
        'exp_path': exp_path,
        'csv_path': csv_path,
        'existing_strategies': existing_strategies,
        'data_config': data_config,
        'strategy_col': strategy_col,
        'next_exp_id': len(df) + 1
    }

def load_experiment_data(data_config):
    """
    Load worker and task data matching the experiment's configuration.
    """
    print("📊 Loading experiment data...")
    
    # Load from 3-hour peak dataset (most common)
    data_path = project_root / "data" / data_config['dataset']
    workers, tasks = load_workers_tasks(data_config['dataset'], str(data_path))
    
    print(f"✅ Loaded {len(workers)} workers and {len(tasks)} tasks")
    
    # Sample if needed
    target_workers = data_config.get('workers')
    target_tasks = data_config.get('tasks')
    
    if target_workers and len(workers) > target_workers:
        print(f"🎯 Sampling {target_workers} workers...")
        # Simple stratified sampling
        workers_df = pd.DataFrame([{
            'worker': w,
            'release_time': w.release_time
        } for w in workers])
        
        workers_df['time_bin'] = pd.cut(
            (workers_df['release_time'] - workers_df['release_time'].min()).dt.total_seconds(),
            bins=50,
            labels=False
        )
        
        sampled_workers_df = workers_df.groupby('time_bin', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(target_workers / 50))), random_state=42)
        )
        workers = sampled_workers_df['worker'].tolist()[:target_workers]
    
    if target_tasks and len(tasks) > target_tasks:
        print(f"🎯 Sampling {target_tasks} tasks...")
        tasks_df = pd.DataFrame([{
            'task': t,
            'release_time': t.release_time
        } for t in tasks])
        
        tasks_df['time_bin'] = pd.cut(
            (tasks_df['release_time'] - tasks_df['release_time'].min()).dt.total_seconds(),
            bins=50,
            labels=False
        )
        
        sampled_tasks_df = tasks_df.groupby('time_bin', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, int(target_tasks / 50))), random_state=42)
        )
        tasks = sampled_tasks_df['task'].tolist()[:target_tasks]
    
    print(f"✅ Final dataset: {len(workers)} workers, {len(tasks)} tasks")
    print()
    
    return workers, tasks

# ============================================================================
# BASELINE EXECUTION
# ============================================================================

def run_baseline(baseline_key, workers, tasks, exp_info):
    """
    Run a single baseline strategy simulation.
    """
    config = BASELINE_STRATEGIES[baseline_key]
    
    print(f"🚀 Running: {config['name']}")
    print(f"   Strategy: {config['assignment_strategy']}")
    
    # Deep copy data for isolation
    sim_workers = copy.deepcopy(workers)
    sim_tasks = copy.deepcopy(tasks)
    
    # Create simulation config
    sim_config = {
        'assignment_strategy': config['assignment_strategy'],
        'strategy_params': config['strategy_params']
    }
    
    # Run simulation
    start_time = datetime.now()
    print(f"   Started: {start_time.strftime('%H:%M:%S')}")
    
    summary = run_simulation(sim_workers, sim_tasks, sim_config=sim_config)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"   Completed: {end_time.strftime('%H:%M:%S')} (Duration: {duration:.1f} min)")
    
    # Extract key metrics
    result = {
        'exp_id': exp_info['next_exp_id'],
        'exp_name': config['name'],
        'strategy': config['assignment_strategy'],
        'completed_tasks': summary.get('completed_tasks', 0),
        'total_tasks': summary.get('total_tasks', len(tasks)),
        'task_assignment_ratio': summary.get('completed_tasks', 0) / len(tasks),
        'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
        'mean_task_wait_time_min': summary.get('avg_wait_time_minutes', 0),
        'p95_task_wait_time_min': summary.get('p95_wait_time_minutes', 0),
        'mean_worker_utilization': summary.get('mean_worker_utilization', 0),
        'tasks_per_worker_gini': summary.get('tasks_per_worker_gini', 0),
        'tasks_per_worker_mean': summary.get('tasks_per_worker_mean', 0),
        'tasks_per_worker_std': summary.get('tasks_per_worker_std', 0),
        'empty_km': summary.get('empty_km', 0),
        'total_travel_km': summary.get('total_travel_km', 0),
    }
    
    # Save individual JSON
    json_path = exp_info['exp_path'] / f"exp_{exp_info['next_exp_id']:03d}_{config['name']}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   💾 Saved: {json_path.name}")
    
    # Print key metrics
    print(f"   📊 Results:")
    print(f"      JFI: {result['jains_fairness_index']:.4f}")
    print(f"      TAR: {result['task_assignment_ratio']:.2%}")
    print(f"      Mean Wait: {result['mean_task_wait_time_min']:.2f} min")
    print()
    
    exp_info['next_exp_id'] += 1
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Add missing baseline strategies to existing experiments'
    )
    parser.add_argument(
        '--exp',
        required=True,
        help='Experiment directory name (e.g., exp_013_fairness_efficiency_tradeoff)'
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        choices=list(BASELINE_STRATEGIES.keys()),
        help=f'Baselines to add: {", ".join(BASELINE_STRATEGIES.keys())}'
    )
    parser.add_argument(
        '--all-missing',
        action='store_true',
        help='Add all baselines that are not already in the experiment'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually running'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BASELINE SUPPLEMENTATION TOOL")
    print("=" * 80)
    print()
    
    # Analyze experiment
    exp_info = analyze_experiment(args.exp)
    
    # Determine which baselines to run
    if args.all_missing:
        # Find all missing baselines
        existing_lower = [s.lower() for s in exp_info['existing_strategies']]
        baselines_to_run = []
        for key, config in BASELINE_STRATEGIES.items():
            # Check if this baseline is already present
            if (key not in existing_lower and 
                config['name'] not in exp_info['existing_strategies'] and
                config['assignment_strategy'] not in existing_lower):
                baselines_to_run.append(key)
    else:
        if not args.baselines:
            parser.error("Must specify --baselines or --all-missing")
        baselines_to_run = args.baselines
    
    if not baselines_to_run:
        print("✅ No missing baselines to add!")
        return
    
    print("📋 Baselines to add:")
    for bl in baselines_to_run:
        print(f"   • {BASELINE_STRATEGIES[bl]['name']} ({bl})")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN - No simulations will be executed")
        return
    
    # Load data
    workers, tasks = load_experiment_data(exp_info['data_config'])
    
    # Run baselines
    results = []
    for baseline_key in baselines_to_run:
        result = run_baseline(baseline_key, workers, tasks, exp_info)
        results.append(result)
    
    # Append to CSV
    if results:
        results_df = pd.DataFrame(results)
        
        # Load existing CSV
        existing_df = pd.read_csv(exp_info['csv_path'])
        
        # Append new results
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        
        # Save
        updated_df.to_csv(exp_info['csv_path'], index=False)
        
        print("=" * 80)
        print(f"✅ Added {len(results)} baseline(s) to {exp_info['csv_path'].name}")
        print("=" * 80)
        print()
        print("Updated experiment now contains:")
        print(f"   Total runs: {len(updated_df)}")
        if exp_info['strategy_col']:
            print(f"   Strategies: {', '.join(updated_df[exp_info['strategy_col']].unique())}")
    
if __name__ == "__main__":
    main()

