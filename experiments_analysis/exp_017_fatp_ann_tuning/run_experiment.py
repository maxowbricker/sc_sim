#!/usr/bin/env python3
"""
Experiment 017: FATP-ANN Parameter Tuning
==========================================

Objective:
    Find optimal or near-optimal parameter settings for the FATP-ANN strategy
    by exploring combinations of:
    - mu (decay factor): How quickly utility decays with task wait time
    - alpha_scale (distance emphasis): Scaling factor for base utility (task distance)

Setup:
    - Dataset: 4K workers / 20K tasks / 15-min expiry (3-hour peak Didi data)
    - Strategy: fatp_ann (use_k_nearest=False)
    - Simulations: 12 total (3 mu values × 4 alpha_scale values)

Parameter Grid:
    mu = [0.01, 0.1, 0.5]
    alpha_scale = [0.5, 1.0, 2.0, 5.0]

Expected Runtime:
    ~22 min/sim × 12 sims = ~4.4 hours total
"""

import sys
import json
import copy
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

print("=" * 80)
print("EXPERIMENT 017: FATP-ANN PARAMETER TUNING")
print("=" * 80)
print()


def load_and_sample_data():
    """Load and sample 4K workers and 20K tasks from 3-hour peak dataset."""
    print("📊 Loading 3-hour peak Didi dataset...")
    data_path = project_root / "data" / "didi"
    workers, tasks = load_workers_tasks('didi', str(data_path))
    print(f"✅ Loaded {len(workers)} workers and {len(tasks)} tasks")
    print()
    
    print("🎯 Sampling 4000 workers and 20000 tasks using stratified temporal sampling...")
    
    # Create DataFrames for sampling
    workers_df = pd.DataFrame([{
        'worker': w,
        'release_time': w.release_time
    } for w in workers])
    
    tasks_df = pd.DataFrame([{
        'task': t,
        'release_time': t.release_time
    } for t in tasks])
    
    # Create time bins for stratified sampling
    workers_df['time_bin'] = pd.cut(
        (workers_df['release_time'] - workers_df['release_time'].min()).dt.total_seconds(),
        bins=50,
        labels=False
    )
    tasks_df['time_bin'] = pd.cut(
        (tasks_df['release_time'] - tasks_df['release_time'].min()).dt.total_seconds(),
        bins=50,
        labels=False
    )
    
    # Stratified sampling
    sampled_workers_df = workers_df.groupby('time_bin', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, int(4000 / 50))), random_state=42)
    )
    sampled_tasks_df = tasks_df.groupby('time_bin', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, int(20000 / 50))), random_state=42)
    )
    
    # Extract sampled objects
    sampled_workers = sampled_workers_df['worker'].tolist()[:4000]
    sampled_tasks = sampled_tasks_df['task'].tolist()[:20000]
    
    print(f"✅ Sampled {len(sampled_workers)} workers and {len(sampled_tasks)} tasks")
    print()
    
    return sampled_workers, sampled_tasks


def create_sim_config(mu, alpha_scale):
    """Create simulation configuration for FATP-ANN."""
    return {
        'assignment_strategy': 'fatp_ann',
        'strategy_params': {
            'mu': mu,
            'alpha_scale': alpha_scale,
            'use_k_nearest': False,  # Use full scan (22 min/sim)
            'k': 15  # Not used when use_k_nearest=False
        }
    }


def extract_metrics(summary):
    """Extract key metrics from simulation summary."""
    total_tasks = summary.get('total_tasks', 20000)
    completed_tasks = summary.get('completed_tasks', 0)
    assigned_tasks = summary.get('total_task_assignments_tracked', 0)
    
    return {
        # Fairness metrics
        'jfi': summary.get('final_jains_fairness_index', 0.0),
        'gini': summary.get('tasks_per_worker_gini', 0.0),
        
        # Wait time metrics
        'mean_wait_min': summary.get('avg_wait_time_minutes', 0.0),
        'p95_wait_min': summary.get('p95_wait_time_minutes', 0.0),
        
        # Task metrics
        'total_tasks': total_tasks,
        'assigned_tasks': assigned_tasks,
        'completed_tasks': completed_tasks,
        'tar': assigned_tasks / total_tasks if total_tasks > 0 else 0.0,  # Task Assignment Ratio
        'throughput': completed_tasks / total_tasks if total_tasks > 0 else 0.0,  # Task Completion Rate
        
        # Worker metrics
        'worker_util': summary.get('mean_worker_utilization', 0.0),
        
        # Travel metrics
        'empty_km': summary.get('empty_km', 0.0),
        'total_travel_km': summary.get('total_travel_km', 0.0),
        'empty_km_pct': (summary.get('empty_km', 0.0) / summary.get('total_travel_km', 1.0) * 100) if summary.get('total_travel_km', 0) > 0 else 0.0
    }


def main():
    # Load and sample data once
    base_workers, base_tasks = load_and_sample_data()
    
    # Define parameter grid
    param_grid = [
        # mu, alpha_scale, description
        (0.01, 0.5, "Slow decay, less distance emphasis"),
        (0.01, 1.0, "Slow decay, default distance emphasis"),
        (0.01, 2.0, "Slow decay, moderate distance emphasis"),
        (0.01, 5.0, "Slow decay, high distance emphasis"),
        (0.1, 0.5, "Paper's mu, less distance emphasis"),
        (0.1, 1.0, "Paper's mu, default alpha_scale"),
        (0.1, 2.0, "Paper's mu, moderate distance emphasis"),
        (0.1, 5.0, "Paper's mu, high distance emphasis"),
        (0.5, 0.5, "Faster decay, less distance emphasis"),
        (0.5, 1.0, "Faster decay, default distance emphasis"),
        (0.5, 2.0, "Faster decay, moderate distance emphasis"),
        (0.5, 5.0, "Faster decay, high distance emphasis"),
    ]
    
    results = []
    output_dir = Path(__file__).parent
    
    print(f"🚀 Starting {len(param_grid)} simulations...")
    print(f"   Estimated total runtime: ~{len(param_grid) * 22} minutes ({len(param_grid) * 22 / 60:.1f} hours)")
    print()
    
    for run_idx, (mu, alpha_scale, description) in enumerate(param_grid, 1):
        print("=" * 80)
        print(f"Run {run_idx}/{len(param_grid)}: mu={mu}, alpha_scale={alpha_scale}")
        print(f"Description: {description}")
        print("=" * 80)
        
        # Deep copy data for isolation
        workers = copy.deepcopy(base_workers)
        tasks = copy.deepcopy(base_tasks)
        
        # Create configuration
        sim_config = create_sim_config(mu, alpha_scale)
        
        # Run simulation
        start_time = datetime.now()
        print(f"⏱️  Started: {start_time.strftime('%H:%M:%S')}")
        
        summary = run_simulation(workers, tasks, sim_config=sim_config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print(f"⏱️  Completed: {end_time.strftime('%H:%M:%S')} (Duration: {duration:.1f} min)")
        print()
        
        # Extract metrics
        metrics = extract_metrics(summary)
        
        # Add run metadata
        result = {
            'run': run_idx,
            'mu': mu,
            'alpha_scale': alpha_scale,
            'description': description,
            'duration_min': duration,
            **metrics
        }
        results.append(result)
        
        # Save individual JSON summary
        summary['experiment_config'] = {
            'run': run_idx,
            'mu': mu,
            'alpha_scale': alpha_scale,
            'description': description
        }
        json_path = output_dir / f"exp_017_run{run_idx:02d}_mu{mu}_alpha{alpha_scale}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Saved: {json_path.name}")
        
        # Print key metrics
        print(f"📊 Results:")
        print(f"   Tasks: {metrics['assigned_tasks']}/{metrics['total_tasks']} assigned, {metrics['completed_tasks']} completed")
        print(f"   TAR (Assignment): {metrics['tar']:.2%}")
        print(f"   Throughput (Completion): {metrics['throughput']:.2%}")
        print(f"   JFI: {metrics['jfi']:.4f}")
        print(f"   Mean Wait: {metrics['mean_wait_min']:.2f} min")
        print(f"   Worker Util: {metrics['worker_util']:.2%}")
        print()
    
    # Save aggregate results to CSV
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "experiment_017_results.csv"
    results_df.to_csv(csv_path, index=False)
    print("=" * 80)
    print(f"✅ ALL SIMULATIONS COMPLETE")
    print(f"💾 Results saved to: {csv_path}")
    print("=" * 80)
    print()
    
    # Print summary table
    print("📊 SUMMARY TABLE:")
    print()
    summary_cols = ['run', 'mu', 'alpha_scale', 'jfi', 'tar', 'throughput', 'mean_wait_min', 'worker_util']
    print(results_df[summary_cols].to_string(index=False))
    print()
    
    # Find best configurations
    print("🏆 BEST CONFIGURATIONS:")
    print()
    print(f"Best JFI: Run {results_df.loc[results_df['jfi'].idxmax(), 'run']:.0f} "
          f"(mu={results_df.loc[results_df['jfi'].idxmax(), 'mu']}, "
          f"alpha_scale={results_df.loc[results_df['jfi'].idxmax(), 'alpha_scale']}, "
          f"JFI={results_df['jfi'].max():.4f})")
    
    print(f"Best TAR (Assignment): Run {results_df.loc[results_df['tar'].idxmax(), 'run']:.0f} "
          f"(mu={results_df.loc[results_df['tar'].idxmax(), 'mu']}, "
          f"alpha_scale={results_df.loc[results_df['tar'].idxmax(), 'alpha_scale']}, "
          f"TAR={results_df['tar'].max():.2%})")
    
    print(f"Best Throughput (Completion): Run {results_df.loc[results_df['throughput'].idxmax(), 'run']:.0f} "
          f"(mu={results_df.loc[results_df['throughput'].idxmax(), 'mu']}, "
          f"alpha_scale={results_df.loc[results_df['throughput'].idxmax(), 'alpha_scale']}, "
          f"Throughput={results_df['throughput'].max():.2%})")
    
    print(f"Lowest Wait: Run {results_df.loc[results_df['mean_wait_min'].idxmin(), 'run']:.0f} "
          f"(mu={results_df.loc[results_df['mean_wait_min'].idxmin(), 'mu']}, "
          f"alpha_scale={results_df.loc[results_df['mean_wait_min'].idxmin(), 'alpha_scale']}, "
          f"Wait={results_df['mean_wait_min'].min():.2f} min)")
    print()


if __name__ == "__main__":
    main()

