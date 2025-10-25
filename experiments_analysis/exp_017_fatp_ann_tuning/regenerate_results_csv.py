#!/usr/bin/env python3
"""
Regenerate Experiment 017 results CSV with corrected TAR and Throughput metrics.

Since all 12 runs have completed, this script:
1. Loads each JSON file
2. Properly calculates TAR (assignment ratio) and Throughput (completion rate)
3. Generates updated CSV with both metrics
"""

import json
import pandas as pd
from pathlib import Path

print("=" * 80)
print("REGENERATING EXPERIMENT 017 RESULTS CSV")
print("=" * 80)
print()

# Get experiment directory
exp_dir = Path(__file__).parent

# Find all JSON result files
json_files = sorted(exp_dir.glob("exp_017_run*.json"))
print(f"Found {len(json_files)} result files")
print()

results = []

for json_file in json_files:
    print(f"Processing: {json_file.name}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract experiment config
    config = data.get('experiment_config', {})
    run = config.get('run', 0)
    mu = config.get('mu', 0)
    alpha_scale = config.get('alpha_scale', 0)
    description = config.get('description', '')
    
    # Extract metrics with proper calculations
    total_tasks = data.get('total_tasks', 20000)
    completed_tasks = data.get('completed_tasks', 0)
    assigned_tasks = data.get('total_task_assignments_tracked', 0)
    
    # Calculate TAR and Throughput
    tar = assigned_tasks / total_tasks if total_tasks > 0 else 0.0
    throughput = completed_tasks / total_tasks if total_tasks > 0 else 0.0
    
    result = {
        'run': run,
        'mu': mu,
        'alpha_scale': alpha_scale,
        'description': description,
        
        # Task metrics
        'total_tasks': total_tasks,
        'assigned_tasks': assigned_tasks,
        'completed_tasks': completed_tasks,
        'tar': tar,  # Task Assignment Ratio
        'throughput': throughput,  # Task Completion Rate
        'expired_tasks': assigned_tasks - completed_tasks,  # Tasks assigned but not completed
        
        # Fairness metrics
        'jfi': data.get('final_jains_fairness_index', 0.0),
        'gini': data.get('tasks_per_worker_gini', 0.0),
        'utility_diff': data.get('final_utility_difference_tasks', 0.0),
        'fairness_loss': data.get('final_fairness_loss', 0.0),
        'ewma_cv': data.get('final_ewma_cv', 0.0),
        
        # Wait time metrics
        'mean_wait_min': data.get('avg_wait_time_minutes', 0.0),
        'std_wait_min': data.get('std_wait_time_minutes', 0.0),
        'p95_wait_min': data.get('p95_wait_time_minutes', 0.0),
        'max_wait_min': data.get('max_wait_time_minutes', 0.0),
        
        # Worker metrics
        'worker_util': data.get('mean_worker_utilization', 0.0),
        'workers_zero_tasks_pct': data.get('pct_workers_zero_tasks', 0.0),
        'tasks_per_worker_mean': data.get('tasks_per_worker_mean', 0.0),
        'tasks_per_worker_std': data.get('tasks_per_worker_std', 0.0),
        'tasks_per_worker_cv': data.get('tasks_per_worker_cv', 0.0),
        
        # Travel metrics
        'total_travel_km': data.get('total_travel_km', 0.0),
        'empty_km': data.get('empty_km', 0.0),
        'passenger_km': data.get('passenger_km', 0.0),
        'empty_km_pct': (data.get('empty_km', 0.0) / data.get('total_travel_km', 1.0) * 100) if data.get('total_travel_km', 0) > 0 else 0.0,
        
        # Other metrics
        'backlog_peak': data.get('backlog_peak', 0),
    }
    
    results.append(result)
    print(f"  ✓ Run {run}: TAR={tar:.2%}, Throughput={throughput:.2%}, JFI={result['jfi']:.4f}")

print()
print("=" * 80)

# Create DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('run')

# Save to CSV
csv_path = exp_dir / "experiment_017_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"✅ Saved updated results to: {csv_path.name}")
print()

# Print summary table
print("📊 SUMMARY TABLE:")
print()
summary_cols = ['run', 'mu', 'alpha_scale', 'jfi', 'tar', 'throughput', 'mean_wait_min', 'worker_util']
print(results_df[summary_cols].to_string(index=False))
print()

# Find best configurations
print("=" * 80)
print("🏆 BEST CONFIGURATIONS:")
print("=" * 80)
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
print("=" * 80)
print("ANALYSIS:")
print("=" * 80)
print()
print(f"TAR Range: {results_df['tar'].min():.2%} - {results_df['tar'].max():.2%}")
print(f"Throughput Range: {results_df['throughput'].min():.2%} - {results_df['throughput'].max():.2%}")
print(f"Average Expiration Rate: {(1 - results_df['throughput'].mean() / results_df['tar'].mean()):.2%}")
print()
print("TAR vs Throughput Gap (expired tasks):")
print(f"  Min Gap: {(results_df['tar'] - results_df['throughput']).min():.2%}")
print(f"  Max Gap: {(results_df['tar'] - results_df['throughput']).max():.2%}")
print(f"  Mean Gap: {(results_df['tar'] - results_df['throughput']).mean():.2%}")
print()

