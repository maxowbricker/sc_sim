#!/usr/bin/env python3
"""
Add Random baseline results to the experiment 015 CSV.
Extract from the simulation output in the log.
"""

import pandas as pd
import os

# Manual extraction from log output
random_results = {
    'exp_id': 3,
    'exp_name': 'Random_Baseline',
    'strategy': 'random_assign',
    'completed_tasks': 15536,
    'total_tasks': 20000,
    'jains_fairness_index': 0.633,
    'gini_coefficient': 0.0,  # Not shown in log, using placeholder
    'mean_wait_time_minutes': 4.7,
    'p90_wait_time_minutes': 11.4,
    'mean_worker_utilization': 0.0,  # Not shown in log, will compute in notebook
    'ewma_final_mean': None  # No EWMA tracking in this rerun
}

# Path to CSV
csv_path = 'data/experiment_015_aggregate_results.csv'

# Load existing CSV
df = pd.read_csv(csv_path)

print(f"Current data: {len(df)} rows")
print(f"Strategies: {df['strategy'].unique()}")

# Remove old Random_Baseline if exists
df = df[df['exp_name'] != 'Random_Baseline']

# Add new Random row
df = pd.concat([df, pd.DataFrame([random_results])], ignore_index=True)
df = df.sort_values('exp_id').reset_index(drop=True)

# Save
df.to_csv(csv_path, index=False)

print(f"\n✅ Updated CSV: {len(df)} rows")
print(f"   Added Random_Baseline:")
print(f"      JFI: {random_results['jains_fairness_index']}")
print(f"      Wait: {random_results['mean_wait_time_minutes']} min")
print(f"      TAR: {random_results['completed_tasks']/random_results['total_tasks']*100:.1f}%")






