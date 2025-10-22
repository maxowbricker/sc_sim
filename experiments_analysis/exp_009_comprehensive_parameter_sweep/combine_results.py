#!/usr/bin/env python3
"""Combine results from both experiment runs into a single CSV."""

import json
import pandas as pd
from pathlib import Path

def combine_experiment_results():
    """Combine all JSON results into a single CSV."""
    
    # Define the two data directories
    data_dir = Path(__file__).parent / "data"
    dir1 = data_dir / "exp_009_20251019_233102"  # Experiments 1-21
    dir2 = data_dir / "exp_009_20251019_232730"  # Experiments 22-42
    
    all_results = []
    
    # Process both directories
    for directory in [dir1, dir2]:
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            continue
            
        json_files = sorted(directory.glob("exp_*_summary.json"))
        print(f"📂 Processing {len(json_files)} files from {directory.name}")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract flattened result
            result = {
                'experiment_id': data['experiment_id'],
                'group': data['group'],
                'name': data['name'],
                'description': data['description'],
                'strategy': data['strategy'],
                'fairness_weight': data.get('fairness_weight'),
                'starvation_weight': data.get('starvation_weight'),
                'utility_weight': data.get('utility_weight'),
                'soft_threshold': data.get('soft_threshold'),
                'normalize_scores': data.get('normalize_scores'),
                'gamma': data.get('gamma'),
                'completed_tasks': data['completed_tasks'],
                'task_assignment_ratio': data['task_assignment_ratio'],
                'jains_fairness_index': data['jains_fairness_index'],
                'mean_task_wait_time_min': data['mean_task_wait_time_min'],
                'mean_pickup_distance_km': data['mean_pickup_distance_km'],
                'total_travel_km': data['total_travel_km'],
                'peak_backlog': data['peak_backlog'],
                'duration_seconds': data['duration_seconds'],
                'timestamp': data['timestamp'],
            }
            
            # Add additional metrics from full_summary if available
            if 'full_summary' in data and data['full_summary']:
                fs = data['full_summary']
                result['empty_km_ratio'] = fs.get('empty_km_ratio')
                result['ewma_cv'] = fs.get('ewma_cv')
                result['utility_difference'] = fs.get('utility_difference')
                result['fairness_loss'] = fs.get('fairness_loss')
                result['max_wait_time'] = fs.get('max_wait_time')
            
            all_results.append(result)
    
    # Create DataFrame and sort by experiment_id
    df = pd.DataFrame(all_results)
    df = df.sort_values('experiment_id').reset_index(drop=True)
    
    # Save combined results
    output_file = data_dir / "experiment_009_combined_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined {len(df)} experiments")
    print(f"📊 Output: {output_file}")
    print(f"\n   Groups: {sorted(df['group'].unique())}")
    print(f"   Experiment IDs: {df['experiment_id'].min()} - {df['experiment_id'].max()}")
    
    return df

if __name__ == '__main__':
    combine_experiment_results()

