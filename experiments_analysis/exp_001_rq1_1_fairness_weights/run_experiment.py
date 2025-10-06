#!/usr/bin/env python3
"""
RQ1.1: Optimal λ₁ (Fairness Weight) Range Analysis
==================================================
Research Question: What is the optimal λ₁ (fairness weight) range for maximizing 
JFI while maintaining >95% task assignment ratio?

This script runs systematic experiments testing different λ₁ values and saves 
results to JSON for later analysis and visualization.

Usage:
    python experiments/run_rq1_1_fairness_weights.py [--quick]
    
Output:
    ../../../results/rq1_1_fairness_weights_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.loader import load_data

def run_rq1_1_experiments(quick_mode=False):
    """Run RQ1.1 fairness weight experiments."""
    
    print("🚀 Starting RQ1.1: Fairness Weight (λ₁) Analysis")
    print("=" * 60)
    
    # Experiment configuration
    if quick_mode:
        config = {
            'lambda1_values': [0.1, 1.0, 2.0, 10.0],  # Quick test
            'lambda2_fixed': 1.0,
            'lambda3_fixed': 0.5, 
            'soft_threshold': 0.01,
            'num_runs': 2,
            'dataset': 'didi',
            'max_tasks': 2000,
            'max_workers': 1000
        }
        print("⚡ QUICK MODE: Reduced parameter space for testing")
    else:
        config = {
            'lambda1_values': [0.1, 2.0, 10.0, 25.0, 50.0, 100.0, 200.0],  # Full range
            'lambda2_fixed': 1.0,
            'lambda3_fixed': 0.5,
            'soft_threshold': 0.01,
            'num_runs': 3,
            'dataset': 'didi',
            'max_tasks': 15000,   # Full dataset power
            'max_workers': 8000
        }
        print("🔬 FULL MODE: Complete parameter exploration")
    
    print(f"📊 Testing {len(config['lambda1_values'])} λ₁ values: {config['lambda1_values']}")
    print(f"🔄 {config['num_runs']} runs per configuration")
    print(f"📈 Total experiments: {len(config['lambda1_values']) * config['num_runs']}")
    
    estimated_time = len(config['lambda1_values']) * config['num_runs'] * 2  # ~2 min per experiment
    print(f"⏱️  Estimated time: ~{estimated_time} minutes")
    print()
    
    # Results storage
    results = []
    start_time = time.time()
    
    # Run experiments
    for run_id in range(config['num_runs']):
        print(f"📊 RUN {run_id + 1}/{config['num_runs']}")
        
        for i, lambda1 in enumerate(config['lambda1_values']):
            print(f"  🔧 Testing λ₁={lambda1} ({i+1}/{len(config['lambda1_values'])})", end=" ")
            
            try:
                # Create simulation configuration
                sim_config = create_composite_config(
                    λ1=lambda1,
                    λ2=config['lambda2_fixed'],
                    λ3=config['lambda3_fixed'],
                    soft_threshold=config['soft_threshold'],
                    assignment_strategy="composite"
                )
                
                # Load data
                workers_df, tasks_df = load_data(
                    config['dataset'],
                    max_workers=config['max_workers'],
                    max_tasks=config['max_tasks']
                )
                
                # Run simulation
                sim = Simulation(sim_config, workers_df, tasks_df)
                sim_results = sim.run()
                
                # Store results
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'lambda1': lambda1,
                    'lambda2': config['lambda2_fixed'],
                    'lambda3': config['lambda3_fixed'],
                    'soft_threshold': config['soft_threshold'],
                    
                    # Primary metrics
                    'jfi': sim_results.get('jfi', 0.0),
                    'tar': sim_results.get('task_assignment_ratio', 0.0) * 100,
                    'avg_wait_time': sim_results.get('avg_wait_time_minutes', 0.0),
                    'ewma_cv': sim_results.get('ewma_cv', 1.0),
                    
                    # Secondary metrics
                    'utility_difference': sim_results.get('utility_difference', 0.0),
                    'fairness_loss': sim_results.get('fairness_loss', 0.0),
                    'total_tasks': sim_results.get('total_tasks', 0),
                    'assigned_tasks': sim_results.get('assigned_tasks', 0),
                    'avg_pickup_distance': sim_results.get('avg_pickup_distance_km', 0.0),
                    
                    # Success criteria
                    'jfi_success': sim_results.get('jfi', 0.0) > 0.85,
                    'tar_success': sim_results.get('task_assignment_ratio', 0.0) > 0.95,
                    'both_success': (sim_results.get('jfi', 0.0) > 0.85) and (sim_results.get('task_assignment_ratio', 0.0) > 0.95),
                    
                    # Experiment metadata
                    'dataset_size': {
                        'max_workers': config['max_workers'],
                        'max_tasks': config['max_tasks'],
                        'actual_workers': len(workers_df),
                        'actual_tasks': len(tasks_df)
                    }
                }
                
                results.append(result_entry)
                
                # Progress feedback
                jfi = result_entry['jfi']
                tar = result_entry['tar'] 
                success = "✅" if result_entry['both_success'] else "❌"
                print(f"→ JFI: {jfi:.3f}, TAR: {tar:.1f}% {success}")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                continue
        
        print()
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../../../results/rq1_1_fairness_weights_{timestamp}.json"
    
    experiment_summary = {
        'experiment': 'RQ1.1 - Optimal Fairness Weight (λ₁) Analysis',
        'timestamp': timestamp,
        'execution_time_minutes': (time.time() - start_time) / 60,
        'config': config,
        'total_experiments': len(results),
        'successful_experiments': len([r for r in results if r.get('both_success', False)]),
        'results': results
    }
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    # Summary
    print("✅ RQ1.1 Experiments Complete!")
    print(f"📊 Ran {len(results)} experiments in {experiment_summary['execution_time_minutes']:.1f} minutes")
    print(f"💾 Results saved to: {results_file}")
    
    if results:
        success_rate = (experiment_summary['successful_experiments'] / len(results)) * 100
        best_jfi = max(r['jfi'] for r in results)
        best_config = next(r for r in results if r['jfi'] == best_jfi)
        
        print(f"🎯 Success Rate: {success_rate:.1f}% (JFI>0.85 AND TAR>95%)")
        print(f"📈 Best JFI: {best_jfi:.3f} (λ₁={best_config['lambda1']})")
        print(f"📋 Use this command to analyze results:")
        print(f"   jupyter notebook analysis/Honours_Results_Analysis.ipynb")
    
    print("=" * 60)
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RQ1.1 fairness weight experiments')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode with reduced parameters')
    args = parser.parse_args()
    
    run_rq1_1_experiments(quick_mode=args.quick)
