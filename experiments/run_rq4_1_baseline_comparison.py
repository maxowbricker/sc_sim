#!/usr/bin/env python3
"""
RQ4.1: Baseline Strategy Comparison
===================================
Research Question: How does composite scoring perform vs greedy nearest-neighbor 
and random assignment?

This is the **most critical validation experiment** for the entire thesis.
Tests whether the composite approach actually outperforms simple baselines.

Usage:
    python experiments/run_rq4_1_baseline_comparison.py [--quick]
    
Output:
    results/rq4_1_baseline_comparison_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config, get_simulation_config
from simulator.simulation import Simulation
from data.loader import load_data

def run_rq4_1_experiments(quick_mode=False):
    """Run RQ4.1 baseline strategy comparison experiments."""
    
    print("🚀 Starting RQ4.1: Baseline Strategy Comparison")
    print("⭐ THIS IS THE MOST CRITICAL THESIS VALIDATION EXPERIMENT!")
    print("=" * 60)
    
    # Use reasonable composite parameters (can be updated after RQ1.1)
    optimal_lambda1 = 1.5
    optimal_lambda3 = 1.0
    
    # Experiment configuration
    if quick_mode:
        config = {
            'strategies': ['composite', 'greedy', 'random'],
            'composite_params': {
                'lambda1': optimal_lambda1,
                'lambda2': 1.0,
                'lambda3': optimal_lambda3,
                'soft_threshold': 0.01
            },
            'num_runs': 2,
            'dataset': 'didi',
            'max_tasks': 2000,
            'max_workers': 1000,
            'random_seed_base': 42
        }
        print("⚡ QUICK MODE: Testing all strategies with small dataset")
    else:
        config = {
            'strategies': ['composite', 'greedy', 'random'],
            'composite_params': {
                'lambda1': optimal_lambda1,
                'lambda2': 1.0, 
                'lambda3': optimal_lambda3,
                'soft_threshold': 0.01
            },
            'num_runs': 3,
            'dataset': 'didi',
            'max_tasks': 8000,
            'max_workers': 5000,
            'random_seed_base': 42
        }
        print("🔬 FULL MODE: Comprehensive strategy comparison")
    
    print(f"📊 Testing strategies: {config['strategies']}")
    print(f"🔄 {config['num_runs']} runs per strategy")
    print(f"📈 Total experiments: {len(config['strategies']) * config['num_runs']}")
    print(f"⚖️  Composite config: λ₁={config['composite_params']['lambda1']}, λ₃={config['composite_params']['lambda3']}")
    
    estimated_time = len(config['strategies']) * config['num_runs'] * 3  # ~3 min per experiment
    print(f"⏱️  Estimated time: ~{estimated_time} minutes")
    print()
    
    # Results storage
    results = []
    start_time = time.time()
    
    # Run experiments
    for run_id in range(config['num_runs']):
        print(f"📊 RUN {run_id + 1}/{config['num_runs']}")
        
        # Load same data subset for all strategies (consistency)
        workers_df, tasks_df = load_data(
            config['dataset'],
            max_workers=config['max_workers'],
            max_tasks=config['max_tasks']
        )
        
        for i, strategy in enumerate(config['strategies']):
            print(f"  🔧 Testing {strategy.upper()} strategy ({i+1}/{len(config['strategies'])})", end=" ")
            
            try:
                # Create appropriate configuration for each strategy
                if strategy == 'composite':
                    sim_config = create_composite_config(
                        λ1=config['composite_params']['lambda1'],
                        λ2=config['composite_params']['lambda2'],
                        λ3=config['composite_params']['lambda3'],
                        soft_threshold=config['composite_params']['soft_threshold'],
                        assignment_strategy="composite"
                    )
                elif strategy == 'greedy':
                    sim_config = get_simulation_config()
                    sim_config['assignment_strategy'] = 'greedy'
                elif strategy == 'random':
                    sim_config = get_simulation_config()
                    sim_config['assignment_strategy'] = 'greedy'  # Use greedy infrastructure but with random selection
                    np.random.seed(config['random_seed_base'] + run_id)  # Reproducible randomness
                
                # Run simulation
                sim = Simulation(sim_config, workers_df, tasks_df)
                
                # For random strategy, modify assignment to be random
                if strategy == 'random':
                    original_assign = sim.state.assign_task_to_worker
                    def random_assign_task_to_worker(task_id, worker_id=None):
                        if worker_id is None:
                            available_workers = [w for w in sim.state.workers.values() if w.is_available]
                            if available_workers:
                                worker_id = np.random.choice([w.id for w in available_workers])
                        return original_assign(task_id, worker_id)
                    
                    sim.state.assign_task_to_worker = random_assign_task_to_worker
                
                sim_results = sim.run()
                
                # Store results
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'strategy': strategy,
                    
                    # Primary comparison metrics
                    'jfi': sim_results.get('jfi', 0.0),
                    'tar': sim_results.get('task_assignment_ratio', 0.0) * 100,
                    'avg_wait_time': sim_results.get('avg_wait_time_minutes', 0.0),
                    'avg_pickup_distance': sim_results.get('avg_pickup_distance_km', 0.0),
                    
                    # Additional fairness metrics
                    'ewma_cv': sim_results.get('ewma_cv', 1.0),
                    'utility_difference': sim_results.get('utility_difference', 0.0),
                    'fairness_loss': sim_results.get('fairness_loss', 0.0),
                    
                    # System performance
                    'total_tasks': sim_results.get('total_tasks', 0),
                    'assigned_tasks': sim_results.get('assigned_tasks', 0),
                    'total_travel_distance': sim_results.get('total_travel_distance_km', 0.0),
                    
                    # Strategy-specific parameters
                    'lambda1': config['composite_params']['lambda1'] if strategy == 'composite' else None,
                    'lambda2': config['composite_params']['lambda2'] if strategy == 'composite' else None,
                    'lambda3': config['composite_params']['lambda3'] if strategy == 'composite' else None,
                    
                    # Dataset info
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
                wait = result_entry['avg_wait_time']
                print(f"→ JFI: {jfi:.3f}, TAR: {tar:.1f}%, Wait: {wait:.1f}min")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                continue
        
        print()
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/rq4_1_baseline_comparison_{timestamp}.json"
    
    experiment_summary = {
        'experiment': 'RQ4.1 - Baseline Strategy Comparison',
        'timestamp': timestamp,
        'execution_time_minutes': (time.time() - start_time) / 60,
        'config': config,
        'strategies_tested': config['strategies'],
        'total_experiments': len(results),
        'results': results
    }
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    # Analysis and summary
    print("✅ RQ4.1 Baseline Comparison Complete!")
    print(f"📊 Ran {len(results)} experiments in {experiment_summary['execution_time_minutes']:.1f} minutes")
    print(f"💾 Results saved to: {results_file}")
    
    if results:
        # Quick analysis
        results_df = pd.DataFrame(results)
        strategy_summary = results_df.groupby('strategy')[['jfi', 'tar', 'avg_wait_time']].mean().round(3)
        
        print(f"\\n📋 STRATEGY PERFORMANCE PREVIEW:")
        for strategy, row in strategy_summary.iterrows():
            print(f"   {strategy.upper()}: JFI={row['jfi']:.3f}, TAR={row['tar']:.1f}%, Wait={row['avg_wait_time']:.1f}min")
        
        # Key thesis validation check
        if 'composite' in strategy_summary.index and 'greedy' in strategy_summary.index:
            composite_jfi = strategy_summary.loc['composite', 'jfi'] 
            greedy_jfi = strategy_summary.loc['greedy', 'jfi']
            fairness_improvement = composite_jfi - greedy_jfi
            
            if fairness_improvement > 0.02:
                print(f"\\n🎉 THESIS VALIDATION: Composite shows {fairness_improvement:.3f} fairness improvement!")
            elif fairness_improvement > 0:
                print(f"\\n⚡ MODEST SUCCESS: Composite shows {fairness_improvement:.3f} fairness improvement")
            else:
                print(f"\\n⚠️  INVESTIGATE: Composite doesn't improve fairness vs greedy")
        
        print(f"\\n📋 Next step: Analyze detailed results in Jupyter:")
        print(f"   jupyter notebook analysis/Honours_Results_Analysis.ipynb")
    
    print("=" * 60)
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RQ4.1 baseline strategy comparison')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode with smaller dataset')
    args = parser.parse_args()
    
    run_rq4_1_experiments(quick_mode=args.quick)

