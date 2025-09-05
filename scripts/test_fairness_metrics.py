#!/usr/bin/env python3
"""
Quick test of different fairness metrics in composite scoring.

Compares EWMA against traditional fairness metrics without modifying core strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import get_fairness_config, create_composite_config
import pandas as pd


def test_fairness_metrics():
    """Test different fairness metrics with composite strategy."""
    
    print("FAIRNESS METRICS COMPARISON")
    print("=" * 50)
    
    # Load data
    workers, tasks = load_workers_tasks("didi")
    print(f"Testing with {len(workers)} workers, {len(tasks)} tasks\n")
    
    # Test each fairness metric (keeping it simple for now)
    results = []
    
    # Baseline: Current EWMA implementation
    print("1. Testing current EWMA implementation...")
    config_ewma = create_composite_config(
        fairness_metric='ewma',
        λ1=1.0, λ2=1.0, λ3=0.5
    )
    
    summary_ewma = run_simulation(workers, tasks, sim_config=config_ewma)
    results.append({
        'fairness_metric': 'EWMA (Current)',
        'completed_tasks': summary_ewma.get('completed_tasks', 0),
        'task_assignment_ratio': summary_ewma.get('completed_tasks', 0) / len(tasks),
        'jains_fairness_index': summary_ewma.get('final_jains_fairness_index', 0),
        'avg_wait_time': summary_ewma.get('total_wait_min', 0) / max(1, summary_ewma.get('completed_tasks', 1)),
        'empty_km_share': summary_ewma.get('empty_km', 0) / max(1, summary_ewma.get('total_travel_km', 1)),
    })
    
    # Test fairness vs efficiency configurations
    print("\n2. Testing fairness-focused vs efficiency-focused...")
    
    configs = [
        {'name': 'Fairness-Focused', 'λ1': 2.0, 'λ2': 1.0, 'λ3': 0.3},
        {'name': 'Efficiency-Focused', 'λ1': 0.5, 'λ2': 1.0, 'λ3': 1.5},
        {'name': 'Balanced', 'λ1': 1.0, 'λ2': 1.0, 'λ3': 1.0},
    ]
    
    for config_info in configs:
        print(f"   Testing {config_info['name']}...")
        
        config = create_composite_config(
            fairness_metric='ewma',
            λ1=config_info['λ1'], 
            λ2=config_info['λ2'], 
            λ3=config_info['λ3']
        )
        
        summary = run_simulation(workers, tasks, sim_config=config)
        results.append({
            'fairness_metric': f"EWMA ({config_info['name']})",
            'completed_tasks': summary.get('completed_tasks', 0),
            'task_assignment_ratio': summary.get('completed_tasks', 0) / len(tasks),
            'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
            'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
            'empty_km_share': summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)),
        })
    
    # Compare with greedy baseline
    print("\n3. Testing greedy baseline...")
    from config import get_simulation_config
    config_greedy = get_simulation_config()
    config_greedy["assignment_strategy"] = "greedy"
    
    summary_greedy = run_simulation(workers, tasks, sim_config=config_greedy)
    results.append({
        'fairness_metric': 'Greedy (Baseline)',
        'completed_tasks': summary_greedy.get('completed_tasks', 0),
        'task_assignment_ratio': summary_greedy.get('completed_tasks', 0) / len(tasks),
        'jains_fairness_index': summary_greedy.get('final_jains_fairness_index', 0),
        'avg_wait_time': summary_greedy.get('total_wait_min', 0) / max(1, summary_greedy.get('completed_tasks', 1)),
        'empty_km_share': summary_greedy.get('empty_km', 0) / max(1, summary_greedy.get('total_travel_km', 1)),
    })
    
    # Display results
    print(f"\n{'='*80}")
    print("FAIRNESS METRICS COMPARISON RESULTS")
    print(f"{'='*80}")
    
    df = pd.DataFrame(results)
    
    print(f"{'Strategy':<25} {'TAR%':<8} {'JFI':<8} {'Wait(min)':<10} {'Empty%':<8}")
    print("-" * 65)
    
    for _, row in df.iterrows():
        print(f"{row['fairness_metric']:<25} {row['task_assignment_ratio']:<8.1%} "
              f"{row['jains_fairness_index']:<8.3f} {row['avg_wait_time']:<10.1f} "
              f"{row['empty_km_share']:<8.1%}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    ewma_baseline = df[df['fairness_metric'] == 'EWMA (Current)'].iloc[0]
    greedy_baseline = df[df['fairness_metric'] == 'Greedy (Baseline)'].iloc[0]
    
    jfi_improvement = (ewma_baseline['jains_fairness_index'] - greedy_baseline['jains_fairness_index']) / greedy_baseline['jains_fairness_index'] * 100
    wait_cost = (ewma_baseline['avg_wait_time'] - greedy_baseline['avg_wait_time']) / greedy_baseline['avg_wait_time'] * 100
    
    print(f"Your EWMA Composite vs Greedy:")
    print(f"  ✓ Fairness improvement (JFI): +{jfi_improvement:.1f}%")
    print(f"  ↓ Wait time cost: +{wait_cost:.1f}%")
    print(f"  → Trade-off: Better fairness at cost of efficiency")
    
    # Find best fairness configuration
    best_fairness = df.loc[df['jains_fairness_index'].idxmax()]
    best_efficiency = df.loc[df['avg_wait_time'].idxmin()]
    
    print(f"\nBest configurations:")
    print(f"  Best Fairness: {best_fairness['fairness_metric']} (JFI={best_fairness['jains_fairness_index']:.3f})")
    print(f"  Best Efficiency: {best_efficiency['fairness_metric']} (Wait={best_efficiency['avg_wait_time']:.1f}min)")
    
    return results


if __name__ == "__main__":
    test_fairness_metrics()