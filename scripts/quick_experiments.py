#!/usr/bin/env python3
"""
Quick experimentation script for testing specific parameter combinations.

Use this for rapid testing of specific ideas before running full sensitivity analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import get_experiment_preset, create_composite_config


def quick_test(name, strategy_params_override=None):
    """Run a quick test with specific parameters."""
    print(f"\n{'='*50}")
    print(f"TESTING: {name}")
    print(f"{'='*50}")
    
    workers, tasks = load_workers_tasks("didi")
    
    # Build configuration using centralized system
    overrides = {"assignment_strategy": "composite"}
    if strategy_params_override:
        overrides.update(strategy_params_override)
        print(f"Parameters: {strategy_params_override}")
    
    config = create_composite_config(**overrides)
    
    summary = run_simulation(workers, tasks, sim_config=config)
    
    # Extract key results
    tar = summary.get('completed_tasks', 0) / len(tasks) * 100
    jfi = summary.get('final_jains_fairness_index', 0)
    fl = summary.get('final_fairness_loss', 0)
    ewma_cv = summary.get('final_ewma_cv', 0)
    avg_wait = summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1))
    empty_share = summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1)) * 100
    
    print(f"\nKEY RESULTS:")
    print(f"  Task Assignment: {tar:.1f}%")
    print(f"  Fairness (JFI): {jfi:.3f}")
    print(f"  Fairness Loss: {fl:.3f}")
    print(f"  EWMA CV: {ewma_cv:.3f}")
    print(f"  Avg Wait Time: {avg_wait:.1f} min")
    print(f"  Empty Travel: {empty_share:.1f}%")
    
    return {
        'tar': tar, 'jfi': jfi, 'fl': fl, 'ewma_cv': ewma_cv, 
        'avg_wait': avg_wait, 'empty_share': empty_share
    }


def main():
    """Run a set of focused experiments."""
    
    print("QUICK PARAMETER EXPERIMENTATION")
    print("Testing key parameter combinations for research insights...")
    
    # Use centralized experiment configurations
    experiment_configs = get_experiment_preset("quick_test_configs")
    experiments = [(config["name"], config["params"]) for config in experiment_configs]
    
    results = []
    for name, params in experiments:
        result = quick_test(name, params)
        result['name'] = name
        result['params'] = params
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Configuration':<25} {'TAR%':<8} {'JFI':<8} {'FL':<8} {'EWMA_CV':<8} {'Wait':<8} {'Empty%':<8}")
    print("-" * 80)
    
    for r in results:
        name = r['name'][:24]
        print(f"{name:<25} {r['tar']:<8.1f} {r['jfi']:<8.3f} {r['fl']:<8.3f} {r['ewma_cv']:<8.3f} {r['avg_wait']:<8.1f} {r['empty_share']:<8.1f}")
    
    # Insights
    print(f"\n{'='*80}")
    print("RESEARCH INSIGHTS")
    print(f"{'='*80}")
    
    # Find best fairness
    best_fairness = max(results, key=lambda x: x['jfi'])
    print(f"Best Fairness (JFI): {best_fairness['name']} (JFI={best_fairness['jfi']:.3f})")
    
    # Find best efficiency  
    best_efficiency = min(results, key=lambda x: x['avg_wait'] + x['empty_share'])
    print(f"Best Efficiency: {best_efficiency['name']} (Wait={best_efficiency['avg_wait']:.1f}min, Empty={best_efficiency['empty_share']:.1f}%)")
    
    # Find balanced
    best_balanced = min(results, key=lambda x: abs(x['jfi'] - 0.85) + abs(x['avg_wait'] - 2.0))
    print(f"Most Balanced: {best_balanced['name']}")
    
    print(f"\nNEXT STEPS:")
    print("1. Use these insights to focus your manual parameter exploration")
    print("2. Run full sensitivity analysis with: python3 scripts/parameter_sensitivity.py")
    print("3. Use best configurations as starting points for PPO training")


if __name__ == "__main__":
    main()