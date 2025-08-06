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
from config import SIM_CONFIG


def quick_test(name, strategy_params_override=None):
    """Run a quick test with specific parameters."""
    print(f"\n{'='*50}")
    print(f"TESTING: {name}")
    print(f"{'='*50}")
    
    workers, tasks = load_workers_tasks("didi")
    
    config = dict(SIM_CONFIG)
    config["assignment_strategy"] = "composite"
    
    if strategy_params_override:
        config["strategy_params"] = dict(config.get("strategy_params", {}))
        config["strategy_params"].update(strategy_params_override)
        print(f"Parameters: {strategy_params_override}")
    
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
    
    experiments = [
        # Baseline
        ("Baseline (Current)", {}),
        
        # Fairness-focused configurations
        ("High Fairness Focus", {"λ1": 2.0, "λ2": 1.0, "λ3": 0.3}),
        ("Very High Fairness", {"λ1": 3.0, "λ2": 1.0, "λ3": 0.2}),
        
        # Efficiency-focused configurations  
        ("High Efficiency Focus", {"λ1": 0.3, "λ2": 1.0, "λ3": 2.0}),
        ("Very High Efficiency", {"λ1": 0.2, "λ2": 0.5, "λ3": 3.0}),
        
        # Starvation-focused
        ("High Starvation Prevention", {"λ1": 1.0, "λ2": 3.0, "λ3": 0.5}),
        
        # Balanced approaches
        ("Balanced Equal Weights", {"λ1": 1.0, "λ2": 1.0, "λ3": 1.0}),
        ("Balanced with Fairness Bias", {"λ1": 1.5, "λ2": 1.0, "λ3": 0.7}),
        
        # Threshold experiments
        ("Permissive Threshold", {"soft_threshold": 0.5}),
        ("Strict Threshold", {"soft_threshold": 2.0}),
        ("Very Strict Threshold", {"soft_threshold": 3.0}),
        
        # EWMA sensitivity
        ("Responsive EWMA", {"gamma": 0.1}),
        ("Smooth EWMA", {"gamma": 0.7}),
    ]
    
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