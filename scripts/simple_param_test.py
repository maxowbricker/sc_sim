#!/usr/bin/env python3
"""
Simple parameter testing script to avoid complex interactions.
Tests specific parameter combinations one by one.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import create_composite_config
import json


def test_parameters():
    """Test key parameter combinations."""
    
    # Load data once
    print("Loading DiDi dataset...")
    workers, tasks = load_workers_tasks("didi")
    print(f"Loaded {len(workers)} workers, {len(tasks)} tasks\n")
    
    # Define test cases (corrected with proper defaults)
    test_cases = [
        {"name": "Baseline", "params": {"λ1": 1.0, "λ2": 1.0, "λ3": 0.5, "soft_threshold": 4.0}},
        {"name": "Fairness Focus", "params": {"λ1": 2.0, "λ2": 1.0, "λ3": 0.3, "soft_threshold": 4.0}},
        {"name": "Efficiency Focus", "params": {"λ1": 0.5, "λ2": 1.0, "λ3": 1.5, "soft_threshold": 4.0}},
        {"name": "Low Threshold", "params": {"λ1": 1.0, "λ2": 1.0, "λ3": 0.5, "soft_threshold": 2.0}},
        {"name": "High Threshold", "params": {"λ1": 1.0, "λ2": 1.0, "λ3": 0.5, "soft_threshold": 6.0}},
        {"name": "Balanced", "params": {"λ1": 1.0, "λ2": 1.0, "λ3": 1.0, "soft_threshold": 4.0}},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] Testing: {test_case['name']}")
        print(f"Parameters: {test_case['params']}")
        
        # Create fresh config using centralized system
        config = create_composite_config(
            assignment_strategy="composite",
            gamma=0.3,  # EWMA parameter
            k=15,       # nearest workers
            **test_case['params']  # Override with test parameters
        )
        
        try:
            # Run simulation
            summary = run_simulation(workers, tasks, sim_config=config)
            
            # Extract results
            completed_tasks = summary.get('completed_tasks', 0)
            task_ratio = completed_tasks / len(tasks)
            
            result = {
                'name': test_case['name'],
                'params': test_case['params'],
                'completed_tasks': completed_tasks,
                'task_assignment_ratio': task_ratio,
                'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                'fairness_loss': summary.get('final_fairness_loss', 0),
                'ewma_cv': summary.get('final_ewma_cv', 0),
                'avg_wait_time': summary.get('total_wait_min', 0) / max(1, completed_tasks),
                'backlog_peak': summary.get('backlog_peak', 0),
            }
            
            results.append(result)
            
            print(f"✅ Completed tasks: {completed_tasks} ({task_ratio:.1%})")
            print(f"   JFI: {result['jains_fairness_index']:.3f}, "
                  f"FL: {result['fairness_loss']:.3f}, "
                  f"EWMA CV: {result['ewma_cv']:.3f}")
            
            if task_ratio < 0.8:
                print(f"⚠️  Low task completion rate: {task_ratio:.1%}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
            
        print("-" * 50)
    
    # Summary
    print(f"\n{'='*60}")
    print("PARAMETER TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Test Case':<20} {'TAR%':<8} {'JFI':<8} {'FL':<8} {'EWMA_CV':<8} {'Wait':<8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<20} {r['task_assignment_ratio']:<8.1%} "
              f"{r['jains_fairness_index']:<8.3f} {r['fairness_loss']:<8.3f} "
              f"{r['ewma_cv']:<8.3f} {r['avg_wait_time']:<8.1f}")
    
    # Save results
    with open('simple_param_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to simple_param_test_results.json")
    print(f"Successfully tested {len(results)} configurations")
    
    return results


if __name__ == "__main__":
    test_parameters()