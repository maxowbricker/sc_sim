#!/usr/bin/env python3
"""
Parameter sensitivity analysis for spatial crowdsourcing assignment strategies.

This script systematically explores the impact of:
- Lambda weights (λ₁, λ₂, λ₃) for fairness, starvation, utility
- Soft threshold values
- EWMA gamma parameter

Generates results for research methodology validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import product
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import SIM_CONFIG
import json


def run_parameter_sweep():
    """Run systematic parameter sensitivity analysis."""
    
    print("Loading dataset...")
    workers, tasks = load_workers_tasks("didi")
    print(f"Loaded {len(workers)} workers and {len(tasks)} tasks\n")
    
    # Define parameter ranges for experimentation
    experiments = {
        "lambda_sweep": {
            "description": "Impact of lambda weight combinations",
            "params": {
                "λ1": [0.5, 1.0, 1.5, 2.0],  # Fairness weight
                "λ2": [0.5, 1.0, 1.5, 2.0],  # Starvation weight  
                "λ3": [0.3, 0.5, 1.0, 1.5],  # Utility weight
                "soft_threshold": [1.0],       # Keep constant
                "gamma": [0.3]                 # Keep constant
            }
        },
        "threshold_sweep": {
            "description": "Impact of soft threshold values",
            "params": {
                "λ1": [1.0],                   # Keep constant
                "λ2": [1.0],                   # Keep constant
                "λ3": [0.5],                   # Keep constant
                "soft_threshold": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                "gamma": [0.3]                 # Keep constant
            }
        },
        "gamma_sweep": {
            "description": "Impact of EWMA gamma parameter",
            "params": {
                "λ1": [1.0],                   # Keep constant
                "λ2": [1.0],                   # Keep constant
                "λ3": [0.5],                   # Keep constant
                "soft_threshold": [1.0],       # Keep constant
                "gamma": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
            }
        },
        "focused_comparison": {
            "description": "Key configurations for research comparison",
            "params": {
                "λ1": [0.5, 1.0, 2.0],        # Low, medium, high fairness
                "λ2": [1.0],                   # Standard starvation
                "λ3": [0.5, 1.0],             # Medium vs high utility
                "soft_threshold": [0.5, 1.0, 2.0],  # Permissive, medium, strict
                "gamma": [0.3]                 # Standard EWMA
            }
        }
    }
    
    all_results = {}
    
    for experiment_name, experiment in experiments.items():
        print(f"{'='*60}")
        print(f"EXPERIMENT: {experiment['description']}")
        print(f"{'='*60}")
        
        results = []
        param_combinations = list(product(*experiment["params"].values()))
        param_names = list(experiment["params"].keys())
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            
            # Show progress
            if i % max(1, total_combinations // 10) == 0:
                progress = (i / total_combinations) * 100
                print(f"Progress: {progress:.0f}% ({i+1}/{total_combinations})")
            
            # Configure and run simulation
            config = dict(SIM_CONFIG)
            config["assignment_strategy"] = "composite"
            config["strategy_params"] = dict(config.get("strategy_params", {}))
            config["strategy_params"].update(params)
            
            try:
                summary = run_simulation(workers, tasks, sim_config=config)
                
                # Extract key metrics
                result = {
                    **params,
                    'completed_tasks': summary.get('completed_tasks', 0),
                    'task_assignment_ratio': summary.get('completed_tasks', 0) / len(tasks),
                    'jains_fairness_index': summary.get('final_jains_fairness_index', 0),
                    'utility_difference': summary.get('final_utility_difference_tasks', 0),
                    'fairness_loss': summary.get('final_fairness_loss', 0),
                    'ewma_cv': summary.get('final_ewma_cv', 0),
                    'mean_jfi_over_time': summary.get('mean_jfi_over_time', 0),
                    'backlog_peak': summary.get('backlog_peak', 0),
                    'avg_wait_time': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
                    'empty_km_share': (summary.get('empty_km', 0) / max(1, summary.get('total_travel_km', 1))),
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        all_results[experiment_name] = results
        print(f"Completed {experiment_name}: {len(results)} successful runs\n")
    
    return all_results


def analyze_results(all_results):
    """Analyze and summarize experimental results."""
    
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    for experiment_name, results in all_results.items():
        if not results:
            continue
            
        print(f"\n{experiment_name.upper().replace('_', ' ')}")
        print("-" * 50)
        
        df = pd.DataFrame(results)
        
        # Key metrics to analyze
        key_metrics = ['jains_fairness_index', 'fairness_loss', 'ewma_cv', 
                      'task_assignment_ratio', 'avg_wait_time', 'empty_km_share']
        
        if experiment_name == "lambda_sweep":
            # Find best fairness vs efficiency trade-offs
            df['fairness_score'] = (df['jains_fairness_index'] - df['fairness_loss'] - df['ewma_cv'])
            df['efficiency_score'] = (df['task_assignment_ratio'] - df['avg_wait_time']/10 - df['empty_km_share'])
            
            print("TOP 5 CONFIGURATIONS BY FAIRNESS:")
            top_fairness = df.nlargest(5, 'fairness_score')[['λ1', 'λ2', 'λ3', 'jains_fairness_index', 'fairness_loss', 'ewma_cv']]
            print(top_fairness.to_string(index=False))
            
            print("\nTOP 5 CONFIGURATIONS BY EFFICIENCY:")
            top_efficiency = df.nlargest(5, 'efficiency_score')[['λ1', 'λ2', 'λ3', 'task_assignment_ratio', 'avg_wait_time', 'empty_km_share']]
            print(top_efficiency.to_string(index=False))
            
        elif experiment_name == "threshold_sweep":
            print("SOFT THRESHOLD IMPACT:")
            threshold_analysis = df.groupby('soft_threshold')[key_metrics].mean()
            print(threshold_analysis.round(3).to_string())
            
        elif experiment_name == "gamma_sweep":
            print("EWMA GAMMA PARAMETER IMPACT:")
            gamma_analysis = df.groupby('gamma')[key_metrics].mean()
            print(gamma_analysis.round(3).to_string())
            
        elif experiment_name == "focused_comparison":
            print("KEY RESEARCH CONFIGURATIONS:")
            focused_df = df[['λ1', 'λ3', 'soft_threshold'] + key_metrics]
            print(focused_df.round(3).to_string(index=False))
    
    return all_results


def save_results(all_results, filename="parameter_sensitivity_results.json"):
    """Save results for further analysis."""
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    """Run parameter sensitivity analysis."""
    print("SPATIAL CROWDSOURCING PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Run experiments
    results = run_parameter_sweep()
    
    # Analyze results
    analyze_results(results)
    
    # Save for future reference
    save_results(results)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR YOUR RESEARCH:")
    print(f"{'='*80}")
    print("1. Use these results to establish parameter ranges for PPO")
    print("2. Include top configurations in your methodology comparison")
    print("3. Analyze trade-offs between fairness and efficiency")
    print("4. Use threshold analysis to tune soft threshold mechanism")
    print("5. Consider EWMA gamma sensitivity for different scenarios")


if __name__ == "__main__":
    main()