#!/usr/bin/env python3
"""
Benchmark script for comparing assignment strategies.

Implements the evaluation methodology described in the research paper.
Compares Greedy vs Composite strategies across multiple metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import get_simulation_config, get_experiment_preset, create_composite_config


def run_benchmark(dataset="didi", strategies=None, num_runs=1):
    """
    Run benchmark comparing different assignment strategies.
    
    Parameters:
    -----------
    dataset : str
        Dataset to use ("didi", "synthetic", etc.)
    strategies : list
        List of strategy names to compare
    num_runs : int
        Number of simulation runs per strategy (for statistical significance)
    """
    if strategies is None:
        strategies = ["greedy", "composite"]
    
    print(f"Loading {dataset} dataset...")
    workers, tasks = load_workers_tasks(dataset)
    print(f"Loaded {len(workers)} workers and {len(tasks)} tasks")
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"EVALUATING STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        strategy_results = []
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Configure simulation using centralized config
            if strategy == "composite":
                config = create_composite_config(
                    assignment_strategy=strategy,
                    soft_threshold=1.0  # Ensure reasonable threshold
                )
            else:
                config = get_simulation_config()
                config["assignment_strategy"] = strategy
            
            # Run simulation
            summary = run_simulation(workers, tasks, sim_config=config)
            strategy_results.append(summary)
        
        results[strategy] = strategy_results
    
    # Analyze and compare results
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    comparison_df = analyze_results(results)
    print(comparison_df.to_string())
    
    return results, comparison_df


def analyze_results(results):
    """Analyze and compare strategy results."""
    
    metrics = [
        'completed_tasks', 'final_jains_fairness_index', 'final_utility_difference_tasks',
        'final_fairness_loss', 'final_ewma_cv', 'mean_jfi_over_time',
        'backlog_peak', 'total_travel_km', 'empty_km'
    ]
    
    analysis = {}
    
    for strategy, runs in results.items():
        strategy_stats = {}
        
        for metric in metrics:
            values = [run.get(metric, 0) for run in runs if metric in run]
            if values:
                strategy_stats[f"{metric}_mean"] = np.mean(values)
                strategy_stats[f"{metric}_std"] = np.std(values) if len(values) > 1 else 0
            else:
                strategy_stats[f"{metric}_mean"] = 0
                strategy_stats[f"{metric}_std"] = 0
        
        analysis[strategy] = strategy_stats
    
    # Convert to DataFrame for easy comparison
    df = pd.DataFrame(analysis).T
    
    # Calculate relative improvements
    if len(analysis) > 1:
        strategies = list(analysis.keys())
        baseline = strategies[0]  # Use first strategy as baseline
        
        for i, strategy in enumerate(strategies[1:], 1):
            for metric in metrics:
                baseline_val = df.loc[baseline, f"{metric}_mean"]
                current_val = df.loc[strategy, f"{metric}_mean"]
                
                if baseline_val != 0:
                    improvement = ((current_val - baseline_val) / baseline_val) * 100
                    df.loc[strategy, f"{metric}_improvement_%"] = improvement
    
    return df


def print_research_summary(results):
    """Print summary aligned with research methodology."""
    
    print(f"\n{'='*80}")
    print("RESEARCH METHODOLOGY EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for strategy, runs in results.items():
        print(f"\n{strategy.upper()} STRATEGY:")
        print("-" * 40)
        
        # Average across runs
        avg_results = {}
        for key in runs[0].keys():
            if isinstance(runs[0][key], (int, float)):
                avg_results[key] = np.mean([run[key] for run in runs])
        
        # RQ1: Adaptive Task Assignment Framework
        print("\n1. TASK ASSIGNMENT PERFORMANCE:")
        print(f"   Task Assignment Ratio: {avg_results.get('completed_tasks', 0) / 1862 * 100:.1f}%")
        print(f"   Average Wait Time: {avg_results.get('total_wait_min', 0) / avg_results.get('completed_tasks', 1):.1f} min")
        print(f"   Peak Backlog: {avg_results.get('backlog_peak', 0)}")
        print(f"   Average Travel Distance: {avg_results.get('total_travel_km', 0) / avg_results.get('completed_tasks', 1):.1f} km")
        
        # RQ2: Fairness Metric Optimization
        print("\n2. FAIRNESS METRICS:")
        print(f"   Jain's Fairness Index: {avg_results.get('final_jains_fairness_index', 0):.3f}")
        print(f"   Utility Difference: {avg_results.get('final_utility_difference_tasks', 0):.1f}")
        print(f"   Fairness Loss: {avg_results.get('final_fairness_loss', 0):.3f}")
        print(f"   EWMA Coefficient of Variation: {avg_results.get('final_ewma_cv', 0):.3f}")
        print(f"   Mean JFI over Time: {avg_results.get('mean_jfi_over_time', 0):.3f}")
        
        # Efficiency Metrics
        print("\n3. EFFICIENCY METRICS:")
        empty_share = avg_results.get('empty_km', 0) / avg_results.get('total_travel_km', 1) if avg_results.get('total_travel_km', 0) > 0 else 0
        print(f"   Empty-km Share: {empty_share:.1%}")
        print(f"   Total Travel Distance: {avg_results.get('total_travel_km', 0):.0f} km")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark spatial crowdsourcing strategies")
    parser.add_argument("--dataset", default="didi", help="Dataset to use")
    parser.add_argument("--strategies", nargs="+", default=["greedy", "composite"], 
                       help="Strategies to compare")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per strategy")
    
    args = parser.parse_args()
    
    results, comparison = run_benchmark(
        dataset=args.dataset,
        strategies=args.strategies,
        num_runs=args.runs
    )
    
    print_research_summary(results)