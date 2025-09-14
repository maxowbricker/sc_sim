#!/usr/bin/env python3
"""
Focused Parameter Sweep: Optimized Ranges for λ₁, λ₂, λ₃, and Soft Threshold
=========================================================================

Research Goal: Deep exploration of promising parameter ranges identified from
previous experiments to find optimal configurations within refined bounds.

This script performs a focused grid search across optimized ranges:
- λ₁ (Fairness weight): 1.0-2.0 (focused around high-fairness configurations)
- λ₂ (Starvation weight): 0.5-1.5 (balanced starvation prevention)
- λ₃ (Utility weight): 0.5-2.0 (efficiency-focused range)
- soft_threshold: 0.25-1.25 (refined threshold exploration)

Scoring Function: Score = λ₁×Fairness + λ₂×Starvation + λ₃×Utility

Dataset: Large scale with 50,000 tasks for robust results

Usage:
    python experiments/run_focused_parameter_sweep.py [--mode MODE]
    
Modes:
    - standard: 5x5x5x5 grid = 625 experiments (~10-12 hours)
    - fine: 7x7x7x7 grid = 2,401 experiments (~20-24 hours)
    - ultra: 9x9x9x9 grid = 6,561 experiments (~40-48 hours)
    
Output:
    results/focused_parameter_sweep_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import time
import argparse
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from notebook_optimized_loader import load_data

def get_focused_parameter_ranges(mode="standard"):
    """Get focused parameter ranges based on promising configurations."""
    
    # Define the focused ranges with different granularities
    ranges = {
        "standard": {
            "λ1": np.linspace(1.0, 2.0, 5).tolist(),           # [1.0, 1.25, 1.5, 1.75, 2.0]
            "λ2": np.linspace(0.5, 1.5, 5).tolist(),           # [0.5, 0.75, 1.0, 1.25, 1.5]
            "λ3": np.linspace(0.5, 2.0, 5).tolist(),           # [0.5, 0.875, 1.25, 1.625, 2.0]
            "soft_threshold": np.linspace(0.25, 1.25, 5).tolist(),  # [0.25, 0.5, 0.75, 1.0, 1.25]
            "dataset_size": {"max_tasks": 50000, "max_workers": 20000},
            "num_runs": 1,  # 625 experiments
            "description": "Standard focused sweep - balanced detail"
        },
        "fine": {
            "λ1": np.linspace(1.0, 2.0, 7).tolist(),           # 7 values
            "λ2": np.linspace(0.5, 1.5, 7).tolist(),           # 7 values
            "λ3": np.linspace(0.5, 2.0, 7).tolist(),           # 7 values
            "soft_threshold": np.linspace(0.25, 1.25, 7).tolist(),  # 7 values
            "dataset_size": {"max_tasks": 50000, "max_workers": 20000},
            "num_runs": 1,  # 2,401 experiments
            "description": "Fine-grained focused sweep - high detail"
        },
        "ultra": {
            "λ1": np.linspace(1.0, 2.0, 9).tolist(),           # 9 values
            "λ2": np.linspace(0.5, 1.5, 9).tolist(),           # 9 values
            "λ3": np.linspace(0.5, 2.0, 9).tolist(),           # 9 values
            "soft_threshold": np.linspace(0.25, 1.25, 9).tolist(),  # 9 values
            "dataset_size": {"max_tasks": 50000, "max_workers": 20000},
            "num_runs": 1,  # 6,561 experiments
            "description": "Ultra-fine focused sweep - maximum detail"
        }
    }
    
    # Round values to avoid floating point precision issues
    for mode_key, config in ranges.items():
        for param in ["λ1", "λ2", "λ3", "soft_threshold"]:
            config[param] = [round(val, 3) for val in config[param]]
    
    return ranges[mode]

def estimate_experiment_time(config):
    """Estimate total experiment time based on M1 MacBook Pro performance."""
    total_combinations = (len(config["λ1"]) * 
                         len(config["λ2"]) * 
                         len(config["λ3"]) * 
                         len(config["soft_threshold"]) * 
                         config["num_runs"])
    
    # Time estimate for large dataset (50k tasks)
    time_per_experiment = 1.5  # minutes per experiment with large dataset
        
    total_time_minutes = total_combinations * time_per_experiment
    return total_time_minutes, total_combinations

def save_intermediate_results(results, config, start_time, filename):
    """Save intermediate results to prevent data loss."""
    current_time = time.time()
    
    experiment_summary = {
        'experiment': 'Focused Parameter Sweep - Optimized Range Deep Exploration',
        'start_timestamp': datetime.fromtimestamp(start_time).isoformat(),
        'current_timestamp': datetime.now().isoformat(),
        'execution_time_minutes': (current_time - start_time) / 60,
        'config': config,
        'total_experiments_planned': estimate_experiment_time(config)[1],
        'completed_experiments': len(results),
        'progress_percent': (len(results) / estimate_experiment_time(config)[1]) * 100,
        'results': results,
        'parameter_ranges': {
            'lambda1_range': f"{min(config['λ1'])}-{max(config['λ1'])}",
            'lambda2_range': f"{min(config['λ2'])}-{max(config['λ2'])}",
            'lambda3_range': f"{min(config['λ3'])}-{max(config['λ3'])}",
            'threshold_range': f"{min(config['soft_threshold'])}-{max(config['soft_threshold'])}"
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(experiment_summary, f, indent=2)

def analyze_focused_results(results):
    """Analyze results with focus on parameter optimization within ranges."""
    if not results:
        return {}
    
    # Enhanced analysis for focused parameter space
    analysis = {
        'best_jfi': max(results, key=lambda x: x.get('jfi', 0)),
        'best_tar': max(results, key=lambda x: x.get('tar', 0)),
        'best_efficiency': min(results, key=lambda x: x.get('avg_pickup_distance', float('inf'))),
        'best_combined': max(results, key=lambda x: (x.get('jfi', 0) * 0.6 + x.get('tar', 0)/100 * 0.4)),
        'best_balanced': None,
        'pareto_optimal': [],
        'parameter_correlations': {},
        'hot_zones': {}
    }
    
    # Find best balanced configuration
    balanced_candidates = [r for r in results if r.get('jfi', 0) > 0.85 and r.get('tar', 0) > 95]
    if balanced_candidates:
        analysis['best_balanced'] = max(balanced_candidates, key=lambda x: x.get('jfi', 0))
    
    # Find Pareto optimal solutions (JFI vs TAR trade-off)
    pareto_candidates = []
    for result in results:
        jfi = result.get('jfi', 0)
        tar = result.get('tar', 0)
        is_pareto = True
        
        # Check if dominated by any other solution
        for other in results:
            other_jfi = other.get('jfi', 0)
            other_tar = other.get('tar', 0)
            if (other_jfi >= jfi and other_tar >= tar and 
                (other_jfi > jfi or other_tar > tar)):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_candidates.append(result)
    
    analysis['pareto_optimal'] = sorted(pareto_candidates, 
                                      key=lambda x: x.get('jfi', 0), 
                                      reverse=True)[:10]  # Top 10
    
    # Parameter correlation analysis
    params = ['lambda1', 'lambda2', 'lambda3', 'soft_threshold']
    metrics = ['jfi', 'tar', 'avg_pickup_distance']
    
    for param in params:
        analysis['parameter_correlations'][param] = {}
        param_values = [r[param] for r in results if param in r]
        
        for metric in metrics:
            metric_values = [r.get(metric, 0) for r in results if param in r and metric in r]
            if len(param_values) > 1 and len(metric_values) > 1:
                correlation = np.corrcoef(param_values, metric_values)[0, 1]
                analysis['parameter_correlations'][param][metric] = float(correlation) if not np.isnan(correlation) else 0.0
    
    # Identify parameter "hot zones" (high-performing regions)
    successful_results = [r for r in results if r.get('balanced_success', False)]
    if successful_results:
        for param in params:
            values = [r[param] for r in successful_results]
            if values:
                analysis['hot_zones'][param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'optimal_range': f"{min(values):.3f} - {max(values):.3f}"
                }
    
    return analysis

def run_focused_parameter_sweep(mode="standard", save_frequency=50):
    """Run focused parameter sweep on promising parameter ranges."""
    
    print("🎯 FOCUSED PARAMETER SWEEP")
    print("=" * 80)
    print(f"🔬 Mode: {mode.upper()}")
    print("📐 Deep exploration of optimized parameter ranges:")
    print("   • λ₁ (Fairness): 1.0-2.0 (high-fairness focus)")
    print("   • λ₂ (Starvation): 0.5-1.5 (balanced prevention)")  
    print("   • λ₃ (Utility): 0.5-2.0 (efficiency-focused)")
    print("   • Soft Threshold: 0.25-1.25 (refined range)")
    print("🎯 Goal: Find optimal configurations within promising bounds")
    print()
    
    # Get configuration
    config = get_focused_parameter_ranges(mode)
    estimated_time, total_combinations = estimate_experiment_time(config)
    
    print(f"📊 FOCUSED PARAMETER RANGES:")
    print(f"   λ₁ (Fairness):     {len(config['λ1'])} values: {min(config['λ1']):.3f} → {max(config['λ1']):.3f}")
    print(f"   λ₂ (Starvation):   {len(config['λ2'])} values: {min(config['λ2']):.3f} → {max(config['λ2']):.3f}")
    print(f"   λ₃ (Utility):      {len(config['λ3'])} values: {min(config['λ3']):.3f} → {max(config['λ3']):.3f}")
    print(f"   Soft Threshold:    {len(config['soft_threshold'])} values: {min(config['soft_threshold']):.3f} → {max(config['soft_threshold']):.3f}")
    print(f"   Runs per config:   {config['num_runs']}")
    print(f"   Description:       {config['description']}")
    print()
    
    print(f"📈 EXPERIMENT SCOPE:")
    print(f"   Total combinations: {total_combinations:,}")
    print(f"   Dataset size: {config['dataset_size']['max_tasks']:,} tasks, {config['dataset_size']['max_workers']:,} workers")
    print(f"   Estimated time: ~{estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)")
    print()
    
    # Confirm for long experiments
    if estimated_time > 120:  # More than 2 hours
        print(f"⚠️  This is a long experiment (~{estimated_time/60:.1f} hours)")
        print("💾 Results will be saved periodically to prevent data loss")
        
    # Setup results storage
    results = []
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/focused_parameter_sweep_{timestamp}.json"
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print(f"🚀 Starting focused experiments... Results will be saved to {results_file}")
    print(f"💾 Intermediate saves every {save_frequency} experiments")
    print("=" * 80)
    print()
    
    experiment_count = 0
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        config["λ1"],
        config["λ2"], 
        config["λ3"],
        config["soft_threshold"]
    ))
    
    print(f"🔄 Processing {len(param_combinations)} focused parameter combinations...")
    
    # Load data ONCE at start
    print("🚀 Loading large dataset once for all experiments...")
    workers_df, tasks_df = load_data(
        'didi',
        max_workers=config['dataset_size']['max_workers'],
        max_tasks=config['dataset_size']['max_tasks']
    )
    print(f"✅ Large dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print("🚀 Now running focused experiments with pre-loaded data...\n")
    
    # Run experiments
    for run_id in range(config['num_runs']):
        print(f"\n📊 RUN {run_id + 1}/{config['num_runs']}")
        print("-" * 50)
        
        for i, (λ1, λ2, λ3, soft_thresh) in enumerate(param_combinations):
            experiment_count += 1
            
            print(f"🧪 [{experiment_count:>5}/{total_combinations}] λ₁={λ1:>5.3f} λ₂={λ2:>5.3f} λ₃={λ3:>5.3f} thresh={soft_thresh:>5.3f}", end=" ")
            
            try:
                # Create simulation configuration
                sim_config = create_composite_config(
                    λ1=λ1,
                    λ2=λ2,
                    λ3=λ3,
                    soft_threshold=soft_thresh,
                    assignment_strategy="composite"
                )
                
                # Run simulation with timeout protection
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Simulation timed out - likely pathological parameter combination")
                
                # Set timeout (10 minutes per experiment for large dataset)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 10 minute timeout for large dataset
                
                try:
                    sim = Simulation(sim_config, workers_df, tasks_df)
                    sim_results = sim.run()
                    signal.alarm(0)  # Cancel timeout
                except TimeoutError as e:
                    signal.alarm(0)  # Cancel timeout
                    print(f"⏰ TIMEOUT: {str(e)}")
                    # Create dummy results for failed simulation
                    sim_results = {
                        'jfi': 0.0,
                        'task_assignment_ratio': 0.0,
                        'avg_wait_time_minutes': 999.0,
                        'avg_pickup_distance_km': 999.0,
                        'total_travel_distance_km': 0.0,
                        'ewma_cv': 999.0,
                        'utility_difference': 999.0,
                        'fairness_loss': 999.0,
                        'total_tasks': len(tasks_df),
                        'assigned_tasks': 0,
                        'simulation_failed': True,
                        'failure_reason': 'timeout'
                    }
                
                # Store results
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'experiment_id': experiment_count,
                    
                    # Parameters
                    'lambda1': λ1,
                    'lambda2': λ2,
                    'lambda3': λ3,
                    'soft_threshold': soft_thresh,
                    'parameter_sum': λ1 + λ2 + λ3,
                    'fairness_ratio': λ1 / (λ1 + λ2 + λ3),
                    'starvation_ratio': λ2 / (λ1 + λ2 + λ3),
                    'utility_ratio': λ3 / (λ1 + λ2 + λ3),
                    
                    # Primary metrics
                    'jfi': float(sim_results.get('jfi', 0.0)),
                    'tar': float(sim_results.get('task_assignment_ratio', 0.0) * 100),
                    'avg_wait_time': float(sim_results.get('avg_wait_time_minutes', 0.0)),
                    'ewma_cv': float(sim_results.get('ewma_cv', 1.0)),
                    
                    # Efficiency metrics
                    'avg_pickup_distance': float(sim_results.get('avg_pickup_distance_km', 0.0)),
                    'total_travel_km': float(sim_results.get('total_travel_km', 0.0)),
                    'utility_difference': float(sim_results.get('utility_difference', 0.0)),
                    
                    # System performance
                    'total_tasks': int(sim_results.get('total_tasks', 0)),
                    'assigned_tasks': int(sim_results.get('assigned_tasks', 0)),
                    'fairness_loss': float(sim_results.get('fairness_loss', 0.0)),
                    
                    # Success criteria
                    'high_jfi': bool(sim_results.get('jfi', 0.0) > 0.85),
                    'high_tar': bool(sim_results.get('task_assignment_ratio', 0.0) > 0.95),
                    'balanced_success': bool((sim_results.get('jfi', 0.0) > 0.85) and 
                                       (sim_results.get('task_assignment_ratio', 0.0) > 0.95)),
                    'efficiency_optimized': bool(sim_results.get('avg_pickup_distance_km', float('inf')) < 2.0),
                    
                    # Failure tracking
                    'simulation_failed': bool(sim_results.get('simulation_failed', False)),
                    'failure_reason': sim_results.get('failure_reason', 'none'),
                    
                    # Configuration metadata
                    'mode': mode,
                    'focus': 'optimized_ranges',
                    'dataset_size': config['dataset_size'].copy()
                }
                
                results.append(result_entry)
                
                # Progress feedback
                jfi = result_entry['jfi']
                tar = result_entry['tar'] 
                success = "✅" if result_entry['balanced_success'] else "❌"
                
                if result_entry['simulation_failed']:
                    failure_reason = result_entry['failure_reason']
                    print(f"→ ❌ FAILED ({failure_reason})")
                else:
                    efficiency = f"🚗{result_entry['avg_pickup_distance']:.1f}km"
                    print(f"→ JFI:{jfi:.3f} TAR:{tar:.1f}% {efficiency} {success}")
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                continue
            
            # Periodic saves
            if experiment_count % save_frequency == 0:
                print(f"💾 Saving intermediate results... ({len(results)} completed)")
                save_intermediate_results(results, config, start_time, results_file)
                
                # Progress summary
                if results:
                    recent_results = results[-save_frequency:] if len(results) >= save_frequency else results
                    success_rate = sum(1 for r in recent_results if r.get('balanced_success', False)) / len(recent_results) * 100
                    avg_jfi = sum(r.get('jfi', 0) for r in recent_results) / len(recent_results)
                    avg_efficiency = sum(r.get('avg_pickup_distance', 0) for r in recent_results) / len(recent_results)
                    print(f"📊 Recent batch: {success_rate:.1f}% success, JFI:{avg_jfi:.3f}, Efficiency:{avg_efficiency:.1f}km")
                print()
    
    # Final analysis and save
    print("\n🎯 FOCUSED EXPERIMENT COMPLETE! Performing advanced analysis...")
    
    # Enhanced analysis for focused results
    analysis = analyze_focused_results(results)
    
    # Create comprehensive final results
    execution_time = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'Focused Parameter Sweep - Optimized Range Deep Exploration',
        'timestamp': timestamp,
        'execution_time_minutes': execution_time,
        'mode': mode,
        'focus': 'optimized_ranges',
        'config': config,
        'total_experiments': len(results),
        'successful_experiments': len([r for r in results if r.get('balanced_success', False)]),
        'analysis': analysis,
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Final comprehensive summary
    print("✅ FOCUSED PARAMETER SWEEP COMPLETE!")
    print("=" * 80)
    print(f"🕒 Execution time: {execution_time:.1f} minutes ({execution_time/60:.1f} hours)")
    print(f"🧪 Total experiments: {len(results):,}")
    print(f"💾 Results saved to: {results_file}")
    print()
    
    if results:
        success_rate = (final_results['successful_experiments'] / len(results)) * 100
        print(f"📊 FOCUSED SWEEP RESULTS:")
        print(f"   Success rate (JFI>0.85 AND TAR>95%): {success_rate:.1f}%")
        print(f"   Pareto optimal solutions found: {len(analysis.get('pareto_optimal', []))}")
        
        if analysis['best_jfi']:
            best = analysis['best_jfi']
            print(f"   🏆 Best JFI: {best['jfi']:.3f} (λ₁={best['lambda1']:.3f}, λ₂={best['lambda2']:.3f}, λ₃={best['lambda3']:.3f}, thresh={best['soft_threshold']:.3f})")
        
        if analysis['best_balanced']:
            best = analysis['best_balanced']
            print(f"   🎯 Best balanced: JFI={best['jfi']:.3f}, TAR={best['tar']:.1f}% (λ₁={best['lambda1']:.3f}, λ₂={best['lambda2']:.3f}, λ₃={best['lambda3']:.3f})")
        
        if analysis.get('hot_zones'):
            print(f"\n🔥 OPTIMAL PARAMETER ZONES:")
            for param, zone in analysis['hot_zones'].items():
                print(f"   {param}: {zone['optimal_range']} (mean: {zone['mean']:.3f})")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Analyze focused results with enhanced parameter correlation plots")
        print(f"   2. Examine Pareto optimal solutions for trade-off insights")
        print(f"   3. Refine parameter zones further based on hot zones")
        print(f"   4. Consider multi-objective optimization within identified ranges")
    
    print("=" * 80)
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run focused parameter sweep on optimized λ₁, λ₂, λ₃, and soft_threshold ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', 
                       choices=['standard', 'fine', 'ultra'],
                       default='standard',
                       help='Experiment granularity (default: standard - 625 experiments)')
    
    parser.add_argument('--save-freq',
                       type=int,
                       default=50,
                       help='Save intermediate results every N experiments (default: 50)')
    
    args = parser.parse_args()
    
    print(f"🎯 Starting focused parameter sweep in {args.mode} mode...")
    print(f"📐 Exploring optimized ranges with {args.save_freq}-experiment save frequency")
    print()
    
    run_focused_parameter_sweep(
        mode=args.mode,
        save_frequency=args.save_freq
    )
