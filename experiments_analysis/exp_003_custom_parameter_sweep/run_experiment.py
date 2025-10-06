#!/usr/bin/env python3
"""
Custom Parameter Sweep: ~30 Experiments with 15K Tasks
=====================================================

Targeted parameter exploration designed for:
- Exactly ~30 experiments (manageable experiment size)  
- 15,000 tasks with 10,000 workers (reasonable 0.67 ratio)
- Focus on most promising parameter ranges based on previous research
- Realistic timing expectations for M1 MacBook Pro

This creates a focused grid search with 3×3×2×2×1 = 36 combinations.

Output:
    ../../../results/custom_parameter_sweep_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import time
import itertools
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def get_custom_parameter_ranges():
    """Define focused parameter ranges for ~30 experiments."""
    
    return {
        # Focus on most promising ranges from previous research
        "fairness_weight": [0.5, 1.0, 2.0],        # 3 values - balanced around baseline
        "starvation_weight": [0.5, 1.0, 2.0],      # 3 values - prevent worker starvation  
        "utility_weight": [1.0, 1.5],              # 2 values - efficiency focus
        "soft_threshold": [0.5, 1.0],              # 2 values - around optimal threshold
        
        # Dataset size optimized for reasonable timing
        "dataset_size": {
            "max_tasks": 15000,      # As requested
            "max_workers": 10000     # 0.67 worker/task ratio (suitable)
        },
        "num_runs": 1  # Single run per configuration
        # Total: 3×3×2×2×1 = 36 experiments
    }

def estimate_experiment_time(config):
    """Estimate total experiment time based on observed performance."""
    total_combinations = (len(config["fairness_weight"]) * 
                         len(config["starvation_weight"]) * 
                         len(config["utility_weight"]) * 
                         len(config["soft_threshold"]) * 
                         config["num_runs"])
    
    # Based on timing verification test - conservative estimate for 15k tasks
    time_per_experiment = 2.0  # minutes per experiment (conservative)
    data_load_time = 0.5       # minutes for one-time data loading
    
    total_time_minutes = data_load_time + (total_combinations * time_per_experiment)
    return total_time_minutes, total_combinations

def save_results_json(data, filename):
    """Save results with proper JSON type conversion."""
    def convert_types(obj):
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    with open(filename, 'w') as f:
        json.dump(convert_types(data), f, indent=2)

def run_custom_parameter_sweep(save_frequency=10):
    """Run focused parameter sweep with exactly ~30 experiments."""
    
    print("🎯 CUSTOM PARAMETER SWEEP")
    print("=" * 60)
    print("📊 Target: ~30 experiments with 15K tasks + 10K workers")
    print("🎯 Goal: Find optimal parameters with realistic timing")
    print()
    
    # Get configuration
    config = get_custom_parameter_ranges()
    estimated_time, total_combinations = estimate_experiment_time(config)
    
    print(f"📊 PARAMETER RANGES:")
    print(f"   fairness_weight:     {config['fairness_weight']}")
    print(f"   starvation_weight:   {config['starvation_weight']}")
    print(f"   utility_weight:      {config['utility_weight']}")
    print(f"   soft_threshold:      {config['soft_threshold']}")
    print(f"   Dataset size:        {config['dataset_size']['max_tasks']:,} tasks, {config['dataset_size']['max_workers']:,} workers")
    print()
    
    print(f"📈 EXPERIMENT SCOPE:")
    print(f"   Total combinations:  {total_combinations:,}")
    print(f"   Worker/task ratio:   {config['dataset_size']['max_workers']/config['dataset_size']['max_tasks']:.2f}")
    print(f"   Estimated time:      ~{estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)")
    print()
    
    # Setup results storage
    results = []
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../../../results/custom_parameter_sweep_{timestamp}.json"
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print(f"🚀 Starting experiments... Results will be saved to {results_file}")
    print(f"💾 Intermediate saves every {save_frequency} experiments")
    print("=" * 60)
    print()
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        config["fairness_weight"],
        config["starvation_weight"], 
        config["utility_weight"],
        config["soft_threshold"]
    ))
    
    print(f"🔄 Processing {len(param_combinations)} parameter combinations...")
    
    # Load data ONCE at start (massive time savings!)
    print("🚀 Loading dataset...")
    data_load_start = time.time()
    
    workers_df, tasks_df = load_data(
        'didi',
        max_workers=config['dataset_size']['max_workers'],
        max_tasks=config['dataset_size']['max_tasks']
    )
    
    data_load_time = time.time() - data_load_start
    print(f"✅ Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print(f"⏱️  Data loading time: {data_load_time:.1f} seconds")
    print("🚀 Now running experiments with pre-loaded data...\n")
    
    experiment_count = 0
    successful_experiments = 0
    
    # Run experiments
    for i, (fairness_weight, starvation_weight, utility_weight, soft_thresh) in enumerate(param_combinations):
        experiment_count += 1
        
        print(f"🧪 [{experiment_count:>2}/{total_combinations}] fw={fairness_weight:>3} sw={starvation_weight:>3} uw={utility_weight:>3} thresh={soft_thresh:>3}", end=" ")
        
        experiment_start = time.time()
        
        try:
            # Create simulation configuration
            sim_config = create_composite_config(
                fairness_weight=fairness_weight,
                starvation_weight=starvation_weight,
                utility_weight=utility_weight,
                soft_threshold=soft_thresh,
                assignment_strategy="composite"
            )
            
            # Run simulation with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Simulation timed out")
            
            # Set 10-minute timeout per experiment
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(600)  # 10 minute timeout
            
            try:
                sim = Simulation(sim_config, workers_df, tasks_df)
                sim_results = sim.run()
                signal.alarm(0)  # Cancel timeout
                
                experiment_time = time.time() - experiment_start
                
                # Store results
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_id': experiment_count,
                    'experiment_time_seconds': experiment_time,
                    
                    # Parameters
                    'fairness_weight': fairness_weight,
                    'starvation_weight': starvation_weight,
                    'utility_weight': utility_weight,
                    'soft_threshold': soft_thresh,
                    
                    # Primary metrics
                    'jfi': float(sim_results.get('jfi', 0.0)),
                    'tar': float(sim_results.get('task_assignment_ratio', 0.0) * 100),
                    'avg_wait_time': float(sim_results.get('avg_wait_time_minutes', 0.0)),
                    
                    # Efficiency metrics
                    'avg_pickup_distance': float(sim_results.get('avg_pickup_distance_km', 0.0)),
                    'total_travel_km': float(sim_results.get('total_travel_km', 0.0)),
                    
                    # Success criteria
                    'high_jfi': bool(sim_results.get('jfi', 0.0) > 0.85),
                    'high_tar': bool(sim_results.get('task_assignment_ratio', 0.0) > 0.95),
                    'balanced_success': bool((sim_results.get('jfi', 0.0) > 0.85) and 
                                           (sim_results.get('task_assignment_ratio', 0.0) > 0.95)),
                    
                    # System performance
                    'total_tasks': int(sim_results.get('total_tasks', 0)),
                    'assigned_tasks': int(sim_results.get('assigned_tasks', 0)),
                    
                    'simulation_failed': False
                }
                
                if result_entry['balanced_success']:
                    successful_experiments += 1
                
                results.append(result_entry)
                
                # Progress feedback
                jfi = result_entry['jfi']
                tar = result_entry['tar'] 
                wait_time = result_entry['avg_wait_time']
                pickup_dist = result_entry['avg_pickup_distance']
                
                success = "✅" if result_entry['balanced_success'] else "❌"
                
                print(f"→ JFI:{jfi:.3f} TAR:{tar:.1f}% Wait:{wait_time:.1f}m Pick:{pickup_dist:.1f}km {success} ({experiment_time:.1f}s)")
                
                signal.alarm(0)  # Cancel timeout
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print(f"⏰ TIMEOUT (>10min)")
                
                # Create dummy results for failed simulation
                result_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_id': experiment_count,
                    'experiment_time_seconds': 600,
                    'fairness_weight': fairness_weight,
                    'starvation_weight': starvation_weight,
                    'utility_weight': utility_weight,
                    'soft_threshold': soft_thresh,
                    'jfi': 0.0,
                    'tar': 0.0,
                    'avg_wait_time': 999.0,
                    'avg_pickup_distance': 999.0,
                    'total_travel_km': 0.0,
                    'high_jfi': False,
                    'high_tar': False,
                    'balanced_success': False,
                    'total_tasks': len(tasks_df),
                    'assigned_tasks': 0,
                    'simulation_failed': True,
                    'failure_reason': 'timeout'
                }
                results.append(result_entry)
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            continue
        
        # Periodic saves
        if experiment_count % save_frequency == 0:
            print(f"💾 Saving intermediate results... ({len(results)} completed)")
            
            # Create intermediate summary
            intermediate_summary = {
                'experiment': 'Custom Parameter Sweep - 15K Tasks Focus',
                'start_timestamp': datetime.fromtimestamp(start_time).isoformat(),
                'current_timestamp': datetime.now().isoformat(),
                'execution_time_minutes': (time.time() - start_time) / 60,
                'config': config,
                'total_experiments_planned': total_combinations,
                'completed_experiments': len(results),
                'successful_experiments': successful_experiments,
                'progress_percent': (len(results) / total_combinations) * 100,
                'results': results
            }
            
            save_results_json(intermediate_summary, results_file)
            
            # Quick progress summary
            if results:
                recent_results = results[-save_frequency:] if len(results) >= save_frequency else results
                success_rate = sum(1 for r in recent_results if r.get('balanced_success', False)) / len(recent_results) * 100
                avg_jfi = sum(r.get('jfi', 0) for r in recent_results) / len(recent_results)
                avg_time = sum(r.get('experiment_time_seconds', 0) for r in recent_results) / len(recent_results)
                print(f"📊 Recent {len(recent_results)} experiments: {success_rate:.1f}% success, {avg_jfi:.3f} avg JFI, {avg_time:.1f}s avg time")
            print()
    
    # Final results analysis
    execution_time = (time.time() - start_time) / 60
    
    print("\n🎯 EXPERIMENT COMPLETE! Analyzing results...")
    
    # Find best results
    best_results = {
        'best_jfi': max(results, key=lambda x: x.get('jfi', 0)) if results else None,
        'best_tar': max(results, key=lambda x: x.get('tar', 0)) if results else None,
        'best_balanced': None,
        'fastest_experiment': min(results, key=lambda x: x.get('experiment_time_seconds', float('inf'))) if results else None
    }
    
    # Find best balanced result
    balanced_candidates = [r for r in results if r.get('balanced_success', False)]
    if balanced_candidates:
        best_results['best_balanced'] = max(balanced_candidates, key=lambda x: x.get('jfi', 0))
    
    # Create final results
    final_results = {
        'experiment': 'Custom Parameter Sweep - 15K Tasks Focus',
        'timestamp': timestamp,
        'execution_time_minutes': execution_time,
        'data_load_time_seconds': data_load_time,
        'config': config,
        'total_experiments': len(results),
        'successful_experiments': successful_experiments,
        'success_rate_percent': (successful_experiments / len(results) * 100) if results else 0,
        'best_results': best_results,
        'results': results
    }
    
    save_results_json(final_results, results_file)
    
    # Final summary
    print("✅ CUSTOM PARAMETER SWEEP COMPLETE!")
    print("=" * 60)
    print(f"🕒 Total execution time: {execution_time:.1f} minutes ({execution_time/60:.1f} hours)")
    print(f"📊 Data loading time: {data_load_time:.1f} seconds")
    print(f"🧪 Total experiments: {len(results):,}")
    print(f"🎯 Successful experiments: {successful_experiments} ({final_results['success_rate_percent']:.1f}%)")
    print(f"💾 Results saved to: {results_file}")
    print()
    
    if results:
        avg_experiment_time = sum(r.get('experiment_time_seconds', 0) for r in results) / len(results)
        print(f"⏱️  TIMING ANALYSIS:")
        print(f"   Average per experiment: {avg_experiment_time:.1f} seconds ({avg_experiment_time/60:.2f} minutes)")
        print(f"   Fastest experiment: {best_results['fastest_experiment']['experiment_time_seconds']:.1f}s")
        
        if best_results['best_balanced']:
            best = best_results['best_balanced']
            print(f"\n🏆 BEST BALANCED RESULT:")
            print(f"   JFI: {best['jfi']:.3f}, TAR: {best['tar']:.1f}%")
            print(f"   Parameters: fw={best['fairness_weight']}, sw={best['starvation_weight']}, uw={best['utility_weight']}, thresh={best['soft_threshold']}")
            print(f"   Wait time: {best['avg_wait_time']:.1f}min, Pickup: {best['avg_pickup_distance']:.1f}km")
        
        if successful_experiments == 0:
            print(f"\n⚠️  No experiments achieved balanced success (JFI>0.85 AND TAR>95%)")
            if best_results['best_jfi']:
                best = best_results['best_jfi']
                print(f"   Best JFI: {best['jfi']:.3f} (fw={best['fairness_weight']}, sw={best['starvation_weight']}, uw={best['utility_weight']}, thresh={best['soft_threshold']})")
    
    print("=" * 60)
    return results_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run custom parameter sweep with ~30 experiments and 15K tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--save-freq',
                       type=int,
                       default=10,
                       help='Save intermediate results every N experiments (default: 10)')
    
    args = parser.parse_args()
    
    run_custom_parameter_sweep(save_frequency=args.save_freq)


