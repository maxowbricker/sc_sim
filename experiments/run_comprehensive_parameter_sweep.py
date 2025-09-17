#!/usr/bin/env python3
"""
Comprehensive Parameter Sweep: λ₁, λ₂, λ₃, and Soft Threshold Exploration
=========================================================================

Research Goal: Systematically explore the relationship between all three lambda 
parameters and the soft threshold to understand their interactions and find 
optimal configurations for different objectives.

This script performs a comprehensive grid search across:
- fairness_weight (Fairness weight): Controls EWMA fairness priority
- starvation_weight (Starvation weight): Controls idle time/starvation prevention priority  
- utility_weight (Utility weight): Controls distance/efficiency priority
- soft_threshold: Controls immediate assignment threshold

Scoring Function: Score = fairness_weight×Fairness + starvation_weight×Starvation + utility_weight×Utility

Usage:
    python experiments/run_comprehensive_parameter_sweep.py [--mode MODE] [--focus FOCUS]
    
Modes:
    - overnight: Full comprehensive sweep (DEFAULT) - 6-8 hours
    - extensive: Extended ranges - 3-4 hours  
    - standard: Reasonable ranges - 1-2 hours
    - quick: Fast testing - 30 minutes
    
Focus Areas:
    - all: Test all combinations (DEFAULT)
    - fairness: Focus on fairness-optimized ranges
    - efficiency: Focus on efficiency-optimized ranges
    - balanced: Focus on balanced configurations
    
Output:
    results/comprehensive_parameter_sweep_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import time
import argparse
import itertools
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def get_parameter_ranges(mode="overnight", focus="all"):
    """Get parameter ranges based on mode and focus area."""
    
    # Base parameter ranges by mode (REALISTIC for M1 MacBook Pro)
    ranges = {
        "quick": {
            "fairness_weight": [0.5, 1.0, 2.0],           # 3 values
            "starvation_weight": [0.5, 1.0, 2.0],         # 3 values
            "utility_weight": [0.5, 1.0],                 # 2 values  
            "soft_threshold": [0.5, 1.0],    # 2 values
            "dataset_size": {"max_tasks": 5000, "max_workers": 2500},  # Research-appropriate scale
            "num_runs": 1                    # = 36 experiments (~1-2 hours)
        },
        "standard": {
            "fairness_weight": [0.3, 1.0, 2.0],           # 3 values
            "starvation_weight": [0.3, 1.0, 2.0],         # 3 values
            "utility_weight": [0.5, 1.0, 1.5],            # 3 values
            "soft_threshold": [0.1, 1.0, 2.0],  # 3 values
            "dataset_size": {"max_tasks": 15000, "max_workers": 7500},  # Medium research scale
            "num_runs": 2                    # = 162 experiments (~6-8 hours)
        },
        "extensive": {
            "fairness_weight": [0.1, 0.5, 1.0, 2.0, 5.0],     # 5 values
            "starvation_weight": [0.1, 0.5, 1.0, 2.0, 5.0],   # 5 values
            "utility_weight": [0.2, 0.5, 1.0, 2.0],           # 4 values
            "soft_threshold": [0.1, 0.5, 1.0, 2.0],  # 4 values
            "dataset_size": {"max_tasks": 50000, "max_workers": 20000},  # Large research scale
            "num_runs": 1                        # = 400 experiments (~12-16 hours)
        },
        "overnight": {
            "fairness_weight": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],      # 7 values
            "starvation_weight": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],    # 7 values
            "utility_weight": [0.2, 0.5, 1.0, 1.5, 2.0],                 # 5 values
            "soft_threshold": [0.05, 0.1, 0.5, 1.0, 2.0],  # 5 values
            "dataset_size": {"max_tasks": None, "max_workers": None},  # FULL DATASET!
            "num_runs": 1                                   # = 1,225 experiments (~24-36 hours)
        }
    }
    
    base_ranges = ranges[mode]
    
    # Apply focus filters
    if focus == "fairness":
        # Focus on fairness-heavy configurations
        base_ranges["fairness_weight"] = [x for x in base_ranges["fairness_weight"] if x >= 1.0]
        base_ranges["starvation_weight"] = [x for x in base_ranges["starvation_weight"] if x <= 2.0]
        base_ranges["utility_weight"] = [x for x in base_ranges["utility_weight"] if x <= 1.0]
    elif focus == "efficiency":
        # Focus on efficiency-heavy configurations  
        base_ranges["fairness_weight"] = [x for x in base_ranges["fairness_weight"] if x <= 1.0]
        base_ranges["starvation_weight"] = [x for x in base_ranges["starvation_weight"] if x <= 1.0]
        base_ranges["utility_weight"] = [x for x in base_ranges["utility_weight"] if x >= 1.0]
    elif focus == "balanced":
        # Focus on balanced configurations
        base_ranges["fairness_weight"] = [x for x in base_ranges["fairness_weight"] if 0.5 <= x <= 2.0]
        base_ranges["starvation_weight"] = [x for x in base_ranges["starvation_weight"] if 0.5 <= x <= 2.0]
        base_ranges["utility_weight"] = [x for x in base_ranges["utility_weight"] if 0.5 <= x <= 2.0]
    # "all" focus keeps all ranges unchanged
    
    return base_ranges

def estimate_experiment_time(config):
    """Estimate total experiment time based on M1 MacBook Pro performance."""
    total_combinations = (len(config["fairness_weight"]) * 
                         len(config["starvation_weight"]) * 
                         len(config["utility_weight"]) * 
                         len(config["soft_threshold"]) * 
                         config["num_runs"])
    
    # Realistic time estimates for M1 MacBook Pro with FIXED conversion speed
    max_tasks = config["dataset_size"]["max_tasks"]
    if max_tasks is None:  # Full dataset
        time_per_experiment = 2.0  # minutes - full dataset (fast conversion)
    elif max_tasks <= 5000:
        time_per_experiment = 0.3  # minutes - small dataset (fast conversion)
    elif max_tasks <= 15000:
        time_per_experiment = 0.5  # minutes - medium dataset (fast conversion)  
    elif max_tasks <= 50000:
        time_per_experiment = 1.0  # minutes - large dataset (fast conversion)
    else:
        time_per_experiment = 1.5  # minutes - very large dataset (fast conversion)
        
    total_time_minutes = total_combinations * time_per_experiment
    return total_time_minutes, total_combinations

def save_intermediate_results(results, config, start_time, filename):
    """Save intermediate results to prevent data loss."""
    current_time = time.time()
    
    experiment_summary = {
        'experiment': 'Comprehensive Parameter Sweep - Multi-dimensional Lambda and Threshold Analysis',
        'start_timestamp': datetime.fromtimestamp(start_time).isoformat(),
        'current_timestamp': datetime.now().isoformat(),
        'execution_time_minutes': (current_time - start_time) / 60,
        'config': config,
        'total_experiments_planned': estimate_experiment_time(config)[1],
        'completed_experiments': len(results),
        'progress_percent': (len(results) / estimate_experiment_time(config)[1]) * 100,
        'results': results
    }
    
    # Use JSON-safe saving to handle numpy types
    try:
        from json_utils import save_results_json
        save_results_json(experiment_summary, filename)
    except ImportError:
        # Fallback for numpy type conversion
        import json
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_types(experiment_summary), f, indent=2)

def analyze_parameter_relationships(results):
    """Provide basic analysis of parameter relationships."""
    if not results:
        return {}
    
    # Find best configurations for different objectives
    analysis = {
        'best_jfi': max(results, key=lambda x: x.get('jfi', 0)),
        'best_tar': max(results, key=lambda x: x.get('tar', 0)),
        'best_combined': max(results, key=lambda x: (x.get('jfi', 0) * 0.6 + x.get('tar', 0)/100 * 0.4)),
        'best_balanced': None,  # JFI > 0.85 AND TAR > 95%
        'parameter_stats': {}
    }
    
    # Find best balanced configuration
    balanced_candidates = [r for r in results if r.get('jfi', 0) > 0.85 and r.get('tar', 0) > 95]
    if balanced_candidates:
        analysis['best_balanced'] = max(balanced_candidates, key=lambda x: x.get('jfi', 0))
    
    # Basic parameter statistics
    for param in ['fairness_weight', 'starvation_weight', 'utility_weight', 'soft_threshold']:
        values = [r[param] for r in results if param in r]
        if values:
            analysis['parameter_stats'][param] = {
                'min': min(values),
                'max': max(values),
                'avg_for_high_jfi': sum([r[param] for r in results if r.get('jfi', 0) > 0.85]) / 
                                   max(1, len([r for r in results if r.get('jfi', 0) > 0.85]))
            }
    
    return analysis

def run_comprehensive_parameter_sweep(mode="overnight", focus="all", save_frequency=50):
    """Run comprehensive parameter sweep across all lambda values and soft threshold."""
    
    print("🌟 COMPREHENSIVE PARAMETER SWEEP")
    print("=" * 80)
    print(f"🎯 Mode: {mode.upper()} | Focus: {focus.upper()}")
    print("📐 Exploring relationships between fairness_weight, starvation_weight, utility_weight, and Soft Threshold")
    print("🔬 Goal: Find optimal parameter combinations for different objectives")
    print()
    
    # Get configuration
    config = get_parameter_ranges(mode, focus)
    estimated_time, total_combinations = estimate_experiment_time(config)
    
    print(f"📊 PARAMETER RANGES:")
    print(f"   fairness_weight:     {config['fairness_weight']}")
    print(f"   starvation_weight:   {config['starvation_weight']}")
    print(f"   utility_weight:      {config['utility_weight']}")
    print(f"   Soft Threshold:    {config['soft_threshold']}")
    print(f"   Runs per config:   {config['num_runs']}")
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
    results_file = f"results/comprehensive_parameter_sweep_{timestamp}.json"
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print(f"🚀 Starting experiments... Results will be saved to {results_file}")
    print(f"💾 Intermediate saves every {save_frequency} experiments")
    print("=" * 80)
    print()
    
    experiment_count = 0
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        config["fairness_weight"],
        config["starvation_weight"], 
        config["utility_weight"],
        config["soft_threshold"]
    ))
    
    print(f"🔄 Processing {len(param_combinations)} unique parameter combinations...")
    
    # Load data ONCE at start (massive time savings!)
    print("🚀 Loading dataset once for all experiments...")
    workers_df, tasks_df = load_data(
        'didi',
        max_workers=config['dataset_size']['max_workers'],
        max_tasks=config['dataset_size']['max_tasks']
    )
    print(f"✅ Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print("🚀 Now running experiments with pre-loaded data...\n")
    
    # Run experiments
    for run_id in range(config['num_runs']):
        print(f"\n📊 RUN {run_id + 1}/{config['num_runs']}")
        print("-" * 50)
        
        for i, (fairness_weight, starvation_weight, utility_weight, soft_thresh) in enumerate(param_combinations):
            experiment_count += 1
            
            print(f"🧪 [{experiment_count:>4}/{total_combinations}] fw={fairness_weight:>4} sw={starvation_weight:>4} uw={utility_weight:>4} thresh={soft_thresh:>4}", end=" ")
            
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
                    raise TimeoutError("Simulation timed out - likely pathological parameter combination")
                
                # Set timeout (5 minutes per experiment max)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minute timeout
                
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
                    'fairness_weight': fairness_weight,
                    'starvation_weight': starvation_weight,
                    'utility_weight': utility_weight,
                    'soft_threshold': soft_thresh,
                    'parameter_sum': fairness_weight + starvation_weight + utility_weight,  # Total weight
                    'fairness_ratio': fairness_weight / (fairness_weight + starvation_weight + utility_weight),  # Relative fairness weight
                    'starvation_ratio': starvation_weight / (fairness_weight + starvation_weight + utility_weight),  # Relative starvation weight
                    'utility_ratio': utility_weight / (fairness_weight + starvation_weight + utility_weight),   # Relative utility weight
                    
                    # Primary metrics (ensure JSON serializable)
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
                    
                    # Success criteria (multiple objectives)
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
                    'focus': focus,
                    'dataset_size': config['dataset_size'].copy()
                }
                
                results.append(result_entry)
                
                # Progress feedback
                jfi = result_entry['jfi']
                tar = result_entry['tar'] 
                success = "✅" if result_entry['balanced_success'] else "❌"
                
                if result_entry['simulation_failed']:
                    failure_reason = result_entry['failure_reason']
                    print(f"→ ❌ FAILED ({failure_reason}) - skipping")
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
                
                # Quick progress summary
                if results:
                    recent_results = results[-save_frequency:] if len(results) >= save_frequency else results
                    success_rate = sum(1 for r in recent_results if r.get('balanced_success', False)) / len(recent_results) * 100
                    avg_jfi = sum(r.get('jfi', 0) for r in recent_results) / len(recent_results)
                    print(f"📊 Recent {len(recent_results)} experiments: {success_rate:.1f}% success, {avg_jfi:.3f} avg JFI")
                print()
    
    # Final save with complete analysis
    print("\n🎯 EXPERIMENT COMPLETE! Analyzing results...")
    
    # Perform final analysis
    analysis = analyze_parameter_relationships(results)
    
    # Create comprehensive final results
    execution_time = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'Comprehensive Parameter Sweep - Multi-dimensional Lambda and Threshold Analysis',
        'timestamp': timestamp,
        'execution_time_minutes': execution_time,
        'mode': mode,
        'focus': focus,
        'config': config,
        'total_experiments': len(results),
        'successful_experiments': len([r for r in results if r.get('balanced_success', False)]),
        'analysis': analysis,
        'results': results
    }
    
    # Use JSON-safe saving to handle numpy types
    try:
        from json_utils import save_results_json
        save_results_json(final_results, results_file)
    except ImportError:
        # Fallback for numpy type conversion
        import json
        def convert_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_types(final_results), f, indent=2)
    
    # Final summary
    print("✅ COMPREHENSIVE PARAMETER SWEEP COMPLETE!")
    print("=" * 80)
    print(f"🕒 Execution time: {execution_time:.1f} minutes ({execution_time/60:.1f} hours)")
    print(f"🧪 Total experiments: {len(results):,}")
    print(f"💾 Results saved to: {results_file}")
    print()
    
    if results:
        success_rate = (final_results['successful_experiments'] / len(results)) * 100
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Success rate (JFI>0.85 AND TAR>95%): {success_rate:.1f}%")
        
        if analysis['best_jfi']:
            best = analysis['best_jfi']
            print(f"   Best JFI: {best['jfi']:.3f} (fw={best['fairness_weight']}, sw={best['starvation_weight']}, uw={best['utility_weight']}, thresh={best['soft_threshold']})")
        
        if analysis['best_balanced']:
            best = analysis['best_balanced']
            print(f"   Best balanced: JFI={best['jfi']:.3f}, TAR={best['tar']:.1f}% (fw={best['fairness_weight']}, sw={best['starvation_weight']}, uw={best['utility_weight']}, thresh={best['soft_threshold']})")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Analyze results: jupyter notebook analysis/Honours_Results_Analysis.ipynb")
        print(f"   2. Create parameter heatmaps and interaction plots")
        print(f"   3. Identify optimal ranges for different objectives")
    
    print("=" * 80)
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run comprehensive parameter sweep across fairness_weight, starvation_weight, utility_weight, and soft_threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', 
                       choices=['quick', 'standard', 'extensive', 'overnight'],
                       default='overnight',
                       help='Experiment mode (default: overnight)')
    
    parser.add_argument('--focus',
                       choices=['all', 'fairness', 'efficiency', 'balanced'], 
                       default='all',
                       help='Focus area for parameter exploration (default: all)')
    
    parser.add_argument('--save-freq',
                       type=int,
                       default=50,
                       help='Save intermediate results every N experiments (default: 50)')
    
    args = parser.parse_args()
    
    run_comprehensive_parameter_sweep(
        mode=args.mode,
        focus=args.focus, 
        save_frequency=args.save_freq
    )
