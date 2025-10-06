#!/usr/bin/env python3
"""
Comprehensive parameter sweep comparing Greedy vs Composite strategies.
Includes ~36 experiments with enhanced metrics collection.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def run_comparative_parameter_sweep():
    """Run comprehensive parameter sweep with Greedy vs Composite comparison."""
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    
    print(f"🚀 COMPARATIVE PARAMETER SWEEP - {timestamp}")
    print("=" * 60)
    print("🎯 Comparing Greedy (baseline) vs Composite (fairness-optimized) strategies")
    print("📊 ~36 experiments with 15K workers dataset")
    print("📈 Enhanced metrics: Supervisor's UD/FL, IOR, temporal data, EWMA trends")
    print()
    
    # Load dataset
    print("📥 Loading dataset...")
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    print(f"✅ Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print()
    
    # Define experimental configurations
    experiments = []
    experiment_id = 1
    
    # 1. Greedy baseline experiments (6 runs for statistical significance)
    print("📋 EXPERIMENT DESIGN:")
    print("=" * 25)
    
    for run in range(1, 7):
        experiments.append({
            'id': experiment_id,
            'name': f'Greedy_Run_{run}',
            'strategy': 'greedy',
            'description': f'Greedy baseline run {run}/6',
            'config_params': {'assignment_strategy': 'greedy'}
        })
        experiment_id += 1
    
    print(f"🎯 Greedy Baseline: {len([e for e in experiments if e['strategy'] == 'greedy'])} experiments")
    
    # 2. Composite strategy experiments (30 parameter combinations)
    fairness_weights = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    starvation_weights = [0.8, 1.0, 1.2]  
    utility_weights = [0.8, 1.0, 1.2]
    soft_thresholds = [0.2, 0.5]
    
    for fw in fairness_weights:
        for sw in starvation_weights:
            for uw in utility_weights:
                for thresh in soft_thresholds:
                    if experiment_id <= 36:  # Cap at ~36 total experiments
                        experiments.append({
                            'id': experiment_id,
                            'name': f'Composite_fw{fw}_sw{sw}_uw{uw}_t{thresh}',
                            'strategy': 'composite',
                            'description': f'Composite: fw={fw}, sw={sw}, uw={uw}, thresh={thresh}',
                            'config_params': {
                                'assignment_strategy': 'composite',
                                'fairness_weight': fw,
                                'starvation_weight': sw,
                                'utility_weight': uw,
                                'soft_threshold': thresh
                            }
                        })
                        experiment_id += 1
    
    print(f"🧮 Composite Variations: {len([e for e in experiments if e['strategy'] == 'composite'])} experiments")
    print(f"🔢 Total Experiments: {len(experiments)}")
    print()
    
    # Run experiments
    results = []
    failed_experiments = []
    
    print("🎬 STARTING EXPERIMENTS:")
    print("=" * 30)
    
    for i, exp in enumerate(experiments, 1):
        try:
            print(f"\n🧪 Experiment {i}/{len(experiments)}: {exp['name']}")
            print(f"   📝 {exp['description']}")
            
            # Create configuration
            config = create_composite_config(**exp['config_params'])
            
            # Run simulation
            exp_start = time.time()
            sim = Simulation(config, workers_df, tasks_df)
            result = sim.run()
            exp_duration = time.time() - exp_start
            
            # Collect comprehensive results
            experiment_result = {
                **exp,
                'duration_seconds': exp_duration,
                'timestamp': datetime.now().isoformat(),
                
                # Core performance metrics
                'tar': result.get('task_assignment_ratio', 0) * 100,
                'jfi': result.get('jfi', 0),
                'avg_wait_time': result.get('avg_wait_time_minutes', 0),
                'avg_pickup_distance': result.get('avg_pickup_distance_km', 0),
                'empty_km_ratio': result.get('empty_km_ratio', 0),
                'backlog_peak': result.get('backlog_peak', 0),
                'completed_tasks': result.get('assigned_tasks', 0),
                
                # Enhanced fairness metrics
                'utility_difference': result.get('utility_difference', 0),
                'fairness_loss': result.get('fairness_loss', 0),
                'ewma_cv': result.get('ewma_cv', 0),
                
                # Supervisor's spatial fairness metrics
                'supervisor_utility_difference': result.get('supervisor_utility_difference'),
                'supervisor_fairness_loss': result.get('supervisor_fairness_loss'),
                'mean_input_output_ratio': result.get('mean_input_output_ratio'),
                'min_input_output_ratio': result.get('min_input_output_ratio'),
                'max_input_output_ratio': result.get('max_input_output_ratio'),
                'workers_with_eligibility_data': result.get('workers_with_eligibility_data', 0),
                'total_task_assignments_tracked': result.get('total_task_assignments_tracked', 0),
            }
            
            results.append(experiment_result)
            
            # Progress summary
            print(f"   ✅ TAR: {experiment_result['tar']:.1f}%, JFI: {experiment_result['jfi']:.3f}")
            print(f"      Wait: {experiment_result['avg_wait_time']:.1f}min, Supervisor FL: {experiment_result['supervisor_fairness_loss']:.3f}" if experiment_result['supervisor_fairness_loss'] else "      Enhanced metrics captured")
            print(f"      Duration: {exp_duration:.1f}s")
            
            # Save temporal data for this experiment
            if hasattr(sim, 'metric_tracker') and sim.metric_tracker:
                temporal_path = f"../../../results/comparative_sweep_{timestamp}/temporal_data/exp_{exp['id']:02d}_{exp['name']}"
                Path(temporal_path).parent.mkdir(parents=True, exist_ok=True)
                sim.metric_tracker.save_all_data(temporal_path)
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            failed_experiments.append({
                'experiment': exp,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            continue
    
    # Save comprehensive results
    total_duration = (datetime.now() - start_time).total_seconds()
    
    final_results = {
        'metadata': {
            'timestamp': timestamp,
            'total_experiments': len(experiments),
            'successful_experiments': len(results),
            'failed_experiments': len(failed_experiments),
            'total_duration_seconds': total_duration,
            'dataset_size': {
                'workers': len(workers_df),
                'tasks': len(tasks_df)
            }
        },
        'results': results,
        'failed_experiments': failed_experiments
    }
    
    # Save results
    results_path = f"../../../results/comparative_parameter_sweep_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n🏆 EXPERIMENT COMPLETED!")
    print("=" * 30)
    print(f"✅ Successful experiments: {len(results)}/{len(experiments)}")
    print(f"⏱️  Total duration: {total_duration/3600:.1f} hours")
    print(f"📁 Results saved: {results_path}")
    
    if failed_experiments:
        print(f"❌ Failed experiments: {len(failed_experiments)}")
    
    # Quick analysis preview
    if results:
        print(f"\n📊 QUICK PREVIEW:")
        print("=" * 20)
        
        greedy_results = [r for r in results if r['strategy'] == 'greedy']
        composite_results = [r for r in results if r['strategy'] == 'composite']
        
        if greedy_results:
            avg_greedy_jfi = sum(r['jfi'] for r in greedy_results) / len(greedy_results)
            avg_greedy_wait = sum(r['avg_wait_time'] for r in greedy_results) / len(greedy_results)
            print(f"🎯 Greedy Average: JFI={avg_greedy_jfi:.3f}, Wait={avg_greedy_wait:.1f}min")
        
        if composite_results:
            best_composite = max(composite_results, key=lambda x: x['jfi'])
            print(f"🏅 Best Composite: JFI={best_composite['jfi']:.3f}, Wait={best_composite['avg_wait_time']:.1f}min")
            print(f"   Params: {best_composite['name']}")
        
        print(f"\n📈 Enhanced data available for all experiments:")
        print(f"   • Supervisor's spatial fairness metrics")  
        print(f"   • Temporal evolution data (EWMA trends, wait time evolution)")
        print(f"   • Task eligibility and IOR analysis")
    
    return results_path

if __name__ == "__main__":
    try:
        results_file = run_comparative_parameter_sweep()
        print(f"\n🚀 Parameter sweep completed successfully!")
        print(f"📊 Load results: {results_file}")
    except Exception as e:
        print(f"\n❌ Parameter sweep failed: {e}")
        print(traceback.format_exc())
