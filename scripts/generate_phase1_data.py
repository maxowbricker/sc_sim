#!/usr/bin/env python3
"""
Generate Phase 1 experimental data for Jupyter notebook analysis.
Runs baseline comparison and parameter sensitivity experiments, saving results to JSON.
"""

import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_workers_tasks
from simulator.simulation import run_simulation
from config import get_simulation_config, create_composite_config, get_experiment_preset

def run_baseline_comparison():
    """Run greedy vs composite baseline comparison"""
    print("🔄 Running baseline comparison (Greedy vs Composite)...")
    
    workers, tasks = load_workers_tasks("didi")
    
    results = {
        "experiment_type": "baseline_comparison",
        "dataset": "didi",
        "total_workers": len(workers),
        "total_tasks": len(tasks),
        "timestamp": datetime.now().isoformat(),
        "strategies": {}
    }
    
    # Test Greedy
    print("   Testing Greedy strategy...")
    greedy_config = get_simulation_config()
    greedy_config["assignment_strategy"] = "greedy"
    greedy_summary = run_simulation(workers, tasks, sim_config=greedy_config)
    
    results["strategies"]["greedy"] = {
        "name": "Greedy (Baseline)",
        "config": greedy_config,
        "results": greedy_summary
    }
    
    # Test Composite with working parameters
    print("   Testing Composite strategy...")
    composite_config = create_composite_config(
        assignment_strategy="composite",
        λ1=1.0, λ2=1.0, λ3=0.5,
        soft_threshold=0.5
    )
    composite_summary = run_simulation(workers, tasks, sim_config=composite_config)
    
    results["strategies"]["composite"] = {
        "name": "Composite (Balanced)",
        "config": composite_config,
        "results": composite_summary
    }
    
    print(f"   ✅ Greedy: {greedy_summary['completed_tasks']}/{len(tasks)} tasks ({greedy_summary['completed_tasks']/len(tasks)*100:.1f}%)")
    print(f"   ✅ Composite: {composite_summary['completed_tasks']}/{len(tasks)} tasks ({composite_summary['completed_tasks']/len(tasks)*100:.1f}%)")
    
    return results

def run_parameter_sensitivity():
    """Run parameter sensitivity analysis using quick experiment configs"""
    print("🔄 Running parameter sensitivity analysis...")
    
    workers, tasks = load_workers_tasks("didi")
    
    results = {
        "experiment_type": "parameter_sensitivity",
        "dataset": "didi", 
        "total_workers": len(workers),
        "total_tasks": len(tasks),
        "timestamp": datetime.now().isoformat(),
        "configurations": []
    }
    
    # Get experiment configurations
    experiment_configs = get_experiment_preset("quick_test_configs")
    
    for config_info in experiment_configs:
        print(f"   Testing: {config_info['name']}...")
        
        # Create configuration
        overrides = {"assignment_strategy": "composite"}
        overrides.update(config_info['params'])
        config = create_composite_config(**overrides)
        
        # Run simulation
        try:
            summary = run_simulation(workers, tasks, sim_config=config)
            
            result_entry = {
                "name": config_info['name'],
                "parameters": config_info['params'],
                "config": config,
                "results": summary,
                "success": True
            }
            
            print(f"      ✅ {summary['completed_tasks']}/{len(tasks)} tasks ({summary['completed_tasks']/len(tasks)*100:.1f}%)")
            
        except Exception as e:
            result_entry = {
                "name": config_info['name'],
                "parameters": config_info['params'],
                "config": config,
                "error": str(e),
                "success": False
            }
            print(f"      ❌ Error: {e}")
        
        results["configurations"].append(result_entry)
    
    return results

def run_fairness_metrics_comparison():
    """Compare different fairness metrics with composite scoring"""
    print("🔄 Running fairness metrics comparison...")
    
    workers, tasks = load_workers_tasks("didi")
    
    results = {
        "experiment_type": "fairness_metrics_comparison",
        "dataset": "didi",
        "total_workers": len(workers), 
        "total_tasks": len(tasks),
        "timestamp": datetime.now().isoformat(),
        "fairness_variants": []
    }
    
    # Test different fairness metrics
    fairness_metrics = ["ewma", "idle_time", "task_count"]
    
    for metric in fairness_metrics:
        print(f"   Testing fairness metric: {metric}...")
        
        config = create_composite_config(
            assignment_strategy="composite",
            fairness_metric=metric,
            λ1=1.0, λ2=1.0, λ3=0.5,
            soft_threshold=0.5
        )
        
        try:
            summary = run_simulation(workers, tasks, sim_config=config)
            
            result_entry = {
                "fairness_metric": metric,
                "config": config,
                "results": summary,
                "success": True
            }
            
            print(f"      ✅ {summary['completed_tasks']}/{len(tasks)} tasks, JFI={summary.get('jains_fairness_index', 0):.3f}")
            
        except Exception as e:
            result_entry = {
                "fairness_metric": metric,
                "config": config,
                "error": str(e),
                "success": False
            }
            print(f"      ❌ Error: {e}")
        
        results["fairness_variants"].append(result_entry)
    
    return results

def save_results(results, filename):
    """Save results to JSON file with proper serialization"""
    
    def make_serializable(obj):
        """Convert non-serializable objects to serializable format"""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'total_seconds'):  # timedelta objects
            return obj.total_seconds()
        elif isinstance(obj, (int, float, str, bool, list, dict, type(None))):
            return obj
        else:
            return str(obj)
    
    def recursive_serialize(data):
        """Recursively serialize nested data structures"""
        if isinstance(data, dict):
            return {k: recursive_serialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_serialize(item) for item in data]
        else:
            return make_serializable(data)
    
    serializable_results = recursive_serialize(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"   💾 Saved results to {filename}")

def main():
    """Run all Phase 1 experiments and save data for notebook analysis"""
    print("🎯 GENERATING PHASE 1 EXPERIMENTAL DATA")
    print("=" * 50)
    
    # Run experiments
    baseline_results = run_baseline_comparison()
    save_results(baseline_results, "phase1_baseline_results.json")
    
    param_results = run_parameter_sensitivity() 
    save_results(param_results, "phase1_parameter_sensitivity.json")
    
    fairness_results = run_fairness_metrics_comparison()
    save_results(fairness_results, "phase1_fairness_comparison.json")
    
    # Create summary
    summary = {
        "phase1_summary": {
            "timestamp": datetime.now().isoformat(),
            "experiments_completed": 3,
            "total_simulations": (
                len(baseline_results["strategies"]) + 
                len([c for c in param_results["configurations"] if c["success"]]) +
                len([f for f in fairness_results["fairness_variants"] if f["success"]])
            ),
            "dataset_info": {
                "name": "didi",
                "workers": baseline_results["total_workers"],
                "tasks": baseline_results["total_tasks"]
            },
            "data_files": [
                "phase1_baseline_results.json",
                "phase1_parameter_sensitivity.json", 
                "phase1_fairness_comparison.json"
            ]
        }
    }
    
    save_results(summary, "phase1_summary.json")
    
    print("\n✅ PHASE 1 DATA GENERATION COMPLETE!")
    print("📊 Generated files:")
    print("   - phase1_baseline_results.json")
    print("   - phase1_parameter_sensitivity.json") 
    print("   - phase1_fairness_comparison.json")
    print("   - phase1_summary.json")
    print("\n🎒 Ready for Jupyter notebook analysis!")

if __name__ == "__main__":
    main()
