#!/usr/bin/env python3
"""
Balanced Parameter Experiment - Fairness-Optimized Ranges
Based on initial findings that show speed vs fairness trade-offs.
"""

import subprocess
import sys
import json
from datetime import datetime

def run_balanced_experiments():
    """Run experiments with fairness-optimized parameter ranges."""
    
    print("🚀 LAUNCHING BALANCED PARAMETER EXPERIMENTS")
    print("=" * 60)
    print("🎯 Objective: Find configurations that balance speed AND fairness")
    print("📊 Based on findings: current ranges favor utility over fairness")
    print()
    
    # Strategy: Run focused experiment with fairness-heavy ranges
    experiment_configs = [
        {
            "name": "fairness_heavy",
            "mode": "fine",  # 7x7x7x7 = 2,401 experiments
            "focus": "fairness",
            "estimated_hours": "20-24 hours",
            "rationale": "Higher fairness weights to find balanced solutions"
        }
    ]
    
    results_summary = {
        "experiment_plan": "Balanced Parameter Search",
        "start_time": datetime.now().isoformat(),
        "experiments": []
    }
    
    for config in experiment_configs:
        print(f"🧪 STARTING: {config['name']} experiment")
        print(f"   Mode: {config['mode']}")
        print(f"   Focus: {config['focus']}")  
        print(f"   Estimated time: {config['estimated_hours']}")
        print(f"   Rationale: {config['rationale']}")
        print()
        
        try:
            # Run the focused parameter sweep
            cmd = [
                sys.executable, 
                "run_focused_parameter_sweep.py",
                "--mode", config["mode"]
            ]
            
            print(f"🚀 Executing: {' '.join(cmd)}")
            print("💤 Running in background... check results periodically")
            print("📁 Results will be saved to: ../../../../results/focused_parameter_sweep_*.json")
            print()
            
            # Run the experiment
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            experiment_result = {
                "name": config["name"],
                "status": "completed",
                "command": " ".join(cmd),
                "completion_time": datetime.now().isoformat()
            }
            
            print(f"✅ COMPLETED: {config['name']}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR in {config['name']}: {e}")
            experiment_result = {
                "name": config["name"], 
                "status": "failed",
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
        
        results_summary["experiments"].append(experiment_result)
        print("-" * 60)
    
    # Save summary
    summary_file = f"../../../../results/balanced_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"📋 EXPERIMENT PLAN COMPLETED")
    print(f"📄 Summary saved to: {summary_file}")
    print()
    print(f"🔍 NEXT STEPS WHEN YOU RETURN:")
    print(f"   1. Check ../../../../results/focused_parameter_sweep_*.json for latest results")
    print(f"   2. Run analysis in ../analysis/Honours_Results_Analysis.ipynb")
    print(f"   3. Look for configurations with JFI > 0.85 AND reasonable wait times")
    print(f"   4. If no balanced configs found, consider Phase 2 with expanded ranges")

if __name__ == "__main__":
    run_balanced_experiments()
