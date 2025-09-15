#!/usr/bin/env python3
"""
Full Dataset Performance Analysis & Estimation Tool
Estimates computational requirements for full DiDi dataset experiments.
"""

import json
import math
from datetime import datetime, timedelta

def analyze_full_dataset_requirements():
    """Analyze computational requirements for full dataset experiments."""
    
    print("🔬 FULL DATASET COMPUTATIONAL ANALYSIS")
    print("=" * 60)
    
    # Dataset characteristics
    full_dataset_tasks = 220139
    current_test_tasks = 15000  # standard mode
    current_test_time_hours = 2.0  # based on recent experience
    
    # Your PC specs
    pc_specs = {
        "cpu": "Intel i7-10700K @ 3.8GHz",
        "cores": 8,
        "threads": 16,
        "ram_gb": 16,
        "performance_multiplier": 2.5  # vs M1 MacBook for sustained workloads
    }
    
    print(f"📊 DATASET SCALE ANALYSIS:")
    print(f"   Full dataset: {full_dataset_tasks:,} tasks")
    print(f"   Current test: {current_test_tasks:,} tasks")  
    print(f"   Scale multiplier: {full_dataset_tasks/current_test_tasks:.1f}x")
    print()
    
    print(f"💻 YOUR PC SPECS:")
    print(f"   CPU: {pc_specs['cpu']}")
    print(f"   Cores/Threads: {pc_specs['cores']}/{pc_specs['threads']}")
    print(f"   RAM: {pc_specs['ram_gb']}GB DDR4")
    print(f"   Performance vs M1: ~{pc_specs['performance_multiplier']}x faster")
    print()
    
    # Time estimates for different experiment types
    experiments = {
        "single_simulation": {
            "description": "One parameter configuration",
            "count": 1,
            "purpose": "Test full dataset performance"
        },
        "quick_sweep": {
            "description": "Quick parameter test", 
            "count": 36,  # 3x3x2x2 from quick mode
            "purpose": "Validate full dataset approach"
        },
        "standard_sweep": {
            "description": "Standard parameter sweep",
            "count": 162,  # from standard mode 
            "purpose": "Comprehensive but reasonable"
        },
        "comprehensive_sweep": {
            "description": "Full comprehensive sweep",
            "count": 2401,  # 7x7x7x7 from fine mode
            "purpose": "Complete parameter space exploration"
        }
    }
    
    print(f"⏱️  TIME ESTIMATES FOR FULL DATASET:")
    print("-" * 50)
    
    # Base time per simulation (scaled from current experience)
    scale_factor = full_dataset_tasks / current_test_tasks
    base_time_hours = current_test_time_hours * scale_factor / pc_specs['performance_multiplier']
    
    print(f"Estimated time per simulation: {base_time_hours:.1f} hours")
    print(f"(Based on: {current_test_time_hours:.1f}h for {current_test_tasks:,} tasks, scaled for {pc_specs['performance_multiplier']}x PC performance)")
    print()
    
    for exp_name, exp_config in experiments.items():
        total_time_hours = base_time_hours * exp_config['count']
        total_time_days = total_time_hours / 24
        
        print(f"📋 {exp_config['description'].upper()}:")
        print(f"   Simulations: {exp_config['count']:,}")
        print(f"   Total time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
        print(f"   Purpose: {exp_config['purpose']}")
        
        # Feasibility assessment
        if total_time_days <= 1:
            feasibility = "✅ EXCELLENT - Weekend project"
        elif total_time_days <= 3:
            feasibility = "🔶 GOOD - Long weekend"  
        elif total_time_days <= 7:
            feasibility = "⚠️  CHALLENGING - Full week"
        else:
            feasibility = "❌ IMPRACTICAL - Multiple weeks"
            
        print(f"   Feasibility: {feasibility}")
        print()
    
    # Memory requirements
    print(f"💾 MEMORY REQUIREMENTS:")
    print("-" * 30)
    
    # Rough estimates based on typical simulation data structures
    tasks_memory_mb = full_dataset_tasks * 0.001  # ~1KB per task object
    workers_memory_mb = 20000 * 0.001  # ~20K workers estimated  
    simulation_overhead_mb = 500  # Pandas dataframes, tracking, etc.
    total_memory_mb = tasks_memory_mb + workers_memory_mb + simulation_overhead_mb
    
    print(f"   Tasks: ~{tasks_memory_mb:.0f} MB")
    print(f"   Workers: ~{workers_memory_mb:.0f} MB") 
    print(f"   Simulation overhead: ~{simulation_overhead_mb:.0f} MB")
    print(f"   Total per simulation: ~{total_memory_mb:.0f} MB")
    print(f"   Available RAM: {pc_specs['ram_gb']*1024:.0f} MB")
    
    if total_memory_mb < pc_specs['ram_gb'] * 1024 * 0.8:  # 80% of RAM
        memory_status = "✅ SUFFICIENT - No memory concerns"
    else:
        memory_status = "⚠️  TIGHT - Monitor memory usage"
        
    print(f"   Status: {memory_status}")
    print()
    
    # Recommendations
    print(f"🎯 STRATEGIC RECOMMENDATIONS:")
    print("-" * 40)
    
    print(f"1. 🧪 START WITH SINGLE SIMULATION:")
    print(f"   - Run one full dataset simulation (~{base_time_hours:.1f} hours)")
    print(f"   - Validate performance and results quality")
    print(f"   - Ensure temporal consistency benefits are realized")
    print()
    
    print(f"2. 🔬 THEN PROCEED TO QUICK SWEEP:")
    print(f"   - 36 simulations (~{base_time_hours * 36:.1f} hours / {base_time_hours * 36 / 24:.1f} days)")
    print(f"   - Tests key parameter combinations")
    print(f"   - Validates full-dataset parameter sweep approach")
    print()
    
    print(f"3. 📊 RESEARCH STRATEGY:")
    print(f"   - Full dataset gives you publication-quality robustness")
    print(f"   - Temporal consistency addresses key research validity concern")
    print(f"   - Your PC can handle standard sweep ({162 * base_time_hours / 24:.1f} days)")
    print()
    
    print(f"4. ⚡ OPTIMIZATION OPPORTUNITIES:")
    print(f"   - Parallel processing: Use all 16 threads efficiently")
    print(f"   - Early termination: Stop poor configs early")
    print(f"   - Checkpointing: Save intermediate results")
    print(f"   - Parameter space reduction: Focus on promising regions")
    
    return {
        "base_time_hours": base_time_hours,
        "experiments": experiments,
        "memory_mb": total_memory_mb,
        "pc_specs": pc_specs
    }

if __name__ == "__main__":
    results = analyze_full_dataset_requirements()
    
    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../results/full_dataset_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Analysis saved to: {output_file}")
