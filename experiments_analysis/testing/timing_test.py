#!/usr/bin/env python3
"""
Simple Timing Test for Full Dataset Simulation
Quick test to measure actual performance before committing to long experiments.
"""

import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_timing_test(dataset_size="small"):
    """Run a timing test with different dataset sizes."""
    
    print("⏱️  SIMULATION TIMING TEST")
    print("=" * 50)
    print(f"🎯 Testing dataset size: {dataset_size}")
    print(f"🖥️  Running on: {os.name} system")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import simulation components
    try:
        from config import create_composite_config
        from simulator.simulation import Simulation
        from data.notebook_optimized_loader import load_data
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        print("💡 And that you've installed all dependencies: pip install pandas numpy matplotlib seaborn")
        return None
    
    # Configure dataset size
    if dataset_size == "small":
        max_tasks, max_workers = 1000, 500
        description = "Small test (1K tasks)"
    elif dataset_size == "medium":  
        max_tasks, max_workers = 15000, 7500
        description = "Medium test (15K tasks)"
    elif dataset_size == "full":
        max_tasks, max_workers = None, None
        description = "Full dataset (220K+ tasks)"
    else:
        print(f"❌ Unknown dataset size: {dataset_size}")
        return None
    
    print(f"📊 Dataset configuration: {description}")
    print()
    
    # Step 1: Time data loading
    print("🚀 Step 1: Loading dataset...")
    load_start = time.time()
    
    try:
        workers_df, tasks_df = load_data(
            'didi',
            max_workers=max_workers,
            max_tasks=max_tasks
        )
        load_time = time.time() - load_start
        
        print(f"✅ Dataset loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
        print(f"⏱️  Loading time: {load_time:.2f} seconds")
        print()
        
    except Exception as e:
        print(f"❌ Loading error: {e}")
        return None
    
    # Step 2: Time simulation setup
    print("🧪 Step 2: Setting up simulation...")
    setup_start = time.time()
    
    # Use baseline parameters
    sim_config = create_composite_config(
        fairness_weight=1.5,
        starvation_weight=1.0,
        utility_weight=0.8,
        soft_threshold=0.5,
        assignment_strategy="composite"
    )
    
    sim = Simulation(sim_config, workers_df, tasks_df)
    setup_time = time.time() - setup_start
    
    print(f"✅ Simulation configured")
    print(f"⏱️  Setup time: {setup_time:.2f} seconds")
    print()
    
    # Step 3: Time the actual simulation
    print("🏃 Step 3: Running simulation...")
    print("⚠️  This is the main performance test - timing starts now!")
    print()
    
    sim_start = time.time()
    
    try:
        results = sim.run()
        sim_time = time.time() - sim_start
        
        print("✅ Simulation completed successfully!")
        print()
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return None
    
    # Calculate total time
    total_time = time.time() - load_start
    
    # Display comprehensive timing results
    print("📊 TIMING RESULTS")
    print("=" * 40)
    print(f"Loading time:    {load_time:>8.2f} seconds ({load_time/60:>5.2f} minutes)")
    print(f"Setup time:      {setup_time:>8.2f} seconds ({setup_time/60:>5.2f} minutes)")
    print(f"Simulation time: {sim_time:>8.2f} seconds ({sim_time/60:>5.2f} minutes)")
    print(f"Total time:      {total_time:>8.2f} seconds ({total_time/60:>5.2f} minutes)")
    print()
    
    # Performance metrics
    tasks_per_second = len(tasks_df) / sim_time if sim_time > 0 else 0
    print(f"📈 PERFORMANCE METRICS")
    print("=" * 30)
    print(f"Tasks processed: {len(tasks_df):,}")
    print(f"Processing rate: {tasks_per_second:,.0f} tasks/second")
    print(f"Processing rate: {tasks_per_second * 60:,.0f} tasks/minute")
    print()
    
    # Simulation results summary
    print(f"🎯 SIMULATION RESULTS")  
    print("=" * 25)
    print(f"JFI:                    {results.get('jfi', 0):.3f}")
    print(f"Task completion rate:   {results.get('task_assignment_ratio', 0)*100:.1f}%")
    print(f"Completed tasks:        {results.get('completed_tasks', 0):,}")
    print(f"Average wait time:      {results.get('avg_wait_time_minutes', 0):.2f} (raw units)")
    print(f"Average pickup distance: {results.get('avg_pickup_distance_km', 0):.2f} km")
    print()
    
    # Extrapolation to full dataset
    if dataset_size != "full":
        full_dataset_tasks = 220139
        scale_factor = full_dataset_tasks / len(tasks_df)
        estimated_full_time = sim_time * scale_factor
        
        print(f"🔮 FULL DATASET EXTRAPOLATION")
        print("=" * 35)
        print(f"Current tasks:       {len(tasks_df):,}")
        print(f"Full dataset tasks:  {full_dataset_tasks:,}")
        print(f"Scale factor:        {scale_factor:.1f}x")
        print(f"Estimated full time: {estimated_full_time:.1f} seconds ({estimated_full_time/60:.1f} minutes)")
        
        if estimated_full_time > 3600:
            print(f"                     {estimated_full_time/3600:.1f} hours")
    
    # Save timing results using JSON utilities
    try:
        from json_utils import save_results_json
    except ImportError:
        # Fallback if json_utils not available
        def save_results_json(data, filepath):
            import json
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Convert numpy types manually
            def convert(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                else:
                    return obj
            
            with open(filepath, 'w') as f:
                json.dump(convert(data), f, indent=2)
    
    timing_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': dataset_size,
        'workers_count': len(workers_df),
        'tasks_count': len(tasks_df),
        'loading_time_seconds': load_time,
        'setup_time_seconds': setup_time,
        'simulation_time_seconds': sim_time,
        'total_time_seconds': total_time,
        'tasks_per_second': tasks_per_second,
        'simulation_results': results
    }
    
    # Save with automatic type conversion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'../../../../results/timing_test_{dataset_size}_{timestamp}.json'
    
    save_results_json(timing_results, results_file)
    
    print(f"📄 Detailed results saved to: {results_file}")
    print()
    print("🎉 Timing test complete!")
    
    return timing_results

if __name__ == "__main__":
    # Allow user to specify dataset size
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
    else:
        size = input("Choose dataset size (small/medium/full): ").lower()
        if not size:
            size = "small"
    
    if size not in ["small", "medium", "full"]:
        print("❌ Invalid size. Choose 'small', 'medium', or 'full'")
        sys.exit(1)
    
    run_timing_test(size)
