#!/usr/bin/env python3
"""
Simple Direct Timing Test - Uses Full Dataset Files Directly
No complex loaders, just direct file access to order.txt and gps.txt
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_didi_direct(data_dir, max_tasks=None):
    """Load DiDi data directly from order.txt and gps.txt files."""
    
    data_path = Path(data_dir)
    
    # Check for required files
    order_file = data_path / "order.txt"
    gps_file = data_path / "gps.txt"
    
    print(f"📁 Data directory: {data_path.absolute()}")
    print(f"📄 Order file: {order_file} (exists: {order_file.exists()})")  
    print(f"📄 GPS file: {gps_file} (exists: {gps_file.exists()})")
    
    if not order_file.exists():
        print(f"❌ ERROR: order.txt not found at {order_file}")
        print(f"💡 Available files in {data_path}:")
        for f in data_path.glob("*.txt"):
            print(f"   - {f.name}")
        return None, None
    
    if not gps_file.exists():
        print(f"❌ ERROR: gps.txt not found at {gps_file}")
        print(f"💡 Available files in {data_path}:")
        for f in data_path.glob("*.txt"):
            print(f"   - {f.name}")
        return None, None
    
    print("✅ Both required files found")
    print()
    
    # Load orders (tasks)
    print("📊 Loading orders (tasks)...")
    order_start = time.time()
    
    # Read with proper column names
    order_cols = ["order_id", "start_billing", "end_billing", "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"]
    
    if max_tasks:
        orders_df = pd.read_csv(order_file, header=None, names=order_cols, nrows=max_tasks)
    else:
        orders_df = pd.read_csv(order_file, header=None, names=order_cols)
    
    # Convert timestamps
    orders_df["start_billing"] = pd.to_datetime(orders_df["start_billing"], unit="s")
    orders_df["end_billing"] = pd.to_datetime(orders_df["end_billing"], unit="s")
    
    # Create task format
    tasks_df = pd.DataFrame({
        'task_id': orders_df['order_id'],
        'pickup_lat': orders_df['pickup_lat'],
        'pickup_lon': orders_df['pickup_lon'], 
        'dropoff_lat': orders_df['dropoff_lat'],
        'dropoff_lon': orders_df['dropoff_lon'],
        'release_time': orders_df['start_billing'],
        'expire_time': orders_df['start_billing'] + pd.Timedelta('2h')
    })
    
    order_time = time.time() - order_start
    print(f"✅ Loaded {len(tasks_df):,} tasks in {order_time:.1f} seconds")
    
    # Load GPS data for workers (sample only for timing)
    print("📍 Loading GPS data (workers)...")
    gps_start = time.time()
    
    # For timing test, just load enough GPS data to create workers
    gps_cols = ["driver_id", "order_id", "timestamp", "lon", "lat"]
    gps_sample = pd.read_csv(gps_file, header=None, names=gps_cols, nrows=50000)  # Sample for speed
    
    # Convert timestamp
    gps_sample["timestamp"] = pd.to_datetime(gps_sample["timestamp"], unit="s")
    
    # Create workers from first GPS ping per driver
    workers_df = gps_sample.groupby("driver_id").first().reset_index()
    workers_df = pd.DataFrame({
        'worker_id': workers_df['driver_id'],
        'start_lat': workers_df['lat'],
        'start_lon': workers_df['lon'],
        'release_time': workers_df['timestamp'],
        'deadline': workers_df['timestamp'] + pd.Timedelta('8h')  # 8 hour work shift
    })
    
    gps_time = time.time() - gps_start
    print(f"✅ Created {len(workers_df):,} workers in {gps_time:.1f} seconds")
    print()
    
    return workers_df, tasks_df

def run_simple_timing_test(dataset_size="small"):
    """Run a simple timing test."""
    
    print("⚡ SIMPLE TIMING TEST - DIRECT FILE ACCESS")
    print("=" * 55)
    print(f"🎯 Dataset size: {dataset_size}")
    print(f"📅 Start time: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Determine task count
    task_limits = {
        "small": 1000,
        "medium": 15000, 
        "large": 50000,
        "full": None
    }
    
    max_tasks = task_limits.get(dataset_size, 1000)
    print(f"📊 Will load {max_tasks if max_tasks else 'ALL'} tasks")
    print()
    
    # Find data directory (works on both Windows and Mac)
    possible_paths = [
        "data/didi",           # From project root
        "../data/didi",        # From experiments folder
        "../../data/didi",     # Just in case
    ]
    
    data_dir = None
    for path in possible_paths:
        if Path(path).exists():
            data_dir = path
            break
    
    if not data_dir:
        print("❌ ERROR: Cannot find data/didi directory")
        print("💡 Make sure you're running from the project root or experiments folder")
        print("💡 Current directory:", os.getcwd())
        return None
    
    # Load data
    total_start = time.time()
    workers_df, tasks_df = load_didi_direct(data_dir, max_tasks)
    
    if workers_df is None:
        print("❌ Data loading failed")
        return None
    
    # Import simulation components
    print("🔧 Setting up simulation...")
    try:
        from config import create_composite_config
        from simulator.simulation import Simulation
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root")
        return None
    
    # Create simulation
    setup_start = time.time()
    sim_config = create_composite_config(
        fairness_weight=1.5,
        starvation_weight=1.0,
        utility_weight=0.8,
        soft_threshold=0.5,
        assignment_strategy="composite"
    )
    
    sim = Simulation(sim_config, workers_df, tasks_df)
    setup_time = time.time() - setup_start
    print(f"✅ Simulation ready ({setup_time:.1f}s)")
    print()
    
    # Run simulation
    print("🚀 RUNNING SIMULATION - TIMING STARTS NOW!")
    print("-" * 45)
    sim_start = time.time()
    
    try:
        results = sim.run()
        sim_time = time.time() - sim_start
        print("✅ SIMULATION COMPLETE!")
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return None
    
    # Calculate times
    total_time = time.time() - total_start
    
    # Results
    print()
    print("🎯 TIMING RESULTS")
    print("=" * 20)
    print(f"Simulation time: {sim_time:>8.1f} seconds ({sim_time/60:>5.1f} minutes)")
    print(f"Total time:      {total_time:>8.1f} seconds ({total_time/60:>5.1f} minutes)")
    print()
    
    # Performance metrics
    tasks_count = len(tasks_df)
    rate = tasks_count / sim_time if sim_time > 0 else 0
    
    print("📊 PERFORMANCE")
    print("=" * 15)
    print(f"Tasks processed: {tasks_count:,}")
    print(f"Processing rate: {rate:>8.0f} tasks/second")
    print(f"               : {rate*60:>8.0f} tasks/minute")
    print()
    
    # Simulation results
    print("🎲 SIMULATION RESULTS") 
    print("=" * 20)
    print(f"JFI:              {results.get('jfi', 0):>6.3f}")
    print(f"Completion rate:  {results.get('task_assignment_ratio', 0)*100:>6.1f}%")
    print(f"Avg wait time:    {results.get('avg_wait_time_minutes', 0):>6.2f}")
    print(f"Avg distance:     {results.get('avg_pickup_distance_km', 0):>6.2f} km")
    print()
    
    # Extrapolation
    if max_tasks and max_tasks < 220139:
        full_estimate = (220139 / tasks_count) * sim_time
        print("🔮 FULL DATASET ESTIMATE")
        print("=" * 25)
        print(f"Full dataset: 220,139 tasks")
        print(f"Scale factor: {220139/tasks_count:>5.1f}x")  
        print(f"Estimated time: {full_estimate:>5.0f} seconds ({full_estimate/60:>5.0f} minutes)")
        if full_estimate > 3600:
            print(f"              : {full_estimate/3600:>5.1f} hours")
    
    return {
        'dataset_size': dataset_size,
        'tasks_count': tasks_count,
        'workers_count': len(workers_df),
        'simulation_time_seconds': sim_time,
        'tasks_per_second': rate,
        'results': results
    }

if __name__ == "__main__":
    # Get dataset size from command line or user input
    if len(sys.argv) > 1:
        size = sys.argv[1].lower()
    else:
        print("Available sizes: small (1K), medium (15K), large (50K), full (220K)")
        size = input("Choose dataset size: ").lower().strip()
        if not size:
            size = "small"
    
    valid_sizes = ["small", "medium", "large", "full"]
    if size not in valid_sizes:
        print(f"❌ Invalid size '{size}'. Choose from: {', '.join(valid_sizes)}")
        sys.exit(1)
    
    result = run_simple_timing_test(size)
    
    if result:
        print("\n🎉 TIMING TEST COMPLETE!")
        print(f"💾 Run this test with different sizes to compare performance")
    else:
        print("\n❌ TIMING TEST FAILED!")
        sys.exit(1)
