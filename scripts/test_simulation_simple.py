#!/usr/bin/env python3
"""
Simple test to verify simulation works after optimizations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation, EventSimulator
from config import DATA_SAMPLING

# Sampling parameters
NUM_BINS = DATA_SAMPLING["stratified_sampling_bins"]  # 12 bins
RANDOM_STATE = DATA_SAMPLING["random_state"]          # 42
NUM_WORKERS = 4000
NUM_TASKS = 20000

def load_and_sample_data():
    """Load full DiDi dataset without sampling."""
    print("Loading dataset...")
    # Use full_didi_gaia data (required for full day data files)
    data_path = "./data/didi/full_didi_gaia"
    
    # Get available day folders and use first day (same as tune_physics_full.py)
    day_folders = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])
    if not day_folders:
        raise FileNotFoundError(f"No day folders found in {data_path}")
    
    # Use first day consistently
    first_day = day_folders[0]
    day_path = os.path.join(data_path, first_day)
    print(f"   Using dataset: {first_day}")
    
    all_workers, all_tasks = load_workers_tasks('didi', root_path=day_path)
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks (full dataset, no sampling)")
    
    return all_workers, all_tasks

def test_basic_simulation():
    """Test basic simulation run with Config 1."""
    try:
        # Load and sample data
        workers, tasks = load_and_sample_data()
        
        # Config 1: (5.0, 0.5, 3.0)
        sim_config = {
            'assignment_strategy': 'composite',
            'strategy_params': {
                'fairness_weight': 5.0,
                'starvation_weight': 0.5,
                'utility_weight': 3.0,
                'soft_threshold': 0.0,
                'normalize_scores': True,
                'k': 15,
                'gamma': 0.5,
            }
        }
        
        # Dynamic print using actual config values
        params = sim_config['strategy_params']
        print("=" * 60)
        print(f"TEST: Basic Simulation Run (Config 1: λ₁={params['fairness_weight']}, λ₂={params['starvation_weight']}, λ₃={params['utility_weight']})")
        print("=" * 60)
        
        # Run simulation
        print("\nRunning simulation...")
        result = run_simulation(
            workers=workers,
            tasks=tasks,
            sim_config=sim_config
        )
        
        # Check results
        print("\n✅ Simulation completed successfully!")
        print(f"   Completed tasks: {result.get('completed_tasks', 0)}")
        print(f"   JFI: {result.get('final_jains_fairness_index', 0):.3f}")
        print(f"   Task Assignment Ratio: {result.get('task_assignment_ratio', 0):.3f}")
        print(f"   Peak Backlog: {result.get('backlog_peak', 0)}")
        print(f"   Expired Tasks: {len(result.get('expired_tasks', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_simulation_config2():
    """Test basic simulation run with Config 2."""
    try:
        # Load and sample data
        workers, tasks = load_and_sample_data()
        
        # Config 2: (4.0, 0.5, 4.0)
        sim_config = {
            'assignment_strategy': 'composite',
            'strategy_params': {
                'fairness_weight': 4.0,
                'starvation_weight': 0.5,
                'utility_weight': 4.0,
                'soft_threshold': 0,
                'normalize_scores': True,
                'k': 15,
                'gamma': 0.5,
            }
        }
        
        # Dynamic print using actual config values
        params = sim_config['strategy_params']
        print("\n" + "=" * 60)
        print(f"TEST: Basic Simulation Run (Config 2: λ₁={params['fairness_weight']}, λ₂={params['starvation_weight']}, λ₃={params['utility_weight']})")
        print("=" * 60)
        
        # Run simulation
        print("\nRunning simulation...")
        result = run_simulation(
            workers=workers,
            tasks=tasks,
            sim_config=sim_config
        )
        
        # Check results
        print("\n✅ Simulation completed successfully!")
        print(f"   Completed tasks: {result.get('completed_tasks', 0)}")
        print(f"   JFI: {result.get('final_jains_fairness_index', 0):.3f}")
        print(f"   Task Assignment Ratio: {result.get('task_assignment_ratio', 0):.3f}")
        print(f"   Peak Backlog: {result.get('backlog_peak', 0)}")
        print(f"   Expired Tasks: {len(result.get('expired_tasks', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_event_simulator():
    """Test EventSimulator (used by RL)."""
    try:
        # Load and sample data
        workers, tasks = load_and_sample_data()
        print(f"✅ Loaded {len(workers)} workers, {len(tasks)} tasks")
        
        # Config for EventSimulator test
        sim_config = {
            'assignment_strategy': 'composite',
            'strategy_params': {
                'fairness_weight': 5.0,
                'starvation_weight': 0.5,
                'utility_weight': 3,
                'soft_threshold': 0.0,
                'normalize_scores': True,
                'k': 15,
                'gamma': 0.5,
            }
        }
        
        # Dynamic print using actual config values
        params = sim_config['strategy_params']
        print("\n" + "=" * 60)
        print(f"TEST: EventSimulator (RL Interface) - Config: λ₁={params['fairness_weight']}, λ₂={params['starvation_weight']}, λ₃={params['utility_weight']}")
        print("=" * 60)
        
        # Create EventSimulator
        print("\nCreating EventSimulator...")
        sim = EventSimulator(workers, tasks, sim_config=sim_config)
        
        # Reset
        print("Resetting simulator...")
        state = sim.reset()
        print(f"✅ Simulator reset. Current time: {sim.current_time}")
        
        # Run a few steps
        print("\nRunning simulation steps...")
        for i in range(3):
            done = sim.step(duration_seconds=900)  # 15 minutes
            state = sim.get_state()
            # get_state() returns counts (ints), not collections
            active_count = state.get('active_tasks', 0)
            deferred_count = state.get('deferred_tasks', 0)
            print(f"   Step {i+1}: Time={sim.current_time}, "
                  f"Active={active_count}, "
                  f"Deferred={deferred_count}, "
                  f"Completed={sim.metrics.summary.get('completed_tasks', 0)}")
            if done:
                print("   Simulation completed!")
                break
        
        print("\n✅ EventSimulator test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ EventSimulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests."""
    print("\n🧪 Testing Simulation After Optimizations\n")
    print("Using full dataset (no sampling)")
    print("normalize_scores=True, γ=0.5\n")
    
    results = []
    
    # Test 1: Basic simulation (Config 1)
    results.append(("Basic Simulation (Config 1)", test_basic_simulation()))
    
    # Test 2: Basic simulation (Config 2)
    results.append(("Basic Simulation (Config 2)", test_basic_simulation_config2()))
    
    # Test 3: EventSimulator
    results.append(("EventSimulator", test_event_simulator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready for RL training.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before RL training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
