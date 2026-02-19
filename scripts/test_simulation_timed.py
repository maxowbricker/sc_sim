#!/usr/bin/env python3
"""
Test Simulation Script with Timing
===================================

Configuration:
- Lambda values: fairness_weight=5.0, starvation_weight=0.5, utility_weight=3.0
- soft_threshold: OFF (disable_soft_threshold=True)
- diagnostics: OFF (enable_diagnostics=False)
- normalization: ON (normalize_scores=True)
- Random day from full_didi_gaia dataset
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
from pathlib import Path
from datetime import datetime
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

# ============================================================================
# CONFIGURATION
# ============================================================================

# Lambda values
FAIRNESS_WEIGHT = 5.0
STARVATION_WEIGHT = 0.5
UTILITY_WEIGHT = 3.0

# Strategy parameters
STRATEGY_PARAMS = {
    'fairness_weight': FAIRNESS_WEIGHT,
    'starvation_weight': STARVATION_WEIGHT,
    'utility_weight': UTILITY_WEIGHT,
    'soft_threshold': 0.0,  # Not used when disable_soft_threshold=True
    'disable_soft_threshold': True,  # Threshold OFF
    'normalize_scores': True,  # Normalization ON
    'enable_diagnostics': False,  # Diagnostics OFF
    'enable_deferral_tracking': False,
    'k': 15,
    'gamma': 0.3,
    'fairness_metric': 'ewma',
}

# Simulation config
SIM_CONFIG = {
    'assignment_strategy': 'composite',
    'strategy_params': STRATEGY_PARAMS,
    'dataset': 'didi',
}

# ============================================================================
# DATA LOADING
# ============================================================================

def get_random_day_path():
    """Select a random day folder from full_didi_gaia."""
    project_root = Path(__file__).resolve().parent.parent
    full_didi_path = project_root / "data" / "didi" / "full_didi_gaia"
    
    if not full_didi_path.exists():
        raise FileNotFoundError(f"full_didi_gaia directory not found at {full_didi_path}")
    
    # Get all day folders
    day_folders = [d for d in full_didi_path.iterdir() if d.is_dir()]
    
    if not day_folders:
        raise FileNotFoundError(f"No day folders found in {full_didi_path}")
    
    # Select random day
    selected_day = random.choice(day_folders)
    print(f"📅 Selected random day: {selected_day.name}")
    
    return str(selected_day)

def load_data():
    """Load workers and tasks from a random day."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    day_path = get_random_day_path()
    
    print(f"📂 Loading from: {day_path}")
    workers, tasks = load_workers_tasks('didi', day_path)
    
    print(f"✅ Loaded: {len(workers)} workers, {len(tasks)} tasks")
    
    return workers, tasks

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run timed simulation test."""
    print("\n" + "=" * 80)
    print("TEST SIMULATION - TIMED RUN")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print(f"  Lambda values: λ₁={FAIRNESS_WEIGHT}, λ₂={STARVATION_WEIGHT}, λ₃={UTILITY_WEIGHT}")
    print(f"  soft_threshold: OFF (disable_soft_threshold=True)")
    print(f"  diagnostics: OFF (enable_diagnostics=False)")
    print(f"  normalization: ON (normalize_scores=True)")
    print("=" * 80)
    
    # Load data
    workers, tasks = load_data()
    
    # Run simulation with timing
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = run_simulation(
            workers=workers,
            tasks=tasks,
            sim_config=SIM_CONFIG
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Print timing results
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"⏱️  Total Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Print key metrics
        print("\nKey Results:")
        completed = result.get('completed_tasks', 0)
        total = result.get('total_tasks', len(tasks))
        tar = completed / total if total > 0 else 0
        jfi = result.get('final_jains_fairness_index', 0)
        
        print(f"  ✅ Completed Tasks: {completed:,}/{total:,} ({tar*100:.1f}%)")
        print(f"  📊 JFI: {jfi:.3f}")
        
        if 'total_wait_min' in result and completed > 0:
            avg_wait = result['total_wait_min'] / completed
            print(f"  ⏳ Avg Wait Time: {avg_wait:.2f} minutes")
        
        if 'total_travel_km' in result:
            print(f"  🚗 Total Travel: {result['total_travel_km']:.2f} km")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n❌ ERROR after {elapsed_time:.2f} seconds: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()

