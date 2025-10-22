#!/usr/bin/env python3
"""
Quick benchmark: Single full-scale simulation with diagnostics + normalization.
This tests the actual performance of the slow path before committing to 12 experiments.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data

def benchmark_single_run():
    print("=" * 80)
    print("BENCHMARK: Single Full-Scale Simulation with Diagnostics + Normalization")
    print("=" * 80)
    print()
    
    # Load FULL dataset
    print("[1/3] Loading dataset (15,000 workers, 20,000 tasks)...")
    start_load = time.time()
    workers_df, tasks_df = load_data('didi', max_workers=15000, max_tasks=20000)
    load_time = time.time() - start_load
    print(f"   ✅ Loaded: {len(workers_df):,} workers, {len(tasks_df):,} tasks")
    print(f"   ⏱️  Load time: {load_time/60:.1f} minutes")
    print()
    
    # Configure with normalization + diagnostics (SLOW PATH)
    print("[2/3] Running simulation with SLOW PATH...")
    print("   normalize_scores=True, enable_diagnostics=True")
    config = create_composite_config(
        fairness_weight=0.5,
        starvation_weight=0.8,
        utility_weight=0.8,
        soft_threshold=0.5,
        gamma=0.5,
        normalize_scores=True,        # Forces slow path
        disable_soft_threshold=False,
        enable_diagnostics=True       # Enables diagnostic tracking
    )
    
    start_sim = time.time()
    sim = Simulation(config, workers_df, tasks_df)
    summary = sim.run()
    sim_time = time.time() - start_sim
    
    print(f"   ✅ Simulation complete!")
    print(f"   ⏱️  Simulation time: {sim_time/60:.1f} minutes")
    print()
    
    # Results
    print("[3/3] Results:")
    print(f"   📊 Completed tasks: {summary.get('completed_tasks', 0):,}")
    print(f"   📊 Task Assignment Ratio: {summary.get('completed_tasks', 0)/20000:.1%}")
    print(f"   📊 JFI: {summary.get('final_jains_fairness_index', 0):.3f}")
    
    if 'diagnostic_tracker' in summary and summary['diagnostic_tracker']:
        stats = summary['diagnostic_tracker'].get_summary_stats()
        print(f"   📊 Diagnostic assignments: {stats['total_assignments']:,}")
        print(f"   📊 Deferral rate: {stats['deferral_rate']*100:.1f}%")
    
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total time: {(load_time + sim_time)/60:.1f} minutes")
    print(f"  - Data loading: {load_time/60:.1f} min")
    print(f"  - Simulation: {sim_time/60:.1f} min")
    print()
    print("Extrapolation for full Experiment 008:")
    print(f"  - 3 Greedy runs: ~{3 * sim_time * 0.6 / 60:.0f} min (faster, no fairness)")
    print(f"  - 9 Composite runs: ~{9 * sim_time / 60:.0f} min (Groups B, C, D)")
    print(f"  - Total estimated: ~{(3 * sim_time * 0.6 + 9 * sim_time) / 60:.0f} minutes ({(3 * sim_time * 0.6 + 9 * sim_time) / 3600:.1f} hours)")
    print()
    
    if sim_time / 60 > 30:
        print("⚠️  WARNING: Simulation took > 30 minutes!")
        print("   Full experiment (12 runs) would take > 6 hours")
        print("   Consider disabling diagnostics for Group B to speed up")
    elif sim_time / 60 > 20:
        print("⚠️  CAUTION: Simulation took > 20 minutes")
        print("   Full experiment would take 4-5 hours")
    else:
        print("✅ Good performance! Full experiment should complete in reasonable time")

if __name__ == "__main__":
    benchmark_single_run()

