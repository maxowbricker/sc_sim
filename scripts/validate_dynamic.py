#!/usr/bin/env python3
"""
validate_dynamic.py

Validates if dynamic weighting strategies have potential to beat static baselines.
Implements the "Bilateral Control Loop" heuristic.

This script runs two simulations:
1. Static Baseline: Fixed weights throughout
2. Dynamic Heuristic: Weights adjusted based on system state

The goal is to prove that dynamic control has potential before investing in RL training.

Updated for current codebase (Jun 2026):
- Uses current MetricsManager interface
- Compatible with revised config and strategy_params
- Uses get_final_results() properly
"""

import os
import sys
# Add project root to path (parent of scripts/ directory)
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(project_root))

import numpy as np
from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

# --- CONFIG ---
# Update this to point to your data folder
DATA_PATH = "./data/didi/full_didi_gaia"
STEP_MINUTES = 5  # Updated to current step duration

# Test mode: Set to True to run only 2 hours of data (for quick validation)
# Set to False to run full dataset (~16 hours)
TEST_MODE = False
TEST_DURATION_HOURS = 2.0

# --- RANGES (From User Definition) ---
# Starvation: Scale from 0.25 (Base) to 0.5 (Max) based on backlog
STARVATION_MIN = 0.25
STARVATION_MAX = 0.5

# Fairness: Scale from 2.0 (Max) down to 0.0 (Min) based on busy ratio
FAIRNESS_MIN = 0.0
FAIRNESS_MAX = 2.0

def get_sim_stats(sim):
    """
    Helper to extract current system state using MetricsManager.
    Uses MetricsManager's interface exclusively.
    """
    # Use MetricsManager's observation data (clean interface)
    obs_data = sim.metrics.get_observation_data(sim.state, sim.current_time)
    
    # 1. Current Backlog - Get from MetricsManager's current_step_stats
    current_backlog = sim.metrics.current_step_stats.get('backlog', 0)
    
    # 2. Worker Utilization (Busy / Total)
    available_workers = obs_data.get('worker_availability_ratio', 0.0)
    total_workers = obs_data.get('total_workers', 1)
    utilization = 1.0 - available_workers  # 1 - (available/total) = busy/total
    
    return current_backlog, utilization

def run_static_baseline(workers, tasks, test_mode=False, test_duration_hours=2.0):
    """
    Run 1: Static parameters.
    Goal: Measure the 'Peak' values to normalize against.
    
    Args:
        test_mode: If True, limit simulation to test_duration_hours
        test_duration_hours: Duration in hours for test mode
    """
    print("\n🔹 Running STATIC Baseline (to establish peaks)...")
    if test_mode:
        print(f"   ⚠️  TEST MODE: Limiting to {test_duration_hours} hours of data")
    
    config = get_simulation_config()
    config['assignment_strategy'] = 'composite'
    
    # Fixed baseline weights from current config
    strategy_params = get_strategy_params('composite')
    strategy_params.update({
        'enable_deferral_tracking': True,
        'enable_diagnostics': False,  # Disable diagnostics for performance
    })
    config['strategy_params'] = strategy_params
    
    sim = EventSimulator(workers, tasks, sim_config=config)
    
    # Set end_time if in test mode
    end_time = None
    if test_mode:
        # Find earliest task/worker release time
        earliest = min(
            min((t.release_time for t in tasks), default=float('inf')),
            min((w.release_time for w in workers), default=float('inf'))
        )
        if earliest != float('inf'):
            end_time = earliest + (test_duration_hours * 3600)  # Convert hours to seconds
            print(f"   📅 Start: {earliest}, End: {end_time} ({test_duration_hours}h window)")
    
    sim.reset(start_time=None, end_time=end_time)
    
    # Track utilization over time to find MIN and MAX range
    utilization_history = []
    
    step_duration = STEP_MINUTES * 60
    done = False
    
    step_count = 0
    while not done:
        # Run the step first (process events for this step)
        done = sim.step(duration_seconds=step_duration)
        sim.metrics.snapshot_step(sim.state, sim.current_time, step_start_time=sim.current_time - step_duration)
        step_count += 1
        
        # Get current utilization from MetricsManager (for range calculation)
        _, u = get_sim_stats(sim)
        utilization_history.append(u)
        
        if step_count % 20 == 0:
            # Get current backlog from MetricsManager
            b, _ = get_sim_stats(sim)
            print(f"   Step {step_count}: Backlog={b}, Utilization={u:.2%}")
        
    # Use get_final_results() to get properly calculated metrics
    results = sim.get_final_results()
    
    # Use MetricsManager's backlog_peak
    metrics_backlog_peak = results.get('backlog_peak', 0)
    
    # Calculate MIN and MAX utilization from our tracking
    min_utilization = np.min(utilization_history) if utilization_history else 0.0
    max_utilization = np.max(utilization_history) if utilization_history else 0.0
    
    peaks = {
        'peak_backlog': metrics_backlog_peak,
        'min_utilization': min_utilization,
        'max_utilization': max_utilization
    }
    
    print(f"   ✅ Completed {step_count} steps")
    print(f"   📊 Peak Backlog (from MetricsManager): {peaks['peak_backlog']:.0f}")
    print(f"   📊 Utilization Range: {peaks['min_utilization']:.2%} - {peaks['max_utilization']:.2%}")
    
    return results, peaks

def run_dynamic_heuristic(workers, tasks, peaks, test_mode=False, test_duration_hours=2.0):
    """
    Run 2: Dynamic Heuristic.
    Goal: Beat the static baseline using the Bilateral Control Loops.
    
    Control Loop 1 (Starvation):
    - Backlog is X% of peak -> Starvation is X% of range [0.25, 0.5]
    
    Control Loop 2 (Fairness):
    - Busy ratio is X% of peak -> Fairness is (100-X)% of range [0.0, 2.0]
    
    Args:
        test_mode: If True, limit simulation to test_duration_hours
        test_duration_hours: Duration in hours for test mode
    """
    print("\n🔸 Running DYNAMIC Heuristic...")
    print(f"   (Normalizing against Peak Backlog: {peaks['peak_backlog']:.0f})")
    print(f"   (Utilization Range: {peaks['min_utilization']:.2%} - {peaks['max_utilization']:.2%})")
    if test_mode:
        print(f"   ⚠️  TEST MODE: Limiting to {test_duration_hours} hours of data")
    
    config = get_simulation_config()
    config['assignment_strategy'] = 'composite'
    
    # Initial weights from current config
    strategy_params = get_strategy_params('composite')
    strategy_params.update({
        'enable_deferral_tracking': True,
        'enable_diagnostics': False,  # Disable diagnostics for performance
    })
    config['strategy_params'] = strategy_params
    
    sim = EventSimulator(workers, tasks, sim_config=config)
    
    # Set end_time if in test mode
    end_time = None
    if test_mode:
        # Find earliest task/worker release time
        earliest = min(
            min((t.release_time for t in tasks), default=float('inf')),
            min((w.release_time for w in workers), default=float('inf'))
        )
        if earliest != float('inf'):
            end_time = earliest + (test_duration_hours * 3600)  # Convert hours to seconds
            print(f"   📅 Start: {earliest}, End: {end_time} ({test_duration_hours}h window)")
    
    sim.reset(start_time=None, end_time=end_time)
    
    step_duration = STEP_MINUTES * 60
    done = False
    
    # For logging what the heuristic did
    history_lambda1 = []
    history_lambda2 = []
    history_backlog = []
    history_utilization = []
    
    step_count = 0
    while not done:
        # 1. OBSERVE (at start of step, before processing)
        # Use MetricsManager to get current state
        current_backlog, current_utilization = get_sim_stats(sim)
        
        # 2. CALCULATE RATIOS (Normalized signals)
        # Backlog ratio: clamp at 1.0 if we exceed historical peak
        backlog_ratio = min(current_backlog / peaks['peak_backlog'], 1.0) if peaks['peak_backlog'] > 0 else 0
        
        # 3. CONTROL LOOP 1: STARVATION (Proportional to Backlog)
        # "Backlog is 70% of peak -> Starvation is 70% of range [0.25, 0.5]"
        starvation_range = STARVATION_MAX - STARVATION_MIN
        new_lambda2 = STARVATION_MIN + (backlog_ratio * starvation_range)
        
        # 4. CONTROL LOOP 2: FAIRNESS (Mapped to Utilization Range)
        min_util = peaks['min_utilization']
        max_util = peaks['max_utilization']
        util_range = max_util - min_util
        
        if util_range > 0:
            if current_utilization < min_util:
                # Below range -> max fairness
                new_lambda1 = FAIRNESS_MAX  # 2.0
            elif current_utilization > max_util:
                # Above range -> min fairness
                new_lambda1 = FAIRNESS_MIN  # 0.0
            else:
                # Within range -> linear interpolation
                util_normalized = (current_utilization - min_util) / util_range  # 0.0 to 1.0
                fairness_range = FAIRNESS_MAX - FAIRNESS_MIN  # 2.0 - 0.0 = 2.0
                # Inverse: high utilization -> low fairness
                new_lambda1 = FAIRNESS_MAX - (util_normalized * fairness_range)
        else:
            # No range variation -> use default
            new_lambda1 = 1.0
        
        # 5. ACT (update weights before processing this step)
        sim.update_weights(fairness_weight=new_lambda1, starvation_weight=new_lambda2, utility_weight=1.0)
        
        # Log the weights we're using
        history_lambda1.append(new_lambda1)
        history_lambda2.append(new_lambda2)
        history_backlog.append(current_backlog)
        history_utilization.append(current_utilization)
        
        # 6. STEP (process events for this step)
        done = sim.step(duration_seconds=step_duration)
        sim.metrics.snapshot_step(sim.state, sim.current_time, step_start_time=sim.current_time - step_duration)
        step_count += 1
        
        if step_count % 20 == 0:
            # Get backlog AFTER step from MetricsManager
            b_after, u_after = get_sim_stats(sim)
            print(f"   Step {step_count}: Backlog={current_backlog}→{b_after}, Util={current_utilization:.2%}→{u_after:.2%}, λ1={new_lambda1:.2f}, λ2={new_lambda2:.2f}")

    # Use get_final_results() to get properly calculated metrics
    results = sim.get_final_results()
    
    # Add stats about the weights used
    results['avg_lambda1'] = np.mean(history_lambda1)
    results['avg_lambda2'] = np.mean(history_lambda2)
    results['min_lambda1'] = np.min(history_lambda1)
    results['max_lambda1'] = np.max(history_lambda1)
    results['min_lambda2'] = np.min(history_lambda2)
    results['max_lambda2'] = np.max(history_lambda2)
    
    print(f"   ✅ Completed {step_count} steps")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate dynamic weighting potential")
    parser.add_argument("--test-mode", action="store_true", 
                       help=f"Run in test mode ({TEST_DURATION_HOURS} hours) for quick validation")
    parser.add_argument("--test-hours", type=float, default=TEST_DURATION_HOURS,
                       help=f"Duration in hours for test mode (default: {TEST_DURATION_HOURS})")
    parser.add_argument("--stress-test", action="store_true",
                       help="Enable stress test mode: subsample workers by 50% (more tasks than workers)")
    args = parser.parse_args()
    
    # Override TEST_MODE if command-line flag is set
    use_test_mode = args.test_mode or TEST_MODE
    test_duration = args.test_hours
    stress_test = args.stress_test
    
    print("=" * 70)
    print("🧪 DYNAMIC WEIGHTING VALIDATION")
    print("=" * 70)
    print("This script validates that dynamic control has potential before RL training.")
    if use_test_mode:
        print(f"⚠️  TEST MODE ENABLED: Running {test_duration} hours of data (full dataset is ~16 hours)")
        print("   This is for quick validation of normalization and lambda changes.")
    else:
        print("   Running full dataset (~16 hours)")
    if stress_test:
        print("⚠️  STRESS TEST MODE ENABLED: Workers will be subsampled by 50%")
        print("   This creates a scenario with more tasks than workers.")
    print()
    
    # 1. Load Data
    print("⏳ Loading Data...")
    # Use first day from full_didi_gaia
    if os.path.exists(DATA_PATH):
        day_folders = sorted([d for d in os.listdir(DATA_PATH) 
                             if os.path.isdir(os.path.join(DATA_PATH, d))])
        if not day_folders:
            print(f"❌ No day folders found in {DATA_PATH}")
            sys.exit(1)
        
        # Use first day (index 0)
        selected_day = day_folders[0]
        print(f"   Using dataset: {selected_day} (first day)")
        
        day_path = os.path.join(DATA_PATH, selected_day)
    else:
        print(f"❌ Data path not found: {DATA_PATH}")
        sys.exit(1)
    
    all_workers, all_tasks = load_workers_tasks("didi", root_path=day_path)
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")
    
    # Apply stress test: subsample workers by 50% using stratified sampling
    if stress_test:
        original_worker_count = len(all_workers)
        target_worker_count = original_worker_count // 2
        
        print(f"⚠️  STRESS TEST: Sampling {target_worker_count:,} workers (50% subsample)...")
        # Use stratified sampler: keep all tasks, sample 50% of workers
        sampled_tasks, worker_samples = stratified_temporal_sample(
            all_workers=all_workers,
            all_tasks=all_tasks,
            target_tasks=len(all_tasks),  # Keep all tasks
            worker_counts=[target_worker_count],  # Sample 50% of workers
            seed=42
        )
        
        workers = worker_samples[target_worker_count]
        tasks = sampled_tasks
        
        print(f"⚠️  STRESS TEST: Reduced workers from {original_worker_count:,} to {len(workers):,} (50% subsample)")
        print(f"   Tasks: {len(tasks):,} (all tasks kept)")
        print(f"   Task-to-worker ratio: {len(tasks) / len(workers):.2f}:1")
    else:
        workers = all_workers
        tasks = all_tasks
    
    # 2. Run Static to get Peaks
    static_res, peaks = run_static_baseline(workers, tasks, test_mode=use_test_mode, test_duration_hours=test_duration)
    
    # 3. Run Dynamic using those Peaks
    dynamic_res = run_dynamic_heuristic(workers, tasks, peaks, test_mode=use_test_mode, test_duration_hours=test_duration)
    
    # 4. Print Report
    print("\n" + "="*70)
    print("🏆 HEURISTIC VALIDATION REPORT")
    print("="*70)
    
    # Metrics organized by category
    metrics = {
        '--- Completion & Revenue ---': {},
        'Tasks Completed': 'completed_tasks',
        'Task Assignment Ratio (%)': 'task_assignment_ratio',
        'Total Revenue ($)': 'total_platform_revenue',
        '--- Fairness ---': {},
        'Jains Fairness Index': 'final_jains_fairness_index',
        'Gini Coefficient': 'final_gini_coefficient',
        'JFI Earnings': 'final_jfi_earnings',
        'JFI Earnings/Opp': 'final_jfi_earnings_opportunity',
        'Gini Earnings': 'final_gini_earnings',
        '--- Latency & Distance ---': {},
        'Avg Wait Time (m)': 'mean_task_wait_time_min',
        'Total Travel (km)': 'total_travel_km',
        'Empty Travel (km)': 'empty_km',
        'Passenger Travel (km)': 'passenger_km',
        'Peak Backlog': 'backlog_peak',
        '--- Worker Utilization ---': {},
        'Mean Worker Idle (m)': 'mean_worker_idle_time_min',
        'Worker Idle CV': 'cv_worker_idle',
    }
    
    print(f"{'Metric':<30} | {'Static':<12} | {'Dynamic':<12} | {'Diff (%)':<10}")
    print("-" * 75)
    
    for label, key in metrics.items():
        # Skip section headers (empty values)
        if key == {}:
            print(f"\n{label}")
            continue
        
        v1 = static_res.get(key, 0)
        v2 = dynamic_res.get(key, 0)
        
        # Handle different metric types
        if 'Ratio' in label or 'Completion' in label:
            # TAR is a ratio, convert to percentage
            display_v1 = v1 * 100
            display_v2 = v2 * 100
        else:
            display_v1, display_v2 = v1, v2
            
        diff = ((display_v2 - display_v1) / display_v1) * 100 if display_v1 != 0 else 0.0
        
        print(f"{label:<30} | {display_v1:>10.4f}   | {display_v2:>10.4f}   | {diff:>+8.2f}%")
        
    print("-" * 70)
    print("Dynamic Weight Behavior:")
    print(f"  Fairness (λ1):   {dynamic_res['avg_lambda1']:.3f} (Range: {dynamic_res['min_lambda1']:.2f} - {dynamic_res['max_lambda1']:.2f})")
    print(f"  Starvation (λ2): {dynamic_res['avg_lambda2']:.3f} (Range: {dynamic_res['min_lambda2']:.2f} - {dynamic_res['max_lambda2']:.2f})")
    print()
    print("="*70)
    print("✅ VALIDATION COMPLETE")
    print()
    print("If Dynamic shows ANY improvement over Static, this proves that:")
    print("  1. Dynamic control has potential in this system")
    print("  2. An RL agent can potentially find even better policies")
    print("  3. The RL training effort is justified")
    print("="*70)
