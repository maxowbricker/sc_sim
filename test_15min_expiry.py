#!/usr/bin/env python3
"""
Quick test to compare greedy vs best fairness with new 15-minute expiry.
Tests the impact of realistic task expiry on assignment strategies.
"""

import sys
import copy
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simulator.simulation import run_simulation
from data.loader import load_workers_tasks
from config import create_composite_config

print("=" * 80)
print("15-MINUTE EXPIRY TEST: Greedy vs Best Fairness")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load data (using 4K workers, 20K tasks like Exp 013)
print("📂 Loading data...")
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))

# Use stratified sampling for consistent comparison
from data.stratified_sampler import stratified_temporal_sample

print("🔄 Stratified temporal sampling...")
sampled_tasks, worker_samples = stratified_temporal_sample(
    all_workers=all_workers,
    all_tasks=all_tasks,
    target_tasks=20000,
    worker_counts=[4000],
    num_bins=12,
    seed=42
)
workers = worker_samples[4000]

print(f"✅ Loaded {len(workers):,} workers, {len(sampled_tasks):,} tasks")
print()

# ============================================================================
# Test 1: Greedy Baseline
# ============================================================================
print("=" * 80)
print("TEST 1: GREEDY BASELINE")
print("=" * 80)
print("Running greedy strategy with 15-min expiry...")
print()

greedy_config = create_composite_config(assignment_strategy="greedy")
greedy_start = datetime.now()
# CRITICAL: Deep copy to avoid mutation between tests
greedy_workers = copy.deepcopy(workers)
greedy_tasks = copy.deepcopy(sampled_tasks)
greedy_summary = run_simulation(greedy_workers, greedy_tasks, sim_config=greedy_config)
greedy_duration = (datetime.now() - greedy_start).total_seconds()

print()
print("📊 Greedy Results:")
print(f"   Completed: {greedy_summary.get('completed_tasks', 0):,} tasks")
print(f"   TAR: {greedy_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}")
print(f"   JFI: {greedy_summary.get('final_jains_fairness_index', 0):.3f}")
print(f"   Gini: {greedy_summary.get('tasks_per_worker_gini', 0):.3f}")
print(f"   Mean Wait: {greedy_summary.get('avg_wait_time_minutes', 0):.2f} min")
print(f"   P95 Wait: {greedy_summary.get('p95_wait_time_minutes', 0):.2f} min")
print(f"   Worker Util: {greedy_summary.get('mean_worker_utilization', 0):.1%}")
print(f"   Zero-task Workers: {greedy_summary.get('pct_workers_zero_tasks', 0):.1%}")
print(f"   Runtime: {greedy_duration:.1f}s")
print()

# ============================================================================
# Test 2: Best Fairness (λ₁=2.5, λ₃=1.75)
# ============================================================================
print("=" * 80)
print("TEST 2: BEST FAIRNESS (λ₁=2.5, λ₃=1.75)")
print("=" * 80)
print("Running composite strategy with best fairness config...")
print()

fairness_config = create_composite_config(
    assignment_strategy="composite",
    fairness_weight=2.5,
    starvation_weight=0.5,
    utility_weight=1.75,
    soft_threshold=0.0,
    normalize_scores=True,
    gamma=0.5,
    k=15
)

fairness_start = datetime.now()
# CRITICAL: Deep copy to get fresh, unassigned workers/tasks
fairness_workers = copy.deepcopy(workers)
fairness_tasks = copy.deepcopy(sampled_tasks)
fairness_summary = run_simulation(fairness_workers, fairness_tasks, sim_config=fairness_config)
fairness_duration = (datetime.now() - fairness_start).total_seconds()

print()
print("📊 Best Fairness Results:")
print(f"   Completed: {fairness_summary.get('completed_tasks', 0):,} tasks")
print(f"   TAR: {fairness_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}")
print(f"   JFI: {fairness_summary.get('final_jains_fairness_index', 0):.3f}")
print(f"   Gini: {fairness_summary.get('tasks_per_worker_gini', 0):.3f}")
print(f"   Mean Wait: {fairness_summary.get('avg_wait_time_minutes', 0):.2f} min")
print(f"   P95 Wait: {fairness_summary.get('p95_wait_time_minutes', 0):.2f} min")
print(f"   Worker Util: {fairness_summary.get('mean_worker_utilization', 0):.1%}")
print(f"   Zero-task Workers: {fairness_summary.get('pct_workers_zero_tasks', 0):.1%}")
print(f"   Runtime: {fairness_duration:.1f}s")
print()

# ============================================================================
# Comparison Summary
# ============================================================================
print("=" * 80)
print("COMPARISON: NEW (15-MIN) vs OLD (2-HOUR) EXPIRY")
print("=" * 80)
print()

# Calculate changes
tar_change = (fairness_summary.get('completed_tasks', 0) / len(sampled_tasks)) - (greedy_summary.get('completed_tasks', 0) / len(sampled_tasks))
jfi_change = fairness_summary.get('final_jains_fairness_index', 0) - greedy_summary.get('final_jains_fairness_index', 0)
gini_change = fairness_summary.get('tasks_per_worker_gini', 0) - greedy_summary.get('tasks_per_worker_gini', 0)
wait_change = fairness_summary.get('avg_wait_time_minutes', 0) - greedy_summary.get('avg_wait_time_minutes', 0)

print("📈 Key Metrics Comparison:")
print()
print(f"{'Metric':<25} {'Greedy':<15} {'Best Fairness':<15} {'Change':<15}")
print("-" * 70)
print(f"{'TAR':<25} {greedy_summary.get('completed_tasks', 0) / len(sampled_tasks):<15.1%} {fairness_summary.get('completed_tasks', 0) / len(sampled_tasks):<15.1%} {tar_change:>+.2%}")
print(f"{'JFI (Fairness)':<25} {greedy_summary.get('final_jains_fairness_index', 0):<15.3f} {fairness_summary.get('final_jains_fairness_index', 0):<15.3f} {jfi_change:>+.3f}")
print(f"{'Gini (Inequality)':<25} {greedy_summary.get('tasks_per_worker_gini', 0):<15.3f} {fairness_summary.get('tasks_per_worker_gini', 0):<15.3f} {gini_change:>+.3f}")
print(f"{'Mean Wait (min)':<25} {greedy_summary.get('avg_wait_time_minutes', 0):<15.2f} {fairness_summary.get('avg_wait_time_minutes', 0):<15.2f} {wait_change:>+.2f}")
print(f"{'P95 Wait (min)':<25} {greedy_summary.get('p95_wait_time_minutes', 0):<15.2f} {fairness_summary.get('p95_wait_time_minutes', 0):<15.2f} {fairness_summary.get('p95_wait_time_minutes', 0) - greedy_summary.get('p95_wait_time_minutes', 0):>+.2f}")
print(f"{'Worker Util.':<25} {greedy_summary.get('mean_worker_utilization', 0):<15.1%} {fairness_summary.get('mean_worker_utilization', 0):<15.1%} {fairness_summary.get('mean_worker_utilization', 0) - greedy_summary.get('mean_worker_utilization', 0):>+.2%}")
print(f"{'Zero-task Workers':<25} {greedy_summary.get('pct_workers_zero_tasks', 0):<15.1%} {fairness_summary.get('pct_workers_zero_tasks', 0):<15.1%} {fairness_summary.get('pct_workers_zero_tasks', 0) - greedy_summary.get('pct_workers_zero_tasks', 0):>+.2%}")
print()

print("💡 Interpretation:")
print()
if greedy_summary.get('completed_tasks', 0) / len(sampled_tasks) < 0.85:
    print("   ⚠️  TAR dropped below 85% with 15-min expiry (realistic pressure!)")
else:
    print("   ✅ TAR remains healthy with 15-min expiry")

expired_tasks = len(sampled_tasks) - greedy_summary.get('completed_tasks', 0)
print(f"   📉 {expired_tasks:,} tasks expired (customer cancelled) with greedy")

expired_tasks_fairness = len(sampled_tasks) - fairness_summary.get('completed_tasks', 0)
print(f"   📉 {expired_tasks_fairness:,} tasks expired with best fairness")

if jfi_change > 0.05:
    print(f"   🎯 Fairness strategy improves JFI by {jfi_change:.3f} (significant!)")
elif jfi_change > 0.02:
    print(f"   ✅ Fairness strategy improves JFI by {jfi_change:.3f} (moderate)")
else:
    print(f"   ⚠️  Fairness improvement is minimal ({jfi_change:.3f})")

print()
print("=" * 80)
print("COMPARISON WITH EXPERIMENT 013 (2-HOUR EXPIRY):")
print("=" * 80)
print()
print("From Exp 013 with 2-hour expiry:")
print("   Greedy: TAR=94.3%, JFI=0.711, Gini=0.363, Wait=2.60min")
print("   Best Fairness: TAR=94.3%, JFI=0.803, Gini=0.303, Wait=2.90min")
print()
print(f"With 15-minute expiry (this test):")
print(f"   Greedy: TAR={greedy_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}, JFI={greedy_summary.get('final_jains_fairness_index', 0):.3f}, Gini={greedy_summary.get('tasks_per_worker_gini', 0):.3f}, Wait={greedy_summary.get('avg_wait_time_minutes', 0):.2f}min")
print(f"   Best Fairness: TAR={fairness_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}, JFI={fairness_summary.get('final_jains_fairness_index', 0):.3f}, Gini={fairness_summary.get('tasks_per_worker_gini', 0):.3f}, Wait={fairness_summary.get('avg_wait_time_minutes', 0):.2f}min")
print()

tar_drop_greedy = 0.943 - (greedy_summary.get('completed_tasks', 0) / len(sampled_tasks))
tar_drop_fairness = 0.943 - (fairness_summary.get('completed_tasks', 0) / len(sampled_tasks))
print(f"📊 TAR Impact:")
print(f"   Greedy: Dropped {tar_drop_greedy:.1%} from 2h to 15min expiry")
print(f"   Best Fairness: Dropped {tar_drop_fairness:.1%} from 2h to 15min expiry")

if tar_drop_fairness < tar_drop_greedy:
    print(f"   🎯 Fairness strategy is MORE ROBUST to time pressure!")
elif tar_drop_fairness > tar_drop_greedy:
    print(f"   ⚠️  Fairness strategy is more affected by time pressure")
else:
    print(f"   ➡️  Both strategies equally affected")

print()
print("=" * 80)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {greedy_duration + fairness_duration:.1f}s (simulation time only)")
print("=" * 80)

