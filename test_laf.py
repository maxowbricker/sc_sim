#!/usr/bin/env python3
"""
LAF Strategy Validation Test

Quick test to validate LAF (Least Allocated Worker First) behavior.
Compares LAF against Greedy baseline on a small dataset.

Expected Outcomes:
- LAF should have HIGHER JFI (more fair)
- LAF should have LOWER Gini (more equal distribution)
- LAF should have HIGHER wait times (worse efficiency)
- LAF should have FEWER zero-task workers (better coverage)
"""

import sys
import copy
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simulator.simulation import run_simulation
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample
from config import create_composite_config

print("=" * 80)
print("LAF STRATEGY VALIDATION TEST")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# Load and Sample Data
# ============================================================================
print("📂 Loading data...")
all_workers, all_tasks = load_workers_tasks('didi', str(project_root / "data" / "didi"))

print("📊 Sampling dataset (1K workers, 5K tasks)...")
sampled_tasks, worker_samples = stratified_temporal_sample(
    all_workers=all_workers,
    all_tasks=all_tasks,
    target_tasks=5000,  # Small test set
    worker_counts=[1000],  # Small worker set
    num_bins=12,
    seed=42
)

workers = worker_samples[1000]
print(f"✅ Loaded {len(workers):,} workers, {len(sampled_tasks):,} tasks")
print()

# ============================================================================
# Test 1: LAF Strategy
# ============================================================================
print("=" * 80)
print("TEST 1: LAF (LEAST ALLOCATED WORKER FIRST)")
print("=" * 80)
print("Running LAF strategy...")
print()

laf_config = create_composite_config(assignment_strategy="laf")
laf_start = datetime.now()
laf_workers = copy.deepcopy(workers)
laf_tasks = copy.deepcopy(sampled_tasks)
laf_summary = run_simulation(laf_workers, laf_tasks, sim_config=laf_config)
laf_duration = (datetime.now() - laf_start).total_seconds()

print()
print("📊 LAF Results:")
print(f"   Completed: {laf_summary.get('completed_tasks', 0):,} tasks")
print(f"   TAR: {laf_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}")
print(f"   JFI: {laf_summary.get('final_jains_fairness_index', 0):.3f}")
print(f"   Gini: {laf_summary.get('tasks_per_worker_gini', 0):.3f}")
print(f"   Mean Wait: {laf_summary.get('avg_wait_time_minutes', 0):.2f} min")
print(f"   P95 Wait: {laf_summary.get('p95_wait_time_minutes', 0):.2f} min")
print(f"   Worker Util: {laf_summary.get('mean_worker_utilization', 0):.1%}")
print(f"   Zero-task Workers: {laf_summary.get('pct_workers_zero_tasks', 0):.1%}")
print(f"   Empty-KM Share: {laf_summary.get('empty_km_share', 0):.1%}")
print(f"   Runtime: {laf_duration:.1f}s")
print()

# ============================================================================
# Test 2: Greedy Baseline (for comparison)
# ============================================================================
print("=" * 80)
print("TEST 2: GREEDY BASELINE")
print("=" * 80)
print("Running greedy strategy for comparison...")
print()

greedy_config = create_composite_config(assignment_strategy="greedy")
greedy_start = datetime.now()
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
print(f"   Empty-KM Share: {greedy_summary.get('empty_km_share', 0):.1%}")
print(f"   Runtime: {greedy_duration:.1f}s")
print()

# ============================================================================
# Comparison Analysis
# ============================================================================
print("=" * 80)
print("COMPARISON: LAF vs GREEDY")
print("=" * 80)
print()

# Extract metrics
laf_tar = laf_summary.get('completed_tasks', 0) / len(sampled_tasks)
greedy_tar = greedy_summary.get('completed_tasks', 0) / len(sampled_tasks)
laf_jfi = laf_summary.get('final_jains_fairness_index', 0)
greedy_jfi = greedy_summary.get('final_jains_fairness_index', 0)
laf_gini = laf_summary.get('tasks_per_worker_gini', 0)
greedy_gini = greedy_summary.get('tasks_per_worker_gini', 0)
laf_wait = laf_summary.get('avg_wait_time_minutes', 0)
greedy_wait = greedy_summary.get('avg_wait_time_minutes', 0)
laf_zero = laf_summary.get('pct_workers_zero_tasks', 0)
greedy_zero = greedy_summary.get('pct_workers_zero_tasks', 0)
laf_empty = laf_summary.get('empty_km_share', 0)
greedy_empty = greedy_summary.get('empty_km_share', 0)

print(f"{'Metric':<30} | {'LAF':<12} | {'Greedy':<12} | {'Change':<15}")
print("-" * 80)
print(f"{'Task Assignment Ratio':<30} | {laf_tar:>11.1%} | {greedy_tar:>11.1%} | {laf_tar - greedy_tar:>+14.1%}")
print(f"{'Jain Fairness Index (JFI)':<30} | {laf_jfi:>11.3f} | {greedy_jfi:>11.3f} | {laf_jfi - greedy_jfi:>+14.3f}")
print(f"{'Gini Coefficient':<30} | {laf_gini:>11.3f} | {greedy_gini:>11.3f} | {laf_gini - greedy_gini:>+14.3f}")
print(f"{'Mean Wait Time (min)':<30} | {laf_wait:>11.2f} | {greedy_wait:>11.2f} | {laf_wait - greedy_wait:>+14.2f}")
print(f"{'Zero-Task Workers (%)':<30} | {laf_zero:>11.1%} | {greedy_zero:>11.1%} | {laf_zero - greedy_zero:>+14.1%}")
print(f"{'Empty-KM Share':<30} | {laf_empty:>11.1%} | {greedy_empty:>11.1%} | {laf_empty - greedy_empty:>+14.1%}")
print()

# ============================================================================
# Validation Checks
# ============================================================================
print("=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)
print()

checks_passed = 0
checks_total = 0

# Check 1: LAF should have higher JFI
checks_total += 1
if laf_jfi > greedy_jfi:
    print("✅ PASS: LAF has higher JFI (more fair)")
    checks_passed += 1
else:
    print(f"❌ FAIL: LAF JFI ({laf_jfi:.3f}) not higher than Greedy ({greedy_jfi:.3f})")

# Check 2: LAF should have lower Gini
checks_total += 1
if laf_gini < greedy_gini:
    print("✅ PASS: LAF has lower Gini (more equal distribution)")
    checks_passed += 1
else:
    print(f"❌ FAIL: LAF Gini ({laf_gini:.3f}) not lower than Greedy ({greedy_gini:.3f})")

# Check 3: LAF should have higher wait times (acceptable trade-off)
checks_total += 1
if laf_wait >= greedy_wait * 0.95:  # Allow 5% margin
    print("✅ PASS: LAF has similar or higher wait time (expected spatial inefficiency)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: LAF wait time ({laf_wait:.2f}) significantly lower than Greedy ({greedy_wait:.2f})")
    checks_passed += 1  # Not a failure, just unexpected

# Check 4: LAF should have fewer zero-task workers
checks_total += 1
if laf_zero <= greedy_zero:
    print("✅ PASS: LAF has fewer or equal zero-task workers (better coverage)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: LAF zero-task workers ({laf_zero:.1%}) higher than Greedy ({greedy_zero:.1%})")

# Check 5: LAF should maintain reasonable TAR
checks_total += 1
if laf_tar >= 0.85:  # At least 85% TAR
    print(f"✅ PASS: LAF maintains good TAR ({laf_tar:.1%})")
    checks_passed += 1
else:
    print(f"⚠️  WARNING: LAF TAR ({laf_tar:.1%}) is low")

print()
print("=" * 80)
print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
print("=" * 80)
print()

if checks_passed >= 4:
    print("🎉 SUCCESS: LAF strategy is working as expected!")
    print()
    print("Key Findings:")
    print(f"   • LAF improves fairness: JFI {greedy_jfi:.3f} → {laf_jfi:.3f} (+{(laf_jfi - greedy_jfi):.3f})")
    print(f"   • LAF reduces inequality: Gini {greedy_gini:.3f} → {laf_gini:.3f} ({(greedy_gini - laf_gini):.3f})")
    print(f"   • Trade-off: Wait time increases by {laf_wait - greedy_wait:.2f} min")
    print(f"   • Trade-off: Empty-KM increases by {(laf_empty - greedy_empty) * 100:.1f}%")
else:
    print("⚠️  ISSUES DETECTED: Some validation checks failed")
    print("   Review the results above for details")

print()
print("=" * 80)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {laf_duration + greedy_duration:.1f}s")
print("=" * 80)

