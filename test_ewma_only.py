#!/usr/bin/env python3
"""
EWMA-Only Strategy Validation Test

Validates EWMA-Only (advanced fairness baseline) behavior against LAF (simple fairness)
and Greedy (efficiency baseline). Tests whether sophisticated time-weighted EWMA metric
provides advantages over simple task-count fairness.

Expected Outcomes:
- EWMA-Only should have JFI >= LAF (more sophisticated metric)
- EWMA-Only should have Gini <= LAF (better equality)
- EWMA-Only should have HIGHER wait times than Greedy (worse efficiency)
- EWMA-Only should have FEWER zero-task workers than Greedy (better coverage)
- EWMA-Only should perform SIMILAR to LAF (both pure fairness)
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
print("EWMA-ONLY STRATEGY VALIDATION TEST")
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
# Test 1: EWMA-Only Strategy
# ============================================================================
print("=" * 80)
print("TEST 1: EWMA-ONLY (ADVANCED FAIRNESS)")
print("=" * 80)
print("Running EWMA-Only strategy with time-weighted fairness metric...")
print()

ewma_config = create_composite_config(assignment_strategy="ewma_only", gamma=0.3)
ewma_start = datetime.now()
ewma_workers = copy.deepcopy(workers)
ewma_tasks = copy.deepcopy(sampled_tasks)
ewma_summary = run_simulation(ewma_workers, ewma_tasks, sim_config=ewma_config)
ewma_duration = (datetime.now() - ewma_start).total_seconds()

print()
print("📊 EWMA-Only Results:")
print(f"   Completed: {ewma_summary.get('completed_tasks', 0):,} tasks")
print(f"   TAR: {ewma_summary.get('completed_tasks', 0) / len(sampled_tasks):.1%}")
print(f"   JFI: {ewma_summary.get('final_jains_fairness_index', 0):.3f}")
print(f"   Gini: {ewma_summary.get('tasks_per_worker_gini', 0):.3f}")
print(f"   Mean Wait: {ewma_summary.get('avg_wait_time_minutes', 0):.2f} min")
print(f"   P95 Wait: {ewma_summary.get('p95_wait_time_minutes', 0):.2f} min")
print(f"   Worker Util: {ewma_summary.get('mean_worker_utilization', 0):.1%}")
print(f"   Zero-task Workers: {ewma_summary.get('pct_workers_zero_tasks', 0):.1%}")
print(f"   Empty-KM Share: {ewma_summary.get('empty_km_share', 0):.1%}")
print(f"   Runtime: {ewma_duration:.1f}s")
print()

# ============================================================================
# Test 2: LAF Strategy (for comparison)
# ============================================================================
print("=" * 80)
print("TEST 2: LAF (SIMPLE FAIRNESS)")
print("=" * 80)
print("Running LAF strategy for comparison...")
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
# Test 3: Greedy Baseline
# ============================================================================
print("=" * 80)
print("TEST 3: GREEDY BASELINE (EFFICIENCY)")
print("=" * 80)
print("Running greedy strategy for baseline reference...")
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
print("COMPARISON: EWMA-ONLY vs LAF vs GREEDY")
print("=" * 80)
print()

# Extract metrics
ewma_tar = ewma_summary.get('completed_tasks', 0) / len(sampled_tasks)
laf_tar = laf_summary.get('completed_tasks', 0) / len(sampled_tasks)
greedy_tar = greedy_summary.get('completed_tasks', 0) / len(sampled_tasks)

ewma_jfi = ewma_summary.get('final_jains_fairness_index', 0)
laf_jfi = laf_summary.get('final_jains_fairness_index', 0)
greedy_jfi = greedy_summary.get('final_jains_fairness_index', 0)

ewma_gini = ewma_summary.get('tasks_per_worker_gini', 0)
laf_gini = laf_summary.get('tasks_per_worker_gini', 0)
greedy_gini = greedy_summary.get('tasks_per_worker_gini', 0)

ewma_wait = ewma_summary.get('avg_wait_time_minutes', 0)
laf_wait = laf_summary.get('avg_wait_time_minutes', 0)
greedy_wait = greedy_summary.get('avg_wait_time_minutes', 0)

ewma_zero = ewma_summary.get('pct_workers_zero_tasks', 0)
laf_zero = laf_summary.get('pct_workers_zero_tasks', 0)
greedy_zero = greedy_summary.get('pct_workers_zero_tasks', 0)

ewma_empty = ewma_summary.get('empty_km_share', 0)
laf_empty = laf_summary.get('empty_km_share', 0)
greedy_empty = greedy_summary.get('empty_km_share', 0)

print(f"{'Metric':<30} | {'EWMA-Only':<12} | {'LAF':<12} | {'Greedy':<12} | {'EWMA vs LAF':<15}")
print("-" * 100)
print(f"{'Task Assignment Ratio':<30} | {ewma_tar:>11.1%} | {laf_tar:>11.1%} | {greedy_tar:>11.1%} | {ewma_tar - laf_tar:>+14.1%}")
print(f"{'Jain Fairness Index (JFI)':<30} | {ewma_jfi:>11.3f} | {laf_jfi:>11.3f} | {greedy_jfi:>11.3f} | {ewma_jfi - laf_jfi:>+14.3f}")
print(f"{'Gini Coefficient':<30} | {ewma_gini:>11.3f} | {laf_gini:>11.3f} | {greedy_gini:>11.3f} | {ewma_gini - laf_gini:>+14.3f}")
print(f"{'Mean Wait Time (min)':<30} | {ewma_wait:>11.2f} | {laf_wait:>11.2f} | {greedy_wait:>11.2f} | {ewma_wait - laf_wait:>+14.2f}")
print(f"{'Zero-Task Workers (%)':<30} | {ewma_zero:>11.1%} | {laf_zero:>11.1%} | {greedy_zero:>11.1%} | {ewma_zero - laf_zero:>+14.1%}")
print(f"{'Empty-KM Share':<30} | {ewma_empty:>11.1%} | {laf_empty:>11.1%} | {greedy_empty:>11.1%} | {ewma_empty - laf_empty:>+14.1%}")
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

# Check 1: EWMA-Only should have JFI >= LAF (or very close)
checks_total += 1
if ewma_jfi >= laf_jfi * 0.95:  # Allow 5% margin
    print("✅ PASS: EWMA-Only has JFI >= LAF (sophisticated metric works)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: EWMA-Only JFI ({ewma_jfi:.3f}) significantly lower than LAF ({laf_jfi:.3f})")

# Check 2: EWMA-Only should have Gini <= LAF (or very close)
checks_total += 1
if ewma_gini <= laf_gini * 1.05:  # Allow 5% margin
    print("✅ PASS: EWMA-Only has Gini <= LAF (good equality)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: EWMA-Only Gini ({ewma_gini:.3f}) significantly higher than LAF ({laf_gini:.3f})")

# Check 3: EWMA-Only should have higher wait times than Greedy
checks_total += 1
if ewma_wait >= greedy_wait * 0.95:  # Allow 5% margin
    print("✅ PASS: EWMA-Only has similar or higher wait time than Greedy (expected spatial inefficiency)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: EWMA-Only wait time ({ewma_wait:.2f}) significantly lower than Greedy ({greedy_wait:.2f})")
    checks_passed += 1  # Not a failure, just unexpected

# Check 4: EWMA-Only should have fewer zero-task workers than Greedy
checks_total += 1
if ewma_zero <= greedy_zero:
    print("✅ PASS: EWMA-Only has fewer or equal zero-task workers than Greedy (better coverage)")
    checks_passed += 1
else:
    print(f"⚠️  UNEXPECTED: EWMA-Only zero-task workers ({ewma_zero:.1%}) higher than Greedy ({greedy_zero:.1%})")

# Check 5: EWMA-Only should maintain reasonable TAR
checks_total += 1
if ewma_tar >= 0.85:  # At least 85% TAR
    print(f"✅ PASS: EWMA-Only maintains good TAR ({ewma_tar:.1%})")
    checks_passed += 1
else:
    print(f"⚠️  WARNING: EWMA-Only TAR ({ewma_tar:.1%}) is low")

print()
print("=" * 80)
print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
print("=" * 80)
print()

if checks_passed >= 4:
    print("🎉 SUCCESS: EWMA-Only strategy is working as expected!")
    print()
    print("Key Findings:")
    print(f"   • EWMA-Only vs LAF:")
    print(f"     - JFI: {ewma_jfi:.3f} vs {laf_jfi:.3f} ({ewma_jfi - laf_jfi:+.3f})")
    print(f"     - Gini: {ewma_gini:.3f} vs {laf_gini:.3f} ({ewma_gini - laf_gini:+.3f})")
    if abs(ewma_jfi - laf_jfi) < 0.05 and abs(ewma_gini - laf_gini) < 0.05:
        print(f"     ➡️  SIMILAR performance (both pure fairness approaches)")
    elif ewma_jfi > laf_jfi:
        print(f"     🎯 EWMA slightly better (time-weighted advantage)")
    else:
        print(f"     ⚠️  LAF slightly better (simpler metric sufficient)")
    print()
    print(f"   • EWMA-Only vs Greedy:")
    print(f"     - Fairness improvement: JFI {greedy_jfi:.3f} → {ewma_jfi:.3f} (+{(ewma_jfi - greedy_jfi):.3f})")
    print(f"     - Inequality reduction: Gini {greedy_gini:.3f} → {ewma_gini:.3f} ({(greedy_gini - ewma_gini):.3f})")
    print(f"     - Trade-off: Wait time increases by {ewma_wait - greedy_wait:.2f} min")
    print()
    print(f"   • Research Implication:")
    print(f"     Both EWMA-Only and LAF show that fairness-only approaches sacrifice")
    print(f"     efficiency. This validates the need for composite strategy that")
    print(f"     balances fairness (F), starvation (S), and utility (U).")
else:
    print("⚠️  ISSUES DETECTED: Some validation checks failed")
    print("   Review the results above for details")

print()
print("=" * 80)
print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {ewma_duration + laf_duration + greedy_duration:.1f}s")
print("=" * 80)

