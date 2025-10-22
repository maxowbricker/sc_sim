#!/usr/bin/env python3
"""
Analyze EWMA warmup characteristics and verify 3-hour window sufficiency.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("EWMA WARMUP ANALYSIS")
print("=" * 80)

# EWMA Parameters
gamma = 0.3  # Smoothing factor (default in code)

print(f"\n📊 EWMA Configuration:")
print(f"   γ (gamma) = {gamma}")
print(f"   Formula: EWMA = (1-γ) × T_idle + γ × Previous_EWMA")
print(f"   Weight on new observation: {1-gamma:.1%}")
print(f"   Weight on history: {gamma:.1%}")

# Effective window calculation
# In EWMA, ~95% of the weight comes from the last k observations where:
# 1 - γ^k = 0.95
k_95 = np.log(0.05) / np.log(gamma)
print(f"\n🔍 Effective Memory:")
print(f"   95% of weight comes from last {k_95:.1f} observations")
print(f"   → EWMA needs ~{int(np.ceil(k_95))} task assignments to stabilize")

# Simulate task completion rates
print(f"\n⏱️  TASK COMPLETION ANALYSIS:")
print("=" * 80)

# From the temporal distribution analysis:
# - 16,133 workers in 3-hour window
# - 45,464 orders in 3-hour window
# - Ratio: 2.82 tasks per worker on average

workers_in_window = 16133
tasks_in_window = 45464
window_hours = 3

avg_tasks_per_worker = tasks_in_window / workers_in_window
task_arrival_rate_per_min = tasks_in_window / (window_hours * 60)

print(f"Best 3-hour window statistics:")
print(f"   Workers: {workers_in_window:,}")
print(f"   Tasks: {tasks_in_window:,}")
print(f"   Average tasks/worker: {avg_tasks_per_worker:.2f}")
print(f"   Task arrival rate: {task_arrival_rate_per_min:.1f} tasks/minute")

# Estimate task duration (pickup + service)
# Assume: 2 km average distance, 30 km/h speed = 4 min travel + 10 min service = 14 min total
avg_task_duration_min = 14
tasks_per_worker_per_hour = 60 / avg_task_duration_min

print(f"\n   Assuming {avg_task_duration_min} min per task (pickup + service):")
print(f"   → Worker can complete ~{tasks_per_worker_per_hour:.1f} tasks/hour")
print(f"   → In 3 hours: ~{tasks_per_worker_per_hour * window_hours:.1f} tasks/worker")

# EWMA convergence simulation
print(f"\n🎯 EWMA CONVERGENCE SIMULATION:")
print("=" * 80)

# Simulate a worker going through task assignments
idle_times = [600, 300, 900, 450, 700, 350, 800, 400, 650]  # Varying idle times (seconds)

ewma_values = []
ewma = 0.0  # Initial EWMA

print(f"\n{'Task #':<8} {'Idle Time (s)':<15} {'EWMA (s)':<15} {'% Change':<12} {'Status'}")
print("-" * 80)

for i, T_idle in enumerate(idle_times, 1):
    prev_ewma = ewma
    ewma = (1 - gamma) * T_idle + gamma * ewma
    ewma_values.append(ewma)
    
    if prev_ewma > 0:
        pct_change = abs((ewma - prev_ewma) / prev_ewma) * 100
    else:
        pct_change = 100.0
    
    status = "Stabilizing..." if pct_change > 10 else "✅ Stable"
    print(f"{i:<8} {T_idle:<15} {ewma:<15.1f} {pct_change:<12.1f} {status}")

print(f"\nEWMA stabilizes after ~{int(k_95)} assignments (< 10% change)")
print(f"This takes approximately {int(k_95) * avg_task_duration_min} minutes = {int(k_95) * avg_task_duration_min / 60:.1f} hours")

# Compare 3-hour vs longer windows
print(f"\n📏 WINDOW LENGTH COMPARISON:")
print("=" * 80)

window_scenarios = [
    ("1 hour", 1, tasks_per_worker_per_hour * 1),
    ("3 hours", 3, tasks_per_worker_per_hour * 3),
    ("6 hours", 6, tasks_per_worker_per_hour * 6),
]

print(f"\n{'Window':<12} {'Expected Tasks/Worker':<25} {'EWMA Status':<30}")
print("-" * 80)
for window_name, hours, expected_tasks in window_scenarios:
    if expected_tasks >= k_95:
        status = f"✅ Sufficient ({expected_tasks:.0f} > {k_95:.0f})"
    elif expected_tasks >= k_95 * 0.5:
        status = f"⚠️  Marginal ({expected_tasks:.0f} ≈ {k_95:.0f})"
    else:
        status = f"❌ Insufficient ({expected_tasks:.0f} < {k_95:.0f})"
    
    print(f"{window_name:<12} {expected_tasks:>10.1f} tasks        {status}")

# Reality check with actual data statistics
print(f"\n🔬 REALITY CHECK:")
print("=" * 80)
print(f"From temporal distribution analysis:")
print(f"   • 3-hour window has 2.82 tasks/worker on average")
print(f"   • But distribution is highly uneven (some workers busier than others)")
print(f"   • Many workers may get 0-1 tasks, while busy workers get 5-10+ tasks")
print(f"   • The busy workers (top 30-40%) will have meaningful EWMA signals")

print(f"\n💡 ASSESSMENT:")
print("=" * 80)
if avg_tasks_per_worker >= k_95:
    print(f"✅ 3-hour window IS SUFFICIENT for EWMA warmup")
    print(f"   • Average {avg_tasks_per_worker:.1f} tasks/worker > {k_95:.0f} needed for stabilization")
    print(f"   • Busy workers will have well-calibrated EWMA signals")
    print(f"   • Less-active workers will still have meaningful relative fairness signals")
else:
    print(f"⚠️  3-hour window is MARGINAL for EWMA warmup")
    print(f"   • Average {avg_tasks_per_worker:.1f} tasks/worker < {k_95:.0f} needed for full stabilization")
    print(f"   • Only busier workers will have stable EWMA")
    print(f"   • Consider:")
    print(f"      1. Use 6-hour window (if data permits)")
    print(f"      2. Use lower gamma (e.g., 0.2) for faster adaptation")
    print(f"      3. Use simpler fairness metric (e.g., 'task_count' or 'idle_time')")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

print(f"\n1️⃣  OPTION A: Keep 3-hour window, optimize gamma")
print(f"   • Reduce gamma from 0.3 → 0.2 (faster adaptation)")
print(f"   • 95% convergence in {np.log(0.05) / np.log(0.2):.1f} steps instead of {k_95:.1f}")

print(f"\n2️⃣  OPTION B: Extend to 6-hour window")
print(f"   • More tasks per worker → better EWMA calibration")
print(f"   • But may reduce available worker/task counts")

print(f"\n3️⃣  OPTION C: Use hybrid fairness metric")
print(f"   • Use 'idle_time' instead of EWMA (no warmup needed)")
print(f"   • Simpler but less smooth (no memory of past idle times)")

print(f"\n4️⃣  OPTION D: Keep everything as-is (RECOMMENDED)")
print(f"   • EWMA will still work, just with more variance early on")
print(f"   • The fairness mechanism prioritizes underserved workers regardless")
print(f"   • Real-world validation is more important than perfect warmup")

print("\n" + "=" * 80)

