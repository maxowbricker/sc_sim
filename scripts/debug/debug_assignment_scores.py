#!/usr/bin/env python3
"""
Debug why 4K workers fail when 2K workers succeed.
Check assignment scores for the first task with different worker pool sizes.
"""

from data.loader import load_workers_tasks
from pathlib import Path
import pandas as pd
import random
from simulator.strategies.composite import score

# Load data
data_path = Path("data/didi")
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))

# Sample 20K tasks
random.seed(42)
sampled_tasks = random.sample(all_tasks, 20000)

# Get first task
first_task = sampled_tasks[0]
task_release = pd.to_datetime(first_task.release_time)

print("=" * 80)
print("ASSIGNMENT SCORE DEBUG")
print("=" * 80)
print(f"\nFirst task:")
print(f"  ID: {first_task.id}")
print(f"  Release: {task_release}")
print(f"  Location: ({first_task.pickup_lat:.4f}, {first_task.pickup_lon:.4f})")

# Test with different worker pool sizes
for worker_count in [2000, 4000]:
    print(f"\n{'='*80}")
    print(f"WITH {worker_count:,} WORKERS")
    print(f"{'='*80}")
    
    random.seed(42)
    sampled_workers = random.sample(all_workers, worker_count)
    
    # Find workers available when first task arrives
    available_workers = []
    for w in sampled_workers:
        w_release = pd.to_datetime(w.release_time)
        w_deadline = pd.to_datetime(w.deadline)
        if w_release <= task_release <= w_deadline:
            available_workers.append(w)
    
    print(f"\nAvailable workers at task release: {len(available_workers):,}")
    
    if len(available_workers) == 0:
        print("❌ NO WORKERS AVAILABLE!")
        continue
    
    # Calculate scores for first 10 available workers
    scores = []
    for w in available_workers[:20]:
        s = score(
            first_task, w,
            fairness_weight=2.0,
            starvation_weight=0.8,
            utility_weight=1.0,
            now=task_release,
            fairness_metric='ewma',
            all_workers=available_workers
        )
        scores.append((w.id, s))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 worker scores:")
    for i, (worker_id, s) in enumerate(scores[:10], 1):
        threshold_status = "✅ PASS" if s >= 0.5 else "❌ REJECT"
        print(f"  {i:2d}. Worker {worker_id}: {s:.4f} {threshold_status}")
    
    # Summary stats
    all_scores = [s for _, s in scores]
    print(f"\nScore statistics (sample of {len(scores)}):")
    print(f"  Max:    {max(all_scores):.4f}")
    print(f"  Min:    {min(all_scores):.4f}")
    print(f"  Mean:   {sum(all_scores)/len(all_scores):.4f}")
    print(f"  Median: {sorted(all_scores)[len(all_scores)//2]:.4f}")
    
    passing = sum(1 for s in all_scores if s >= 0.5)
    print(f"\n  Passing threshold (≥0.5): {passing}/{len(all_scores)} ({passing/len(all_scores)*100:.1f}%)")
    
    if passing == 0:
        print(f"\n⚠️  PROBLEM: NO SCORES PASS THRESHOLD!")
        print(f"  With soft_threshold=0.5, ALL assignments will be rejected")
        print(f"  → Tasks get deferred indefinitely")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print()
print("If 4K workers show lower scores than 2K workers:")
print("  → More workers = more fairness competition")
print("  → Fairness scores get diluted")
print("  → No worker passes threshold")
print("  → All tasks deferred → expire")
print("=" * 80)

