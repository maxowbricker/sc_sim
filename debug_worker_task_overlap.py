#!/usr/bin/env python3
"""
Quick debug script to check worker-task temporal overlap for different sample sizes.
"""

from data.loader import load_workers_tasks
from pathlib import Path
import pandas as pd
import random

# Load the 3-hour peak dataset
data_path = Path("data/didi")
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))

print("=" * 80)
print("TEMPORAL OVERLAP DEBUG: 2K vs 4K Workers")
print("=" * 80)

# Sample 20K tasks (same as experiment)
random.seed(42)
sampled_tasks = random.sample(all_tasks, min(20000, len(all_tasks)))

task_releases = [pd.to_datetime(t.release_time) for t in sampled_tasks]
task_min = min(task_releases)
task_max = max(task_releases)

print(f"\nTasks (N={len(sampled_tasks)}):")
print(f"  Release time range: {task_min} to {task_max}")
print(f"  Span: {(task_max - task_min).total_seconds() / 3600:.2f} hours")

for worker_count in [2000, 4000, 6000, 8000]:
    print(f"\n{'='*80}")
    print(f"SAMPLING {worker_count:,} WORKERS")
    print(f"{'='*80}")
    
    random.seed(42)
    sampled_workers = random.sample(all_workers, min(worker_count, len(all_workers)))
    
    # Analyze worker availability
    worker_releases = [pd.to_datetime(w.release_time) for w in sampled_workers]
    worker_deadlines = [pd.to_datetime(w.deadline) for w in sampled_workers]
    
    worker_min_release = min(worker_releases)
    worker_max_deadline = max(worker_deadlines)
    
    print(f"\nWorker availability:")
    print(f"  First release: {worker_min_release}")
    print(f"  Last deadline: {worker_max_deadline}")
    print(f"  Span: {(worker_max_deadline - worker_min_release).total_seconds() / 3600:.2f} hours")
    
    # Check overlap with task window
    overlap_start = max(task_min, worker_min_release)
    overlap_end = min(task_max, worker_max_deadline)
    
    if overlap_start < overlap_end:
        overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
        print(f"\n✅ Temporal overlap: {overlap_hours:.2f} hours")
        print(f"   {overlap_start} to {overlap_end}")
    else:
        print(f"\n❌ NO OVERLAP!")
        print(f"   Tasks end at: {task_max}")
        print(f"   Workers start at: {worker_min_release}")
    
    # Check how many workers are available during task window
    workers_available_during_tasks = []
    for w in sampled_workers:
        w_release = pd.to_datetime(w.release_time)
        w_deadline = pd.to_datetime(w.deadline)
        
        # Worker is available if their window overlaps with task window
        if w_release <= task_max and w_deadline >= task_min:
            workers_available_during_tasks.append(w)
    
    pct_available = len(workers_available_during_tasks) / len(sampled_workers) * 100
    print(f"\n📊 Workers available during task window:")
    print(f"   {len(workers_available_during_tasks):,} / {len(sampled_workers):,} ({pct_available:.1f}%)")
    
    # Check distribution of worker release times
    print(f"\n⏰ Worker release time distribution:")
    releases_df = pd.DataFrame({'release': worker_releases})
    releases_df['hour'] = releases_df['release'].dt.hour
    hour_counts = releases_df['hour'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        bar = '█' * int(count / len(sampled_workers) * 50)
        print(f"   {hour:02d}:xx - {count:>5,} workers {bar}")
    
    # Check how many tasks arrive when NO workers are available
    print(f"\n🚨 Task arrival analysis:")
    tasks_with_no_workers = 0
    for t in sampled_tasks[:100]:  # Sample first 100 for speed
        t_release = pd.to_datetime(t.release_time)
        
        # Count how many workers are available when this task arrives
        available_at_release = sum(1 for w in workers_available_during_tasks 
                                   if pd.to_datetime(w.release_time) <= t_release <= pd.to_datetime(w.deadline))
        
        if available_at_release == 0:
            tasks_with_no_workers += 1
    
    print(f"   Tasks with 0 available workers (sample of 100): {tasks_with_no_workers}")
    if tasks_with_no_workers > 50:
        print(f"   ⚠️  WARNING: Most tasks have no available workers!")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print()
print("If 4K+ workers show significantly lower temporal overlap than 2K,")
print("the random sampling is grabbing workers at the edges of the window")
print("who can't reach tasks before they expire.")
print()
print("=" * 80)

