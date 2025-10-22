#!/usr/bin/env python3
"""
Stratified Temporal Sampling Utility
=====================================

Ensures temporal alignment between workers and tasks by sampling
proportionally across time bins.

This addresses the temporal misalignment issue discovered in Experiment 011
where early tasks had insufficient worker availability.
"""

import random
from typing import List, Tuple
import pandas as pd
from datetime import timedelta


def stratified_temporal_sample(
    all_workers: List,
    all_tasks: List,
    target_tasks: int = 20000,
    worker_counts: List[int] = None,
    num_bins: int = 12,
    seed: int = 42
) -> Tuple[List, dict]:
    """
    Sample tasks and workers using stratified temporal sampling.
    
    Args:
        all_workers: Full list of worker objects
        all_tasks: Full list of task objects
        target_tasks: Number of tasks to sample (default 20K)
        worker_counts: List of worker counts to sample (e.g., [2K, 4K, 8K])
        num_bins: Number of temporal bins (default 12 = 15-min bins for 3 hours)
        seed: Random seed for reproducibility
    
    Returns:
        sampled_tasks: List of sampled tasks (stratified across time)
        worker_samples: Dict mapping worker_count -> list of sampled workers
    """
    random.seed(seed)
    
    print("=" * 80)
    print("STRATIFIED TEMPORAL SAMPLING")
    print("=" * 80)
    print(f"Target tasks: {target_tasks:,}")
    print(f"Worker counts: {worker_counts}")
    print(f"Temporal bins: {num_bins}")
    print(f"Random seed: {seed}")
    print()
    
    # ========================================================================
    # STEP 1: Analyze temporal distribution of tasks
    # ========================================================================
    print("[STEP 1] Analyzing task temporal distribution...")
    
    # Sort tasks by release time
    sorted_tasks = sorted(all_tasks, key=lambda t: pd.to_datetime(t.release_time))
    
    task_times = [pd.to_datetime(t.release_time) for t in sorted_tasks]
    task_start = min(task_times)
    task_end = max(task_times)
    task_duration = task_end - task_start
    
    print(f"  Task window: {task_start} to {task_end}")
    print(f"  Duration: {task_duration.total_seconds() / 3600:.2f} hours")
    print(f"  Total tasks available: {len(sorted_tasks):,}")
    print()
    
    # ========================================================================
    # STEP 2: Sample tasks stratified across temporal bins
    # ========================================================================
    print("[STEP 2] Sampling tasks stratified across temporal bins...")
    
    bin_duration = task_duration / num_bins
    tasks_per_bin = target_tasks // num_bins
    
    sampled_tasks = []
    task_bin_counts = []
    
    for i in range(num_bins):
        bin_start = task_start + i * bin_duration
        bin_end = bin_start + bin_duration
        
        # Get tasks in this bin
        bin_tasks = [t for t in sorted_tasks 
                    if bin_start <= pd.to_datetime(t.release_time) < bin_end]
        
        # Sample from this bin
        n_to_sample = min(tasks_per_bin, len(bin_tasks))
        bin_sample = random.sample(bin_tasks, n_to_sample)
        sampled_tasks.extend(bin_sample)
        
        task_bin_counts.append((bin_start, bin_end, len(bin_tasks), n_to_sample))
    
    # If we're short, sample remaining from last bin
    if len(sampled_tasks) < target_tasks:
        remaining = target_tasks - len(sampled_tasks)
        last_bin_tasks = [t for t in sorted_tasks 
                         if pd.to_datetime(t.release_time) >= task_bin_counts[-1][0]]
        additional = random.sample(last_bin_tasks, min(remaining, len(last_bin_tasks)))
        sampled_tasks.extend(additional)
    
    print(f"  ✅ Sampled {len(sampled_tasks):,} tasks")
    print(f"  Distribution across bins:")
    for bin_start, bin_end, available, sampled in task_bin_counts:
        print(f"    {bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}: "
              f"{sampled:>4} tasks (from {available:>5} available)")
    print()
    
    # ========================================================================
    # STEP 3: Analyze worker temporal distribution
    # ========================================================================
    print("[STEP 3] Analyzing worker temporal distribution...")
    
    # Sort workers by release time
    sorted_workers = sorted(all_workers, key=lambda w: pd.to_datetime(w.release_time))
    
    worker_times = [pd.to_datetime(w.release_time) for w in sorted_workers]
    worker_start = min(worker_times)
    worker_end = max(worker_times)
    
    print(f"  Worker window: {worker_start} to {worker_end}")
    print(f"  Total workers available: {len(sorted_workers):,}")
    print()
    
    # Verify overlap
    overlap_start = max(task_start, worker_start)
    overlap_end = min(task_end, worker_end)
    
    print(f"  ✅ Overlap window: {overlap_start} to {overlap_end}")
    print(f"     Duration: {(overlap_end - overlap_start).total_seconds() / 3600:.2f} hours")
    print()
    
    # ========================================================================
    # STEP 4: Sample workers stratified across temporal bins (for each count)
    # ========================================================================
    worker_samples = {}
    
    if worker_counts is None:
        worker_counts = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000]
    
    for worker_count in worker_counts:
        print(f"[STEP 4.{worker_counts.index(worker_count) + 1}] Sampling {worker_count:,} workers...")
        
        workers_per_bin = worker_count // num_bins
        sampled_workers = []
        worker_bin_counts = []
        
        for i in range(num_bins):
            bin_start = overlap_start + i * bin_duration
            bin_end = bin_start + bin_duration
            
            # Get workers available during this bin
            # Worker is available if: release_time <= bin_end AND deadline >= bin_start
            bin_workers = [
                w for w in sorted_workers
                if (pd.to_datetime(w.release_time) <= bin_end and 
                    pd.to_datetime(w.deadline) >= bin_start)
            ]
            
            # Sample from this bin
            n_to_sample = min(workers_per_bin, len(bin_workers))
            
            # Avoid duplicate sampling
            available_for_sampling = [w for w in bin_workers if w not in sampled_workers]
            n_to_sample = min(n_to_sample, len(available_for_sampling))
            
            if n_to_sample > 0:
                bin_sample = random.sample(available_for_sampling, n_to_sample)
                sampled_workers.extend(bin_sample)
            
            worker_bin_counts.append((bin_start, bin_end, len(bin_workers), n_to_sample))
        
        # If we're short, sample remaining from eligible workers not yet sampled
        if len(sampled_workers) < worker_count:
            remaining = worker_count - len(sampled_workers)
            eligible_workers = [
                w for w in sorted_workers
                if (w not in sampled_workers and
                    pd.to_datetime(w.release_time) <= overlap_end and
                    pd.to_datetime(w.deadline) >= overlap_start)
            ]
            if len(eligible_workers) > 0:
                additional = random.sample(eligible_workers, min(remaining, len(eligible_workers)))
                sampled_workers.extend(additional)
        
        worker_samples[worker_count] = sampled_workers
        
        print(f"  ✅ Sampled {len(sampled_workers):,} workers")
        print(f"  Distribution across bins:")
        for bin_start, bin_end, available, sampled in worker_bin_counts:
            print(f"    {bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}: "
                  f"{sampled:>4} workers (from {available:>5} available)")
        
        # Verify temporal coverage
        sampled_worker_times = [pd.to_datetime(w.release_time) for w in sampled_workers]
        earliest = min(sampled_worker_times)
        latest = max(sampled_worker_times)
        
        print(f"  Coverage: {earliest.strftime('%H:%M')} to {latest.strftime('%H:%M')}")
        
        # Check availability at first task
        first_task_time = min([pd.to_datetime(t.release_time) for t in sampled_tasks])
        available_at_start = sum(
            1 for w in sampled_workers
            if pd.to_datetime(w.release_time) <= first_task_time <= pd.to_datetime(w.deadline)
        )
        print(f"  📊 Workers available at first task ({first_task_time.strftime('%H:%M')}): "
              f"{available_at_start:,} ({available_at_start/len(sampled_workers)*100:.1f}%)")
        print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"✅ Tasks sampled: {len(sampled_tasks):,}")
    print(f"✅ Worker samples created: {len(worker_samples)}")
    print()
    print("Worker availability at first task:")
    first_task_time = min([pd.to_datetime(t.release_time) for t in sampled_tasks])
    for worker_count, workers in worker_samples.items():
        available = sum(
            1 for w in workers
            if pd.to_datetime(w.release_time) <= first_task_time <= pd.to_datetime(w.deadline)
        )
        print(f"  {worker_count:>6,} workers: {available:>5,} available ({available/worker_count*100:>5.1f}%)")
    print("=" * 80)
    print()
    
    return sampled_tasks, worker_samples



