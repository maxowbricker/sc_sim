#!/usr/bin/env python3
"""
Quick test to verify Exp 012 setup before full run.
Tests with just 2 worker counts to ensure everything works.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

print("=" * 80)
print("EXP 012 QUICK TEST")
print("=" * 80)
print()

# Load data
print("[1] Loading 3-hour peak dataset...")
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks\n")

# Test stratified sampling with just 2 worker counts
print("[2] Testing stratified sampling (2K and 8K workers)...")
test_counts = [2000, 8000]

try:
    sampled_tasks, worker_samples = stratified_temporal_sample(
        all_workers=all_workers,
        all_tasks=all_tasks,
        target_tasks=20000,
        worker_counts=test_counts,
        num_bins=12,
        seed=42
    )
    
    print("\n✅ Stratified sampling successful!")
    print(f"   Tasks: {len(sampled_tasks):,}")
    print(f"   Worker samples: {len(worker_samples)}")
    print()
    
    # Verify samples
    for wc in test_counts:
        workers = worker_samples[wc]
        print(f"   {wc:>5,} workers: {len(workers):,} sampled ✅")
    
    print()
    print("=" * 80)
    print("✅ ALL CHECKS PASSED - READY TO RUN EXPERIMENT 012")
    print("=" * 80)
    print()
    print("To run the full experiment:")
    print("  cd experiments_analysis/exp_012_worker_ratio_analysis")
    print("  ../../venv/bin/python run_experiment.py")
    print()
    print("Or in background:")
    print("  nohup ../../venv/bin/python run_experiment.py > experiment_012_run.log 2>&1 &")
    print()
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)




