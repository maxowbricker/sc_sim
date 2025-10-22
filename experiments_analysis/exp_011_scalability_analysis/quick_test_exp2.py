#!/usr/bin/env python3
"""
Quick test: Run just Experiment 2 (4K workers) with soft_threshold=0.0
"""

import sys
from pathlib import Path
from datetime import datetime
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_simulation_config, STRATEGY_PARAMS
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation

print("=" * 80)
print("QUICK TEST: Experiment 2 (4K workers, soft_threshold=0.0)")
print("=" * 80)

# Load data
print("\nLoading data...")
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")

# Sample 20K tasks (same as main experiment)
random.seed(42)
sampled_tasks = random.sample(all_tasks, 20000)
print(f"✅ Sampled {len(sampled_tasks):,} tasks")

# Sample 4K workers (same seed as fixed version)
random.seed(42)
sampled_workers = random.sample(all_workers, 4000)
print(f"✅ Sampled {len(sampled_workers):,} workers")
print(f"   Tasks per worker: {len(sampled_tasks) / len(sampled_workers):.1f}")

# Configure
STRATEGY_PARAMS['composite']['fairness_weight'] = 2.0
STRATEGY_PARAMS['composite']['starvation_weight'] = 0.8
STRATEGY_PARAMS['composite']['utility_weight'] = 1.0
STRATEGY_PARAMS['composite']['soft_threshold'] = 0.0  # DISABLED
STRATEGY_PARAMS['composite']['normalize_scores'] = True
STRATEGY_PARAMS['composite']['gamma'] = 0.5
STRATEGY_PARAMS['composite']['enable_diagnostics'] = False

print(f"\n⚙️  Configuration:")
print(f"   λ₁=2.0, λ₂=0.8, λ₃=1.0")
print(f"   θ=0.0 (DISABLED)")
print(f"   normalize_scores=True")

# Run
print(f"\n🏃 Running simulation...")
start = datetime.now()
cfg = get_simulation_config()
summary = run_simulation(sampled_workers, sampled_tasks, sim_config=cfg)
duration = (datetime.now() - start).total_seconds()

# Results
completed = summary.get('completed_tasks', 0)
tar = completed / 20000
jfi = summary.get('final_jains_fairness_index', 0)

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Duration:        {duration/60:.1f} minutes")
print(f"Completed:       {completed:,} / 20,000 tasks")
print(f"TAR:             {tar:.1%}")
print(f"JFI:             {jfi:.3f}")
print(f"Mean Wait:       {summary.get('avg_wait_time_minutes', 0):.1f} min")
print(f"Gini:            {summary.get('tasks_per_worker_gini', 0):.3f}")
print(f"{'='*80}")

if tar > 0.5:
    print(f"\n✅ SUCCESS! With θ=0.0, 4K workers work fine.")
    print(f"   → The soft_threshold=0.5 was the problem")
else:
    print(f"\n❌ STILL FAILING with θ=0.0")
    print(f"   → Something else is wrong beyond the threshold")

