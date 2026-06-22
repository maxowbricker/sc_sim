#!/usr/bin/env python3
"""
validate_dynamic_sweep.py

Automated sweep of multiple dynamic weighting heuristics vs the static baseline.

Workflow:
  1. Run the STATIC baseline once and TIME it (establishes peaks + runtime budget).
  2. Estimate how many dynamic strategies fit in the remaining time budget.
  3. Run each enabled dynamic policy and collect full metrics.
  4. Print a side-by-side comparison table (+ % vs static) and save to JSON.

Each "policy" is a function: policy(stats, peaks) -> (lambda1, lambda2).
  - lambda1 = fairness_weight (λ1)
  - lambda2 = starvation_weight (λ2)
  - lambda3 (utility) is fixed at 1.0

Strategies included:
  - static          : fixed config weights (reference)
  - bilateral       : backlog -> λ2, utilization -> λ1 (the original validate_dynamic loop)
  - peak_offpeak    : simple on/off switch by hour-of-day (peak = efficiency, off-peak = fairness)
  - supply_demand   : (task arrival rate / available workers) -> λ1 (demand-aware)

Usage:
  python scripts/validate_dynamic_sweep.py
  python scripts/validate_dynamic_sweep.py --test-mode --test-hours 2
  python scripts/validate_dynamic_sweep.py --strategies static bilateral supply_demand
  python scripts/validate_dynamic_sweep.py --max-runtime-min 20
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path

# Add project root to path (parent of scripts/ directory)
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(project_root))

import numpy as np
from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

# --- CONFIG ---
DATA_PATH = "./data/didi/full_didi_gaia"
STEP_MINUTES = 5  # 5-min steps gave the best dynamic response in earlier tests

TEST_DURATION_HOURS = 2.0

# --- WEIGHT RANGES ---
STARVATION_MIN = 0.25
STARVATION_MAX = 0.5
FAIRNESS_MIN = 0.0
FAIRNESS_MAX = 2.0

# --- PEAK/OFF-PEAK ---
# Chengdu DiDi: morning + evening commute peaks. Tweak freely.
PEAK_HOURS = {7, 8, 9, 17, 18, 19}
PEAK_LAMBDA1 = 0.25   # peak: efficiency-focused (low fairness)
PEAK_LAMBDA2 = 0.50   # peak: aggressive starvation prevention (clear backlog)
OFFPEAK_LAMBDA1 = 2.0  # off-peak: fairness-focused
OFFPEAK_LAMBDA2 = 0.25


# ============================================================================
# STATS EXTRACTION
# ============================================================================

def get_sim_stats(sim):
    """Extract current system state via MetricsManager's clean interface."""
    obs = sim.metrics.get_observation_data(sim.state, sim.current_time)

    backlog = sim.metrics.current_step_stats.get('backlog', 0)
    availability_ratio = obs.get('worker_availability_ratio', 0.0)
    total_workers = obs.get('total_workers', 1)
    utilization = 1.0 - availability_ratio  # busy / total

    arrival_rate = obs.get('task_arrival_rate', 0.0)  # tasks per minute
    available_workers = max(1.0, availability_ratio * total_workers)
    supply_demand = arrival_rate / available_workers  # demand pressure per available worker

    hour = datetime.datetime.fromtimestamp(sim.current_time).hour

    return {
        'backlog': backlog,
        'utilization': utilization,
        'arrival_rate': arrival_rate,
        'available_workers': available_workers,
        'supply_demand': supply_demand,
        'hour': hour,
    }


# ============================================================================
# WEIGHT POLICIES  (stats, peaks) -> (lambda1, lambda2)
# ============================================================================

def policy_static(stats, peaks):
    """Reference: fixed config weights."""
    return peaks['base_lambda1'], peaks['base_lambda2']


def policy_bilateral(stats, peaks):
    """Original loop: backlog drives λ2, utilization (inverse) drives λ1."""
    # Control Loop 1: Starvation proportional to backlog
    backlog_ratio = min(stats['backlog'] / peaks['peak_backlog'], 1.0) if peaks['peak_backlog'] > 0 else 0.0
    l2 = STARVATION_MIN + backlog_ratio * (STARVATION_MAX - STARVATION_MIN)

    # Control Loop 2: Fairness mapped inversely to utilization range
    l1 = _inverse_range_map(stats['utilization'], peaks['min_utilization'], peaks['max_utilization'])
    return l1, l2


def policy_peak_offpeak(stats, peaks):
    """Simple on/off switch by hour-of-day."""
    if stats['hour'] in PEAK_HOURS:
        return PEAK_LAMBDA1, PEAK_LAMBDA2
    return OFFPEAK_LAMBDA1, OFFPEAK_LAMBDA2


def policy_supply_demand(stats, peaks):
    """Demand-aware: high (arrival / available workers) -> efficiency; low -> fairness."""
    l1 = _inverse_range_map(stats['supply_demand'], peaks['min_supply_demand'], peaks['max_supply_demand'])
    # Backlog still nudges starvation weight
    backlog_ratio = min(stats['backlog'] / peaks['peak_backlog'], 1.0) if peaks['peak_backlog'] > 0 else 0.0
    l2 = STARVATION_MIN + backlog_ratio * (STARVATION_MAX - STARVATION_MIN)
    return l1, l2


def _inverse_range_map(value, lo, hi):
    """Map value in [lo, hi] to fairness [FAIRNESS_MAX, FAIRNESS_MIN] (inverse)."""
    rng = hi - lo
    if rng <= 0:
        return 1.0
    if value <= lo:
        return FAIRNESS_MAX
    if value >= hi:
        return FAIRNESS_MIN
    norm = (value - lo) / rng
    return FAIRNESS_MAX - norm * (FAIRNESS_MAX - FAIRNESS_MIN)


POLICIES = {
    'static': policy_static,
    'bilateral': policy_bilateral,
    'peak_offpeak': policy_peak_offpeak,
    'supply_demand': policy_supply_demand,
}


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def _build_sim(workers, tasks, test_mode, test_hours):
    config = get_simulation_config()
    config['assignment_strategy'] = 'composite'
    strategy_params = get_strategy_params('composite')
    strategy_params.update({
        'enable_deferral_tracking': True,
        'enable_diagnostics': False,
    })
    config['strategy_params'] = strategy_params

    sim = EventSimulator(workers, tasks, sim_config=config)

    end_time = None
    if test_mode:
        earliest = min(
            min((t.release_time for t in tasks), default=float('inf')),
            min((w.release_time for w in workers), default=float('inf')),
        )
        if earliest != float('inf'):
            end_time = earliest + test_hours * 3600
    sim.reset(start_time=None, end_time=end_time)
    return sim, strategy_params


def run_policy(name, policy_fn, workers, tasks, peaks, test_mode=False, test_hours=2.0):
    """Run one full simulation applying `policy_fn` each step. Returns (results, info)."""
    print(f"\n▶️  Running strategy: {name}")
    sim, _ = _build_sim(workers, tasks, test_mode, test_hours)

    step_duration = STEP_MINUTES * 60
    l1_hist, l2_hist = [], []
    # Range trackers (used during the static run to build peaks)
    util_hist, sd_hist = [], []

    t0 = time.perf_counter()
    done = False
    step_count = 0
    while not done:
        # Observe current state (from previous step's snapshot)
        stats = get_sim_stats(sim)
        util_hist.append(stats['utilization'])
        sd_hist.append(stats['supply_demand'])

        # Decide + apply weights for this step
        l1, l2 = policy_fn(stats, peaks)
        sim.update_weights(fairness_weight=l1, starvation_weight=l2, utility_weight=1.0)
        l1_hist.append(l1)
        l2_hist.append(l2)

        # Step (this internally calls snapshot_step)
        done = sim.step(duration_seconds=step_duration)
        step_count += 1

    elapsed = time.perf_counter() - t0
    results = sim.get_final_results()

    info = {
        'elapsed_sec': elapsed,
        'steps': step_count,
        'avg_lambda1': float(np.mean(l1_hist)),
        'avg_lambda2': float(np.mean(l2_hist)),
        'min_lambda1': float(np.min(l1_hist)),
        'max_lambda1': float(np.max(l1_hist)),
        'min_lambda2': float(np.min(l2_hist)),
        'max_lambda2': float(np.max(l2_hist)),
        # Range data (only meaningful for the static run)
        'min_utilization': float(np.min(util_hist)) if util_hist else 0.0,
        'max_utilization': float(np.max(util_hist)) if util_hist else 1.0,
        'min_supply_demand': float(np.min(sd_hist)) if sd_hist else 0.0,
        'max_supply_demand': float(np.max(sd_hist)) if sd_hist else 1.0,
        'peak_backlog': float(results.get('backlog_peak', 0)),
    }
    print(f"   ✅ {step_count} steps in {elapsed:.1f}s "
          f"(λ1 {info['min_lambda1']:.2f}-{info['max_lambda1']:.2f}, "
          f"λ2 {info['min_lambda2']:.2f}-{info['max_lambda2']:.2f})")
    return results, info


# ============================================================================
# REPORTING
# ============================================================================

REPORT_METRICS = [
    ('Tasks Completed', 'completed_tasks', 1.0, '{:.0f}'),
    ('TAR (%)', 'task_assignment_ratio', 100.0, '{:.2f}'),
    ('Revenue ($)', 'total_platform_revenue', 1.0, '{:.0f}'),
    ('JFI', 'final_jains_fairness_index', 1.0, '{:.4f}'),
    ('Gini', 'final_gini_coefficient', 1.0, '{:.4f}'),
    ('JFI Earnings', 'final_jfi_earnings', 1.0, '{:.4f}'),
    ('Gini Earnings', 'final_gini_earnings', 1.0, '{:.4f}'),
    ('Avg Wait (m)', 'mean_task_wait_time_min', 1.0, '{:.3f}'),
    ('Total Travel (km)', 'total_travel_km', 1.0, '{:.0f}'),
    ('Mean Idle (m)', 'mean_worker_idle_time_min', 1.0, '{:.2f}'),
    ('Idle CV', 'cv_worker_idle', 1.0, '{:.4f}'),
]


def print_report(all_results):
    """all_results: list of (name, results_dict, info_dict). First entry must be static."""
    static_name, static_res, _ = all_results[0]

    print("\n" + "=" * 100)
    print("🏆 DYNAMIC STRATEGY SWEEP — COMPARISON REPORT")
    print("=" * 100)

    names = [name for name, _, _ in all_results]
    header = f"{'Metric':<20}" + "".join(f"{n:>16}" for n in names)
    print(header)
    print("-" * len(header))

    for label, key, scale, fmt in REPORT_METRICS:
        row = f"{label:<20}"
        for _, res, _ in all_results:
            val = res.get(key, 0) * scale
            row += f"{fmt.format(val):>16}"
        print(row)

    # % vs static block
    print("\n" + "-" * len(header))
    print("Δ vs static (%)")
    print("-" * len(header))
    for label, key, scale, _ in REPORT_METRICS:
        base = static_res.get(key, 0)
        row = f"{label:<20}"
        for name, res, _ in all_results:
            if name == static_name:
                row += f"{'—':>16}"
                continue
            val = res.get(key, 0)
            diff = ((val - base) / base * 100) if base != 0 else 0.0
            row += f"{diff:>+15.2f}%"
        print(row)

    # Weight behaviour
    print("\n" + "-" * len(header))
    print("Weight behaviour (λ1 fairness / λ2 starvation)")
    print("-" * len(header))
    for name, _, info in all_results:
        print(f"  {name:<16} λ1 avg={info['avg_lambda1']:.3f} "
              f"[{info['min_lambda1']:.2f}-{info['max_lambda1']:.2f}]  "
              f"λ2 avg={info['avg_lambda2']:.3f} "
              f"[{info['min_lambda2']:.2f}-{info['max_lambda2']:.2f}]  "
              f"({info['elapsed_sec']:.1f}s)")

    # Best-per-metric callout (fairness up, gini down, wait down, revenue up)
    print("\n" + "-" * len(header))
    print("Best strategy per key metric")
    print("-" * len(header))
    _best(all_results, 'final_jains_fairness_index', 'JFI', higher_better=True)
    _best(all_results, 'final_jfi_earnings', 'JFI Earnings', higher_better=True)
    _best(all_results, 'final_gini_earnings', 'Gini Earnings', higher_better=False)
    _best(all_results, 'task_assignment_ratio', 'TAR', higher_better=True)
    _best(all_results, 'total_platform_revenue', 'Revenue', higher_better=True)
    _best(all_results, 'mean_task_wait_time_min', 'Avg Wait', higher_better=False)
    print("=" * 100)


def _best(all_results, key, label, higher_better=True):
    vals = [(name, res.get(key, 0)) for name, res, _ in all_results]
    winner = max(vals, key=lambda x: x[1]) if higher_better else min(vals, key=lambda x: x[1])
    print(f"  {label:<16} → {winner[0]:<16} ({winner[1]:.4f})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automated dynamic-weighting strategy sweep")
    parser.add_argument("--test-mode", action="store_true",
                        help=f"Limit each run to --test-hours of data")
    parser.add_argument("--test-hours", type=float, default=TEST_DURATION_HOURS)
    parser.add_argument("--stress-test", action="store_true",
                        help="Subsample workers by 50%% (more tasks than workers)")
    parser.add_argument("--strategies", nargs="+", default=list(POLICIES.keys()),
                        choices=list(POLICIES.keys()),
                        help="Which strategies to run (static always runs first)")
    parser.add_argument("--max-runtime-min", type=float, default=None,
                        help="Soft budget; warns/stops if estimated total exceeds this")
    args = parser.parse_args()

    print("=" * 100)
    print("🧪 AUTOMATED DYNAMIC WEIGHTING SWEEP")
    print("=" * 100)

    # 1. Load data
    print("\n⏳ Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path not found: {DATA_PATH}")
        sys.exit(1)
    day_folders = sorted(d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)))
    if not day_folders:
        print(f"❌ No day folders in {DATA_PATH}")
        sys.exit(1)
    selected_day = day_folders[0]
    print(f"   Using dataset: {selected_day} (first day)")
    all_workers, all_tasks = load_workers_tasks("didi", root_path=os.path.join(DATA_PATH, selected_day))
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")

    if args.stress_test:
        target_w = len(all_workers) // 2
        print(f"⚠️  STRESS TEST: sampling {target_w:,} workers (50%)...")
        sampled_tasks, worker_samples = stratified_temporal_sample(
            all_workers=all_workers, all_tasks=all_tasks,
            target_tasks=len(all_tasks), worker_counts=[target_w], seed=42,
        )
        workers, tasks = worker_samples[target_w], sampled_tasks
        print(f"   Ratio: {len(tasks) / len(workers):.2f} tasks/worker")
    else:
        workers, tasks = all_workers, all_tasks

    base_params = get_strategy_params('composite')
    base_l1 = float(base_params.get('fairness_weight', 1.0))
    base_l2 = float(base_params.get('starvation_weight', 0.25))

    # 2. STATIC baseline first (time it + establish peaks)
    peaks = {
        'base_lambda1': base_l1, 'base_lambda2': base_l2,
        'peak_backlog': 1.0, 'min_utilization': 0.0, 'max_utilization': 1.0,
        'min_supply_demand': 0.0, 'max_supply_demand': 1.0,
    }
    static_res, static_info = run_policy('static', policy_static, workers, tasks, peaks,
                                         test_mode=args.test_mode, test_hours=args.test_hours)

    # Fill peaks from the static run so dynamic policies normalize correctly
    peaks.update({
        'peak_backlog': static_info['peak_backlog'],
        'min_utilization': static_info['min_utilization'],
        'max_utilization': static_info['max_utilization'],
        'min_supply_demand': static_info['min_supply_demand'],
        'max_supply_demand': static_info['max_supply_demand'],
    })

    # 3. Budget estimate
    t_static = static_info['elapsed_sec']
    dynamic_strats = [s for s in args.strategies if s != 'static']
    est_total = t_static * (1 + len(dynamic_strats))
    print("\n" + "-" * 60)
    print(f"⏱️  Static run: {t_static:.1f}s")
    print(f"   Estimated total for {len(dynamic_strats)} dynamic strategies: ~{est_total:.0f}s "
          f"(~{est_total/60:.1f} min)")
    if args.max_runtime_min is not None:
        budget_sec = args.max_runtime_min * 60
        max_runs = int(budget_sec / t_static) if t_static > 0 else 0
        print(f"   Budget {args.max_runtime_min:.0f} min → ~{max_runs} runs fit (incl. static)")
        if est_total > budget_sec:
            keep = max(0, max_runs - 1)
            print(f"   ⚠️  Trimming to first {keep} dynamic strategies to fit budget.")
            dynamic_strats = dynamic_strats[:keep]
    print("-" * 60)

    # 4. Run dynamic strategies
    all_results = [('static', static_res, static_info)]
    for name in dynamic_strats:
        res, info = run_policy(name, POLICIES[name], workers, tasks, peaks,
                               test_mode=args.test_mode, test_hours=args.test_hours)
        all_results.append((name, res, info))

    # 5. Report + save
    print_report(all_results)

    out_dir = Path(project_root).parent / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"dynamic_sweep_{stamp}.json"
    payload = {
        'dataset': selected_day,
        'step_minutes': STEP_MINUTES,
        'test_mode': args.test_mode,
        'stress_test': args.stress_test,
        'peaks': peaks,
        'runs': [
            {'name': name, 'info': info,
             'metrics': {key: res.get(key, 0) for _, key, _, _ in REPORT_METRICS}}
            for name, res, info in all_results
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n💾 Saved results to: {out_path}")


if __name__ == "__main__":
    main()
