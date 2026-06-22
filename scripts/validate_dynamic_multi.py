#!/usr/bin/env python3
"""
validate_dynamic_multi.py

Automated harness to try several dynamic-weighting heuristics in one run and
compare them side-by-side against the static baseline.

Workflow:
  1. Load data ONCE.
  2. Run + TIME the static baseline. Use that timing to estimate how many
     strategies fit inside a time budget (default 15 min).
  3. Run every enabled strategy through a SHARED simulation loop. Strategies
     differ only by their "controller" — a function that maps the current
     system state to (fairness_weight λ1, starvation_weight λ2) each step.
  4. Print a side-by-side comparison report.

Strategies included:
  - static         : fixed config weights (baseline)
  - bilateral      : current approach (Bilateral Control Loop, backlog/util)
  - peak_offpeak   : simple on/off switch by time of day
  - supply_demand  : basic heuristic on incoming tasks vs available workers

Examples:
  python scripts/validate_dynamic_multi.py --test-mode            # quick 2h smoke test
  python scripts/validate_dynamic_multi.py --time-budget-min 15   # full runs, est. how many fit
  python scripts/validate_dynamic_multi.py --only static,supply_demand
"""

import os
import sys
import time

# Add project root to path (parent of scripts/ directory)
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(project_root))

import datetime
import numpy as np

from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks
from data.stratified_sampler import stratified_temporal_sample

# --- CONFIG ---
DATA_PATH = "./data/didi/full_didi_gaia"
STEP_MINUTES = 5
STEP_DURATION = STEP_MINUTES * 60

# --- BILATERAL CONTROL LOOP RANGES (current approach) ---
STARVATION_MIN = 0.25
STARVATION_MAX = 0.5
FAIRNESS_MIN = 0.0
FAIRNESS_MAX = 2.0

# --- PEAK / OFF-PEAK SWITCH ---
# Commute peaks (local time, hour of day). On = high demand, push throughput.
PEAK_HOURS = {7, 8, 9, 17, 18, 19}
PEAK_WEIGHTS = {"fairness": 0.5, "starvation": 0.5}      # high demand: clear backlog
OFFPEAK_WEIGHTS = {"fairness": 2.0, "starvation": 0.0}   # low demand: spread fairly

# --- SUPPLY / DEMAND HEURISTIC ---
# ratio = incoming tasks this step / currently-available workers.
# ratio high  -> system stressed -> low fairness, high starvation
# ratio low   -> slack capacity  -> high fairness, low starvation
SD_FAIRNESS_HI = 2.0   # at ratio = 0
SD_FAIRNESS_LO = 0.5   # at ratio >= 1
SD_STARVATION_LO = 0.0
SD_STARVATION_HI = 0.5


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------
def get_sim_state(sim):
    """Snapshot of the signals every controller can use this step."""
    obs = sim.metrics.get_observation_data(sim.state, sim.current_time)

    stats = sim.metrics.current_step_stats
    backlog = stats.get('backlog', 0)
    avail_ratio = obs.get('worker_availability_ratio', 0.0)
    total_workers = obs.get('total_workers', 1)
    utilization = 1.0 - avail_ratio
    available_workers = max(1.0, avail_ratio * total_workers)

    # Supply/demand signal: tasks/min per active worker.
    # We read the SNAPSHOTTED value from current_step_stats (manager.snapshot_step),
    # NOT get_observation_data().task_arrival_rate — the latter is derived from
    # step_tasks_released, which is reset to 0 after every snapshot, so it reads 0
    # when observed at the start of a step (before stepping).
    demand_raw = stats.get('task_worker_ratio', 0.0)

    # Local hour of day from the sim clock
    if isinstance(sim.current_time, (int, float)):
        hour = datetime.datetime.fromtimestamp(sim.current_time).hour
    else:
        hour = sim.current_time.hour

    return {
        'backlog': backlog,
        'utilization': utilization,
        'available_workers': available_workers,
        'hour': hour,
        'demand_raw': demand_raw,
    }


# ---------------------------------------------------------------------------
# Controllers: state (+ peaks) -> (fairness_weight, starvation_weight)
# Each returns None to mean "leave weights untouched" (used by static).
# ---------------------------------------------------------------------------
def controller_static(state, peaks):
    """Baseline: never change weights from the config defaults."""
    return None


def controller_bilateral(state, peaks):
    """Current approach: Bilateral Control Loop on backlog + utilization."""
    # Loop 1: starvation proportional to backlog (vs historical peak)
    peak_backlog = peaks.get('peak_backlog', 0)
    backlog_ratio = min(state['backlog'] / peak_backlog, 1.0) if peak_backlog > 0 else 0.0
    lambda2 = STARVATION_MIN + backlog_ratio * (STARVATION_MAX - STARVATION_MIN)

    # Loop 2: fairness inversely mapped to utilization range
    min_u, max_u = peaks.get('min_utilization', 0.0), peaks.get('max_utilization', 0.0)
    u_range = max_u - min_u
    u = state['utilization']
    if u_range > 0:
        if u < min_u:
            lambda1 = FAIRNESS_MAX
        elif u > max_u:
            lambda1 = FAIRNESS_MIN
        else:
            u_norm = (u - min_u) / u_range
            lambda1 = FAIRNESS_MAX - u_norm * (FAIRNESS_MAX - FAIRNESS_MIN)
    else:
        lambda1 = 1.0

    return lambda1, lambda2


def controller_peak_offpeak(state, peaks):
    """Simple on/off switch driven purely by time of day."""
    if state['hour'] in PEAK_HOURS:
        return PEAK_WEIGHTS['fairness'], PEAK_WEIGHTS['starvation']
    return OFFPEAK_WEIGHTS['fairness'], OFFPEAK_WEIGHTS['starvation']


def controller_supply_demand(state, peaks):
    """Basic heuristic: incoming-task pressure vs available workers.

    The raw signal (tasks/min per active worker) is tiny in absolute terms
    (~0.001-0.05 on DiDi), so we normalize it against the peak demand observed
    during the static run — mirroring how the bilateral loop normalizes backlog.
    """
    peak_demand = peaks.get('peak_demand', 0.0)
    r = min(state['demand_raw'] / peak_demand, 1.0) if peak_demand > 0 else 0.0
    lambda1 = SD_FAIRNESS_HI - r * (SD_FAIRNESS_HI - SD_FAIRNESS_LO)
    lambda2 = SD_STARVATION_LO + r * (SD_STARVATION_HI - SD_STARVATION_LO)
    return lambda1, lambda2


CONTROLLERS = {
    'static':        ("Static Baseline",        controller_static),
    'bilateral':     ("Bilateral Control Loop", controller_bilateral),
    'peak_offpeak':  ("Peak / Off-Peak Switch", controller_peak_offpeak),
    'supply_demand': ("Supply/Demand Ratio",    controller_supply_demand),
}


# ---------------------------------------------------------------------------
# Generic simulation runner
# ---------------------------------------------------------------------------
def make_config():
    config = get_simulation_config()
    config['assignment_strategy'] = 'composite'
    strategy_params = get_strategy_params('composite')
    strategy_params.update({
        'enable_deferral_tracking': True,
        'enable_diagnostics': False,
    })
    config['strategy_params'] = strategy_params
    return config


def compute_end_time(workers, tasks, test_mode, test_hours):
    if not test_mode:
        return None
    earliest = min(
        min((t.release_time for t in tasks), default=float('inf')),
        min((w.release_time for w in workers), default=float('inf')),
    )
    if earliest == float('inf'):
        return None
    return earliest + test_hours * 3600


def run_strategy(workers, tasks, controller, label, peaks=None,
                 test_mode=False, test_hours=2.0, collect_peaks=False, verbose=True):
    """Run one full simulation, driving weights with `controller` each step.

    Returns (results, collected_peaks, elapsed_seconds).
    """
    if verbose:
        print(f"\n▶  {label}")

    sim = EventSimulator(workers, tasks, sim_config=make_config())
    end_time = compute_end_time(workers, tasks, test_mode, test_hours)
    sim.reset(start_time=None, end_time=end_time)

    util_history = []
    demand_history = []
    lambda1_hist, lambda2_hist = [], []

    t0 = time.perf_counter()
    done = False
    step_count = 0
    while not done:
        state = get_sim_state(sim)

        weights = controller(state, peaks or {})
        if weights is not None:
            lambda1, lambda2 = weights
            sim.update_weights(fairness_weight=lambda1,
                               starvation_weight=lambda2,
                               utility_weight=1.0)
            lambda1_hist.append(lambda1)
            lambda2_hist.append(lambda2)

        # NOTE: sim.step() calls metrics.snapshot_step() internally (see
        # simulation.py). Do NOT snapshot again here — a second call lands after
        # step_tasks_released has been reset to 0, zeroing out rate-based metrics
        # (task_worker_ratio, avg_wait, ...).
        done = sim.step(duration_seconds=STEP_DURATION)
        step_count += 1

        if collect_peaks:
            post = get_sim_state(sim)
            util_history.append(post['utilization'])
            demand_history.append(post['demand_raw'])

        if verbose and step_count % 40 == 0:
            print(f"   step {step_count:4d} | backlog={state['backlog']:.0f} "
                  f"util={state['utilization']:.0%} demand={state['demand_raw']:.4f}")

    elapsed = time.perf_counter() - t0
    results = sim.get_final_results()

    if lambda1_hist:
        results['avg_lambda1'] = float(np.mean(lambda1_hist))
        results['avg_lambda2'] = float(np.mean(lambda2_hist))
        results['min_lambda1'] = float(np.min(lambda1_hist))
        results['max_lambda1'] = float(np.max(lambda1_hist))
        results['min_lambda2'] = float(np.min(lambda2_hist))
        results['max_lambda2'] = float(np.max(lambda2_hist))

    collected = {}
    if collect_peaks:
        collected = {
            'peak_backlog': results.get('backlog_peak', 0),
            'min_utilization': float(np.min(util_history)) if util_history else 0.0,
            'max_utilization': float(np.max(util_history)) if util_history else 0.0,
            'peak_demand': float(np.max(demand_history)) if demand_history else 0.0,
        }

    if verbose:
        print(f"   ✅ {step_count} steps in {elapsed:.1f}s")

    return results, collected, elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
REPORT_METRICS = [
    ('--- Completion & Revenue ---', None),
    ('Tasks Completed', 'completed_tasks'),
    ('Task Assignment Ratio (%)', 'task_assignment_ratio'),
    ('Total Revenue ($)', 'total_platform_revenue'),
    ('--- Fairness ---', None),
    ('Jains Fairness Index', 'final_jains_fairness_index'),
    ('Gini Coefficient', 'final_gini_coefficient'),
    ('JFI Earnings', 'final_jfi_earnings'),
    ('Gini Earnings', 'final_gini_earnings'),
    ('--- Latency & Distance ---', None),
    ('Avg Wait Time (m)', 'mean_task_wait_time_min'),
    ('Total Travel (km)', 'total_travel_km'),
    ('Empty Travel (km)', 'empty_km'),
    ('Peak Backlog', 'backlog_peak'),
    ('--- Worker Utilization ---', None),
    ('Mean Worker Idle (m)', 'mean_worker_idle_time_min'),
    ('Worker Idle CV', 'cv_worker_idle'),
]


def _display_value(label, value):
    if 'Ratio (%)' in label:
        return value * 100
    return value


def print_report(order, results_by_key):
    print("\n" + "=" * (34 + 16 * len(order)))
    print("🏆 MULTI-STRATEGY COMPARISON (vs Static Baseline)")
    print("=" * (34 + 16 * len(order)))

    header = f"{'Metric':<30}"
    for key in order:
        name, _ = CONTROLLERS[key]
        header += f" | {name[:14]:>14}"
    print(header)
    print("-" * len(header))

    baseline_key = 'static' if 'static' in order else order[0]

    for label, metric_key in REPORT_METRICS:
        if metric_key is None:
            print(f"\n{label}")
            continue
        row = f"{label:<30}"
        for key in order:
            v = _display_value(label, results_by_key[key].get(metric_key, 0) or 0)
            row += f" | {v:>14.4f}"
        print(row)

    # Percentage deltas vs baseline
    print("\n" + "-" * len(header))
    print(f"Δ% vs {CONTROLLERS[baseline_key][0]}:")
    for label, metric_key in REPORT_METRICS:
        if metric_key is None:
            continue
        base = _display_value(label, results_by_key[baseline_key].get(metric_key, 0) or 0)
        row = f"{label:<30}"
        for key in order:
            v = _display_value(label, results_by_key[key].get(metric_key, 0) or 0)
            diff = ((v - base) / base * 100) if base != 0 else 0.0
            row += f" | {diff:>+13.2f}%"
        print(row)

    # Weight behaviour for dynamic strategies
    print("\n" + "-" * len(header))
    print("Dynamic weight behaviour (λ1 fairness / λ2 starvation):")
    for key in order:
        res = results_by_key[key]
        if 'avg_lambda1' in res:
            print(f"  {CONTROLLERS[key][0]:<26} "
                  f"λ1={res['avg_lambda1']:.2f} [{res['min_lambda1']:.2f}-{res['max_lambda1']:.2f}]  "
                  f"λ2={res['avg_lambda2']:.2f} [{res['min_lambda2']:.2f}-{res['max_lambda2']:.2f}]")
        else:
            cfg = get_strategy_params('composite')
            print(f"  {CONTROLLERS[key][0]:<26} "
                  f"λ1={cfg.get('fairness_weight'):.2f} (fixed)  "
                  f"λ2={cfg.get('starvation_weight'):.2f} (fixed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Try multiple dynamic-weighting heuristics in one run")
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick run on a short window for validation")
    parser.add_argument("--test-hours", type=float, default=2.0,
                        help="Window length (hours) for --test-mode (default: 2.0)")
    parser.add_argument("--stress-test", action="store_true",
                        help="Subsample workers by 50%% (more tasks than workers)")
    parser.add_argument("--time-budget-min", type=float, default=15.0,
                        help="Time budget in minutes used to estimate how many runs fit")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of strategies to run "
                             f"(choices: {', '.join(CONTROLLERS)})")
    args = parser.parse_args()

    # Resolve which strategies to run (static always first so it sets peaks)
    if args.only:
        requested = [s.strip() for s in args.only.split(',') if s.strip()]
        unknown = [s for s in requested if s not in CONTROLLERS]
        if unknown:
            print(f"❌ Unknown strategies: {unknown}. Choices: {list(CONTROLLERS)}")
            sys.exit(1)
        order = requested
    else:
        order = list(CONTROLLERS)
    if 'static' in order:
        order = ['static'] + [s for s in order if s != 'static']

    print("=" * 70)
    print("🧪 MULTI-STRATEGY DYNAMIC WEIGHTING VALIDATION")
    print("=" * 70)
    print(f"Strategies: {', '.join(order)}")
    if args.test_mode:
        print(f"⚠️  TEST MODE: {args.test_hours}h window")
    if args.stress_test:
        print("⚠️  STRESS TEST: workers subsampled by 50%")
    print()

    # 1. Load data once
    print("⏳ Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path not found: {DATA_PATH}")
        sys.exit(1)
    day_folders = sorted([d for d in os.listdir(DATA_PATH)
                          if os.path.isdir(os.path.join(DATA_PATH, d))])
    if not day_folders:
        print(f"❌ No day folders found in {DATA_PATH}")
        sys.exit(1)
    day_path = os.path.join(DATA_PATH, day_folders[0])
    print(f"   dataset: {day_folders[0]} (first day)")

    all_workers, all_tasks = load_workers_tasks("didi", root_path=day_path)
    print(f"✅ Loaded {len(all_workers):,} workers, {len(all_tasks):,} tasks")

    if args.stress_test:
        target = len(all_workers) // 2
        sampled_tasks, worker_samples = stratified_temporal_sample(
            all_workers=all_workers, all_tasks=all_tasks,
            target_tasks=len(all_tasks), worker_counts=[target], seed=42,
        )
        workers, tasks = worker_samples[target], sampled_tasks
        print(f"⚠️  Stress: {len(all_workers):,} → {len(workers):,} workers, "
              f"ratio {len(tasks)/len(workers):.2f}:1")
    else:
        workers, tasks = all_workers, all_tasks

    results_by_key = {}

    # 2. Run + TIME the static baseline first (also collects peaks for bilateral)
    static_label = CONTROLLERS['static'][0]
    static_res, peaks, elapsed = run_strategy(
        workers, tasks, controller_static, static_label,
        test_mode=args.test_mode, test_hours=args.test_hours, collect_peaks=True,
    )
    results_by_key['static'] = static_res

    # Estimate how many runs fit in the budget
    budget_s = args.time_budget_min * 60
    fits = int(budget_s // elapsed) if elapsed > 0 else 0
    print("\n" + "-" * 70)
    print(f"⏱  One simulation took {elapsed:.1f}s "
          f"({elapsed/60:.2f} min) on this dataset/window.")
    print(f"   In a {args.time_budget_min:.0f}-min budget you can run ~{fits} simulations.")
    print(f"   Requested {len(order)} strategies → "
          f"{'fits comfortably' if len(order) <= fits else 'may exceed budget'}.")
    print("-" * 70)

    # 3. Run remaining strategies through the shared loop
    for key in order:
        if key == 'static':
            continue
        _, controller = CONTROLLERS[key]
        res, _, _ = run_strategy(
            workers, tasks, controller, CONTROLLERS[key][0], peaks=peaks,
            test_mode=args.test_mode, test_hours=args.test_hours,
        )
        results_by_key[key] = res

    # 4. Report
    print_report(order, results_by_key)
    print("\n" + "=" * 70)
    print("✅ DONE — any strategy beating Static suggests dynamic control has headroom.")
    print("=" * 70)


if __name__ == "__main__":
    main()
