#!/usr/bin/env python3
"""
Parameter Verification — ONRTA-OP expected_a / expected_b injection

ONRTA-OP splits the simulation horizon into two stages:
    Stage 1: greedy matching  (arrivals <= floor((expected_a + expected_b) / 2))
    Stage 2: global Hungarian matching  (arrivals > threshold)

expected_a = |R|  (total task count)
expected_b = sum(w.c) = |W|  (total worker count, unit capacity)

These are injected into strategy_params at EventSimulator.reset() when the
config values are None (simulation.py lines ~186-190). This script verifies:

    1. That expected_a and expected_b are set to non-None values after reset()
    2. That they match the dataset sizes (len(tasks), len(workers))
    3. That the Stage 2 threshold is reasonable relative to total events
    4. That Stage 2 actually fires during a short trial run

No full simulation is run — the trial run processes only the first N events
to confirm Stage 2 triggers at roughly the right point.

Usage:
    python scripts/experiments/parameter_tuning/verify_onrta_op_injection.py
    python scripts/experiments/parameter_tuning/verify_onrta_op_injection.py --didi-only
    python scripts/experiments/parameter_tuning/verify_onrta_op_injection.py --events 5000
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
sys.path.insert(0, PROJECT_ROOT)

from config import create_composite_config
from data.loader import load_workers_tasks

DIDI_ROOT  = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
DIDI_DAY   = "496528674@qq.com_20161109"

GOWALLA_ROOT   = os.path.join(PROJECT_ROOT, "data", "gowalla")
GOWALLA_KWARGS = dict(
    region                 = "austin",
    date_start             = "2010-09-01",
    date_end               = "2010-09-30",
    task_mode              = "checkin",
    task_window_hours      = 0.5,
    shift_hours            = 8.0,
    dropoff_noise_km       = 2.0,
    compress_to_day        = True,
    workers_per_task_ratio = 0.20,
    random_state           = 42,
)

# How many events to step through to confirm Stage 2 fires
DEFAULT_TRIAL_EVENTS = 3_000


# ---------------------------------------------------------------------------
# Monkey-patch to intercept the arrivals counter mid-run
# ---------------------------------------------------------------------------

def _patch_tracker(sim) -> Dict[str, Any]:
    """Return a reference to the live onrta_tracker dict inside the sim."""
    return sim.strategy_params.get("onrta_tracker", {})


def verify_dataset(dataset_name: str, workers_tmpl, tasks_tmpl,
                   trial_events: int) -> bool:
    from simulator.simulation import EventSimulator

    cfg = create_composite_config(assignment_strategy="onrta_op")

    # Use fresh copies so we don't mutate the template
    sim = EventSimulator(
        copy.deepcopy(workers_tmpl),
        copy.deepcopy(tasks_tmpl),
        cfg,
    )

    print(f"\n{'─' * 65}")
    print(f"  {dataset_name}")
    print(f"{'─' * 65}")

    # ── Step 1: call reset() and inspect injected values ───────────────────
    sim.reset()
    params = sim.strategy_params

    expected_a = params.get("expected_a")
    expected_b = params.get("expected_b")
    tracker    = params.get("onrta_tracker")

    n_tasks   = len(tasks_tmpl)
    n_workers = len(workers_tmpl)

    print(f"\n  After reset():")
    print(f"    expected_a (tasks)   = {expected_a}  (dataset |R| = {n_tasks})")
    print(f"    expected_b (workers) = {expected_b}  (dataset |W| = {n_workers})")
    print(f"    onrta_tracker        = {tracker}")

    passed = True

    # Check 1: values are not None
    if expected_a is None or expected_b is None:
        print(f"\n  ✗ FAIL: expected_a or expected_b is None after reset().")
        print(f"    Injection in simulation.py is not firing. Check strategy_name "
              f"equals 'onrta_op' exactly.")
        return False

    # Check 2: values match dataset sizes
    if expected_a != n_tasks:
        print(f"\n  ✗ WARN: expected_a={expected_a} != len(tasks)={n_tasks}.")
        print(f"    This can happen if tasks were pre-filtered. Verify intentional.")
        passed = False
    else:
        print(f"\n  ✓ expected_a matches dataset task count ({n_tasks})")

    if expected_b != n_workers:
        print(f"  ✗ WARN: expected_b={expected_b} != len(workers)={n_workers}.")
        print(f"    Correct only if workers have capacity != 1 (multi-capacity extension).")
        passed = False
    else:
        print(f"  ✓ expected_b matches dataset worker count ({n_workers})")

    # Check 3: tracker initialised correctly
    if tracker is None or tracker.get("arrivals") != 0:
        print(f"  ✗ FAIL: onrta_tracker not properly initialised. Got: {tracker}")
        return False
    else:
        print(f"  ✓ onrta_tracker initialised with arrivals=0")

    # Check 4: threshold is sane
    threshold = (expected_a + expected_b) / 2.0
    total_events = n_tasks + n_workers  # rough upper bound on event count
    threshold_pct = (threshold / total_events) * 100 if total_events else 0
    print(f"\n  Stage 2 threshold = floor(({expected_a} + {expected_b}) / 2) "
          f"= {math.floor(threshold)}")
    print(f"  Total expected events ≈ {total_events:,}  "
          f"(Stage 2 fires after ≈ {threshold_pct:.1f}% of events)")

    if threshold_pct < 5:
        print(f"  ✗ WARN: Stage 2 fires very early ({threshold_pct:.1f}% into run). "
              f"expected_b may be too small.")
        passed = False
    elif threshold_pct > 95:
        print(f"  ✗ WARN: Stage 2 fires very late ({threshold_pct:.1f}% into run). "
              f"Stage 2 may barely activate before the horizon ends.")
        passed = False
    else:
        print(f"  ✓ Stage 2 threshold looks reasonable ({threshold_pct:.1f}% of events)")

    # ── Step 2: run a batch of trial_events to confirm Stage 2 fires ───────
    print(f"\n  Stepping through {trial_events:,} events to confirm Stage 2 fires ...")

    arrivals_before = tracker.get("arrivals", 0)
    events_before   = sim.event_count

    # Run exactly trial_events events in one shot by setting the budget limit.
    # The simulation prints "Simulation terminated: Exceeded …" when the limit
    # fires — suppress that by temporarily redirecting stdout.
    import io, sys as _sys
    sim.max_events = sim.event_count + trial_events
    _captured = io.StringIO()
    _old_stdout, _sys.stdout = _sys.stdout, _captured
    try:
        sim.step(duration_seconds=None)
    finally:
        _sys.stdout = _old_stdout

    events_processed = sim.event_count - events_before
    arrivals_at_end  = tracker.get("arrivals", 0)
    stage2_fired     = arrivals_at_end > threshold

    print(f"    Events processed: {events_processed:,}")
    print(f"    Arrivals counted: {arrivals_at_end:,}  "
          f"(threshold: {math.floor(threshold):,})")

    if stage2_fired:
        print(f"  ✓ Stage 2 fired during the {trial_events:,}-event trial window "
              f"(arrivals reached {arrivals_at_end:,} > threshold {math.floor(threshold):,})")
    else:
        eta_events = int((threshold - arrivals_at_end) / max(arrivals_at_end, 1)
                         * events_processed)
        print(f"  ○ Stage 2 not yet reached (need {int(threshold - arrivals_at_end):,} "
              f"more arrivals; ~{eta_events:,} more events at current rate).")
        print(f"    Try --events {trial_events * 3} for a longer trial window.")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--didi-only",    action="store_true")
    parser.add_argument("--gowalla-only", action="store_true")
    parser.add_argument("--events",       type=int, default=DEFAULT_TRIAL_EVENTS,
                        help=f"Events to step for Stage 2 check (default: {DEFAULT_TRIAL_EVENTS})")
    args = parser.parse_args()

    run_didi    = not args.gowalla_only
    run_gowalla = not args.didi_only

    print("=" * 65)
    print("  ONRTA-OP  expected_a / expected_b injection verification")
    print("=" * 65)

    results: Dict[str, bool] = {}

    if run_gowalla:
        print("\n  Loading Gowalla ...", end="  ", flush=True)
        workers_g, tasks_g = load_workers_tasks(
            "gowalla", root_path=GOWALLA_ROOT, **GOWALLA_KWARGS
        )
        print(f"{len(workers_g):,} workers | {len(tasks_g):,} tasks")
        results["Gowalla"] = verify_dataset(
            "Gowalla Austin (Sep 2010)", workers_g, tasks_g, args.events
        )

    if run_didi:
        day_path = os.path.join(DIDI_ROOT, DIDI_DAY)
        print(f"\n  Loading Didi {DIDI_DAY} ...", end="  ", flush=True)
        workers_d, tasks_d = load_workers_tasks("didi", root_path=day_path)
        print(f"{len(workers_d):,} workers | {len(tasks_d):,} tasks")
        results["Didi"] = verify_dataset(
            "Didi Chengdu 20161109", workers_d, tasks_d, args.events
        )

    print(f"\n{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}")
    for ds, ok in results.items():
        status = "PASS ✓" if ok else "WARN / FAIL ✗"
        print(f"  {ds:<20} injection check  →  {status}")
    print()


if __name__ == "__main__":
    main()
