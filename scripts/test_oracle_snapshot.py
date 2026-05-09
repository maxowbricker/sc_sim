#!/usr/bin/env python3
"""
Oracle snapshot correctness test.

Strategy:
  For every 5-minute step across a full simulated day:
    1. Take a snapshot_state() and a deepcopy() of the simulator (identical
       pre-step state captured two ways).
    2. Run one greedy step — this mutates the simulator exactly as the Oracle
       will in production.
    3. Restore from snapshot_state().
    4. Compare the restored simulator against the deepcopy field-by-field.
    5. Advance the real step (composite / greedy, doesn't matter for this
       test) so the loop progresses through the whole day.

Pass criterion: zero mismatches across all 192+ step checks.

Usage:
    cd /path/to/sc_sim
    conda activate sc
    python scripts/test_oracle_snapshot.py
"""

import sys
import os
import copy
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_workers_tasks
from simulator.simulation import EventSimulator
from config import get_simulation_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "./data/didi/full_didi_gaia"
STEP_SECONDS = 5 * 60           # 5-minute steps
WARMUP_SECONDS = 30 * 60        # 30-minute greedy warmup before RL phase
EPISODE_SECONDS = 8 * 60 * 60   # 8-hour RL phase → ~96 steps
# We run the full day (warmup included) so the dataset event queue is natural.

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _fmt(val):
    """Compact repr for diffs — truncate long lists."""
    if isinstance(val, list) and len(val) > 6:
        return f"[{val[:3]} ... {val[-3:]}] (len={len(val)})"
    return repr(val)


def compare_sims(restored, ref):
    """
    Compare every meaningful mutable field between restored and ref simulators.
    Returns a list of (field_name, restored_val, ref_val) tuples for any mismatch.
    """
    mismatches = []

    def check(name, a, b):
        if a != b:
            mismatches.append((name, _fmt(a), _fmt(b)))

    # --- Top-level EventSimulator scalars ---
    check("current_time", restored.current_time, ref.current_time)
    check("end_time", restored.end_time, ref.end_time)
    check("event_count", restored.event_count, ref.event_count)
    check("step_start_time", restored.step_start_time, ref.step_start_time)

    # --- Event queue (sort so heap-order differences don't matter) ---
    check("event_queue", sorted(restored.event_queue), sorted(ref.event_queue))

    # --- StateManager pool sets (compare as sets of IDs) ---
    def pool_ids(pool):
        return {obj.id for obj in pool}

    state_r = restored.state
    state_ref = ref.state

    check("available_workers", pool_ids(state_r.available_workers), pool_ids(state_ref.available_workers))
    check("active_tasks",      pool_ids(state_r.active_tasks),      pool_ids(state_ref.active_tasks))
    check("deferred_tasks",    pool_ids(state_r.deferred_tasks),    pool_ids(state_ref.deferred_tasks))
    check("assigned_tasks",    pool_ids(state_r.assigned_tasks),    pool_ids(state_ref.assigned_tasks))
    check("assigned_workers",  pool_ids(state_r.assigned_workers),  pool_ids(state_ref.assigned_workers))
    check("completed_tasks",   pool_ids(state_r.completed_tasks),   pool_ids(state_ref.completed_tasks))

    # --- Per-worker primitive fields ---
    for w_id, w_r in state_r.all_workers_map.items():
        w_ref = state_ref.all_workers_map.get(w_id)
        if w_ref is None:
            mismatches.append((f"worker[{w_id}]", "EXISTS", "MISSING"))
            continue
        wr_dict = w_r.get_state_dict()
        wref_dict = w_ref.get_state_dict()
        for field, val_r in wr_dict.items():
            val_ref = wref_dict.get(field)
            if val_r != val_ref:
                mismatches.append((f"worker[{w_id}].{field}", _fmt(val_r), _fmt(val_ref)))

    # --- Per-task primitive fields ---
    for t_id, t_r in state_r.all_tasks_map.items():
        t_ref = state_ref.all_tasks_map.get(t_id)
        if t_ref is None:
            mismatches.append((f"task[{t_id}]", "EXISTS", "MISSING"))
            continue
        tr_dict = t_r.get_state_dict()
        tref_dict = t_ref.get_state_dict()
        for field, val_r in tr_dict.items():
            val_ref = tref_dict.get(field)
            if val_r != val_ref:
                mismatches.append((f"task[{t_id}].{field}", _fmt(val_r), _fmt(val_ref)))

    # --- MetricsManager: compare via snapshot dicts (primitives only) ---
    snap_r   = restored.metrics.snapshot_metrics()
    snap_ref = ref.metrics.snapshot_metrics()

    for key, val_r in snap_r.items():
        val_ref = snap_ref.get(key)
        # For nested dicts (e.g. _summary_minimal) recurse one level
        if isinstance(val_r, dict):
            for sub_key, sub_r in val_r.items():
                sub_ref = val_ref.get(sub_key) if isinstance(val_ref, dict) else None
                if sub_r != sub_ref:
                    mismatches.append((f"metrics.{key}.{sub_key}", _fmt(sub_r), _fmt(sub_ref)))
        else:
            if val_r != val_ref:
                mismatches.append((f"metrics.{key}", _fmt(val_r), _fmt(val_ref)))

    return mismatches


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_all_days(data_path):
    """Return list of all day folder paths in chronological order."""
    day_folders = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])
    return [os.path.join(data_path, d) for d in day_folders]


def load_day(day_path):
    """Load workers and tasks from a single day folder."""
    day_name = os.path.basename(day_path)
    print(f"  📂 Loading day: {day_name}")
    workers, tasks = load_workers_tasks("didi", root_path=day_path)
    print(f"     Workers: {len(workers):,}  Tasks: {len(tasks):,}")
    return workers, tasks


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------

def run_test():
    print("=" * 80)
    print("Oracle Snapshot Correctness Test — All Days")
    print("=" * 80)

    day_paths = get_all_days(DATA_PATH)
    if not day_paths:
        print(f"❌ No day folders found in {DATA_PATH}")
        return

    print(f"\n📋 Found {len(day_paths)} days to test.\n")

    all_day_results = []

    for day_idx, day_path in enumerate(day_paths, 1):
        day_name = os.path.basename(day_path)
        print(f"\n{'=' * 80}")
        print(f"Day {day_idx}/{len(day_paths)}: {day_name}")
        print("=" * 80)

        try:
            workers, tasks = load_day(day_path)
        except Exception as e:
            print(f"  ❌ Failed to load day: {e}")
            all_day_results.append({
                'day': day_name,
                'status': 'FAILED_LOAD',
                'error': str(e)
            })
            continue

        day_result = run_day_test(day_idx, len(day_paths), day_name, workers, tasks)
        all_day_results.append(day_result)

    # --- Global summary ---
    print("\n\n" + "=" * 80)
    print("GLOBAL SUMMARY — All Days")
    print("=" * 80)
    total_steps = 0
    total_mismatches = 0
    passed_days = 0
    failed_days = 0

    for result in all_day_results:
        status_icon = "✅" if result['status'] == 'PASS' else "❌"
        if result['status'] == 'PASS':
            passed_days += 1
            total_steps += result['steps']
            total_mismatches += result['mismatches']
            print(
                f"{status_icon} {result['day']:<40}  "
                f"Steps: {result['steps']:>3}  Mismatches: {result['mismatches']:>3}  "
                f"Snap: {result['snap_avg']:>6.2f}ms  Restore: {result['restore_avg']:>6.2f}ms"
            )
        else:
            failed_days += 1
            print(f"{status_icon} {result['day']:<40}  {result.get('error', 'UNKNOWN')}")

    print("\n" + "-" * 80)
    print(f"Total days tested   : {len(all_day_results)}")
    print(f"  Passed            : {passed_days}")
    print(f"  Failed            : {failed_days}")
    if passed_days > 0:
        print(f"Total steps checked : {total_steps}")
        print(f"Total mismatches    : {total_mismatches}")
        print(f"Avg snapshot time   : {sum(r['snap_avg'] for r in all_day_results if r['status'] == 'PASS') / passed_days:.2f} ms/step")
        print(f"Avg restore time    : {sum(r['restore_avg'] for r in all_day_results if r['status'] == 'PASS') / passed_days:.2f} ms/step")
    print("=" * 80)

    if total_mismatches == 0 and failed_days == 0:
        print("\n🎉 ALL DAYS PASSED — snapshot/restore is correct across the entire dataset!")
    else:
        print(f"\n⚠️  Some days had issues. Review above for details.")
    print("=" * 80 + "\n")


def run_day_test(day_idx, total_days, day_name, workers, tasks):
    """Run oracle snapshot test on a single day."""

    sim_config = get_simulation_config()
    sim_config["assignment_strategy"] = "greedy"  # start with greedy for warmup

    sim = EventSimulator(workers, tasks, sim_config)
    sim.reset()

    print(f"\nSimulator initialised. Start time: {sim.current_time:.0f}")

    # --- Warmup (greedy, no snapshotting) ---
    warmup_end = sim.current_time + WARMUP_SECONDS
    print(f"  ⏳ Running {WARMUP_SECONDS // 60}-minute greedy warmup …")
    warmup_steps = 0
    while sim.current_time < warmup_end:
        done = sim.step(duration_seconds=STEP_SECONDS)
        warmup_steps += 1
        if done:
            print("  ⚠️  Simulation ended during warmup — day data exhausted early.")
            return {
                'day': day_name,
                'status': 'COMPLETED_EARLY',
                'steps': 0,
                'mismatches': 0,
                'snap_avg': 0,
                'restore_avg': 0
            }
    print(f"  ✅ Warmup complete after {warmup_steps} steps.\n")

    # Switch to composite for the "real" steps (weights don't affect correctness)
    sim.switch_strategy("composite")

    # --- Step loop ---
    episode_end = sim.current_time + EPISODE_SECONDS
    step = 0
    total_mismatches = 0
    total_time_snap  = 0.0
    total_time_restore = 0.0
    total_time_compare = 0.0

    print(f"{'Step':>5}  {'Time':>10}  {'Snap(ms)':>10}  {'Restore(ms)':>12}  {'Cmp(ms)':>9}  {'Result':>8}")
    print("-" * 65)

    while sim.current_time < episode_end:
        step += 1

        # 1. Snapshot (our implementation)
        t0 = time.perf_counter()
        snap = sim.snapshot_state()
        t1 = time.perf_counter()
        snap_ms = (t1 - t0) * 1000

        # 2. Deep copy (ground truth — the expensive reference)
        ref = copy.deepcopy(sim)

        # 3. Run greedy oracle step (mutates sim)
        sim.switch_strategy("greedy")
        done = sim.step(duration_seconds=STEP_SECONDS)

        # 4. Restore from our snapshot
        t2 = time.perf_counter()
        sim.restore_state(snap)
        t3 = time.perf_counter()
        restore_ms = (t3 - t2) * 1000

        # Switch strategy back so the comparison isn't tainted by handler state
        sim.switch_strategy("composite")

        # 5. Compare restored sim vs deepcopy ref
        t4 = time.perf_counter()
        mismatches = compare_sims(sim, ref)
        t5 = time.perf_counter()
        cmp_ms = (t5 - t4) * 1000

        total_time_snap    += snap_ms
        total_time_restore += restore_ms
        total_time_compare += cmp_ms

        result_str = "PASS" if not mismatches else f"FAIL({len(mismatches)})"
        print(f"{step:>5}  {sim.current_time:>10.0f}  {snap_ms:>10.2f}  {restore_ms:>12.2f}  {cmp_ms:>9.2f}  {result_str:>8}")

        if mismatches:
            total_mismatches += len(mismatches)
            print(f"        ↳  First 5 mismatches:")
            for name, v_r, v_ref in mismatches[:5]:
                print(f"           {name}")
                print(f"             restored: {v_r}")
                print(f"             ref:      {v_ref}")

        # 6. Advance the real composite step
        done = sim.step(duration_seconds=STEP_SECONDS)
        if done:
            print(f"\n  ℹ️  Simulation completed all events — stopping after {step} steps.")
            break

    # --- Day summary ---
    print("\n" + "-" * 65)
    print(f"Day {day_idx}/{total_days} Result")
    print("-" * 65)
    print(f"  Steps checked     : {step}")
    print(f"  Total mismatches  : {total_mismatches}")
    if step > 0:
        print(f"  Snapshot avg      : {total_time_snap / step:.2f} ms/step")
        print(f"  Restore avg       : {total_time_restore / step:.2f} ms/step")
        print(f"  Compare avg       : {total_time_compare / step:.2f} ms/step")

    if total_mismatches == 0:
        print(f"  Status            : ✅ PASS")
        return {
            'day': day_name,
            'status': 'PASS',
            'steps': step,
            'mismatches': 0,
            'snap_avg': total_time_snap / step if step > 0 else 0,
            'restore_avg': total_time_restore / step if step > 0 else 0
        }
    else:
        print(f"  Status            : ❌ FAIL")
        return {
            'day': day_name,
            'status': 'FAIL',
            'steps': step,
            'mismatches': total_mismatches,
            'snap_avg': total_time_snap / step if step > 0 else 0,
            'restore_avg': total_time_restore / step if step > 0 else 0
        }


if __name__ == "__main__":
    run_test()
