"""
k-Nearest Temporal Fairness (k-NTF) Strategy Variants

Addresses the "Billy vs John" problem in raw task-count fairness (k-NLF):

    Billy: 5 hours online, 8 tasks  → rate = 1.6 tasks/hr
    John:  2 hours online, 5 tasks  → rate = 2.5 tasks/hr

    Under k-NLF (raw task count), John wins (5 < 8).
    But John is earning tasks *faster* — Billy is being systematically under-served.

Both variants use the same two-phase spatial+fairness architecture as k-NLF:

    Phase 1 (Spatial, O(k)):   query k nearest available workers via spatial index
    Phase 2 (Temporal, O(k)):  sort candidates by a time-normalised fairness metric

This preserves O(k) complexity and wait-time efficiency while correcting the
raw-count bias against long-shift workers.

--- Strategy Variants ---

k-NTF-EPH   Economic Fairness ("Earnings Per Hour")
    Sort by ascending EPH = total_earnings / shift_elapsed
    The worker who has earned the least per hour of work gets priority.
    Directly addresses revenue inequality across different task lengths/values.

k-NTF-IR    Temporal Fairness ("Idle Ratio")
    Sort by descending IR = live_idle_time / shift_elapsed
    The worker who has spent the largest *fraction* of their shift waiting gets priority.
    Closest to the original EWMA intent, but time-normalised so shift length is fair.
    Directly addresses "Silent Labour" — workers who are available but not receiving tasks.

FREE_WORKER handler (both variants): identical to Greedy/k-NLF — nearest feasible task.
"""

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30
_TINY = 1e-6   # Guard against division-by-zero at shift start


def _defer(state, task, now, kwargs):
    """Defer a task and schedule expiry + record in deferral tracker."""
    if state.defer_task(task, now):
        expiry_scheduler = kwargs.get("expiry_scheduler")
        if expiry_scheduler:
            expiry_scheduler(task)
        deferral_tracker = kwargs.get("deferral_tracker")
        if deferral_tracker:
            deferral_tracker.record_deferral(str(task.id), now, 0.0, "no_k_candidates")


def _commit_assignment(task, worker, now):
    pickup_km = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    drop_km = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )
    task.pickup_km = pickup_km
    task.drop_km   = drop_km
    task.start_time  = now + (pickup_km / AVG_SPEED_KMH) * 3600
    task.finish_time = task.start_time + (drop_km / AVG_SPEED_KMH) * 3600
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _is_feasible(worker, task, now, pickup_km, drop_km):
    pickup_eta = now + (pickup_km / AVG_SPEED_KMH) * 3600
    finish_eta = now + ((pickup_km + drop_km) / AVG_SPEED_KMH) * 3600
    return pickup_eta <= task.expire_time and finish_eta <= worker.deadline


def _earnings_per_hour(worker, now):
    """EPH = total_earnings / hours_online.  Lower = more under-served economically."""
    shift_elapsed_h = (now - worker.release_time) / 3600.0
    return worker.total_earnings / max(shift_elapsed_h, _TINY)


def _idle_ratio(worker, now):
    """
    IR = live_idle_time / shift_elapsed.  Higher = more time spent waiting.

    'Live' idle adds the time elapsed since the last snapshot update so the metric
    stays accurate even between MetricsManager snapshot steps.
    """
    shift_elapsed = max(now - worker.release_time, _TINY)
    last_update = worker.last_state_ts if worker.last_state_ts is not None else worker.release_time
    live_idle = worker.total_idle_time + max(0.0, now - last_update)
    return live_idle / shift_elapsed


# ---------------------------------------------------------------------------
# k-NTF-EPH  (Economic Fairness)
# ---------------------------------------------------------------------------

def assign_new_tasks_kntf_eph(state, now, tasks_to_assign, k=15, **_):
    """
    NEW_TASK: query k nearest workers, assign to the one with the lowest
    Earnings Per Hour.  Distance is the tie-breaker.
    """
    assignments = []

    for task in tasks_to_assign:
        if not state.available_workers:
            _defer(state, task, now, _)
            continue

        nearest = state.spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
        if not nearest:
            _defer(state, task, now, _)
            continue

        drop_km = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )

        best_worker  = None
        best_eph     = float("inf")
        best_dist    = float("inf")

        for worker in nearest:
            pickup_km = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            if not _is_feasible(worker, task, now, pickup_km, drop_km):
                continue

            eph = _earnings_per_hour(worker, now)

            if (eph < best_eph or (eph == best_eph and pickup_km < best_dist)):
                best_eph    = eph
                best_dist   = pickup_km
                best_worker = worker

        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            _defer(state, task, now, _)

    return assignments


def match_worker_kntf(state, now, worker, **_):
    """
    FREE_WORKER: greedy nearest-task selection (same as Greedy / k-NLF).
    The temporal fairness signal applies at task-assignment time only.
    """
    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    best_task = None
    best_dist = float("inf")

    for task in pending:
        pickup_km = fast_manhattan_km(
            worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
        )
        drop_km = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )
        if not _is_feasible(worker, task, now, pickup_km, drop_km):
            continue
        if pickup_km < best_dist:
            best_dist = pickup_km
            best_task = task

    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_dist)

    return None


# ---------------------------------------------------------------------------
# k-NTF-IR  (Temporal / Idle-Ratio Fairness)
# ---------------------------------------------------------------------------

def assign_new_tasks_kntf_ir(state, now, tasks_to_assign, k=15, **_):
    """
    NEW_TASK: query k nearest workers, assign to the one with the highest
    Idle Ratio (fraction of shift spent waiting).  Distance is the tie-breaker.
    """
    assignments = []

    for task in tasks_to_assign:
        if not state.available_workers:
            _defer(state, task, now, _)
            continue

        nearest = state.spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
        if not nearest:
            _defer(state, task, now, _)
            continue

        drop_km = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )

        best_worker = None
        best_ir     = float("-inf")
        best_dist   = float("inf")

        for worker in nearest:
            pickup_km = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            if not _is_feasible(worker, task, now, pickup_km, drop_km):
                continue

            ir = _idle_ratio(worker, now)

            if (ir > best_ir or (ir == best_ir and pickup_km < best_dist)):
                best_ir     = ir
                best_dist   = pickup_km
                best_worker = worker

        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            _defer(state, task, now, _)

    return assignments


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@register("kntf_eph")
def get_kntf_eph_handlers():
    """k-Nearest Temporal Fairness — Economic (Earnings Per Hour)."""
    return {
        "NEW_TASK":    assign_new_tasks_kntf_eph,
        "FREE_WORKER": match_worker_kntf,
    }


@register("kntf_ir")
def get_kntf_ir_handlers():
    """k-Nearest Temporal Fairness — Temporal (Idle Ratio)."""
    return {
        "NEW_TASK":    assign_new_tasks_kntf_ir,
        "FREE_WORKER": match_worker_kntf,
    }
