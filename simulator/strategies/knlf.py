"""
k-Nearest Least-First (k-NLF) Strategy

An O(k) fairness signal that directly optimises task-count equity by querying
the k nearest workers via the spatial index and then selecting the candidate
with the fewest completed tasks.  Distance is used only as a tie-breaker.

Research Context:
-----------------
EWMA idle time is a poor proxy for allocation fairness in spatial settings because
it conflates spatial inaccessibility (worker is in a sparse area) with assignment
unfairness.  LAF fixes this by using task count directly, but scans all W workers
— trading away the spatial constraint and causing long wait times.

k-NLF combines the best of both:
  - Spatial constraint preserved: only the k geographically nearest workers are
    considered, so pickup distances and wait times stay close to Greedy.
  - Direct fairness signal: within the candidate set, the worker who has completed
    the fewest tasks wins — the same quantity that Jain's Fairness Index measures.

Complexity:  O(k log k) per NEW_TASK event (k-NN query + sort) — same class as Greedy.

NEW_TASK  → query k-nearest workers → pick feasible worker with min(completed_tasks)
FREE_WORKER → nearest feasible pending task (greedy, same as baseline)
"""

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30


def _commit_assignment(task, worker, now):
    pickup_km = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    drop_km = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )
    task.pickup_km = pickup_km
    task.drop_km = drop_km
    task.start_time = now + (pickup_km / AVG_SPEED_KMH) * 3600
    task.finish_time = task.start_time + (drop_km / AVG_SPEED_KMH) * 3600
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _is_feasible(worker, task, now, pickup_km, drop_km):
    pickup_eta = now + (pickup_km / AVG_SPEED_KMH) * 3600
    finish_eta = now + ((pickup_km + drop_km) / AVG_SPEED_KMH) * 3600
    return pickup_eta <= task.expire_time and finish_eta <= worker.deadline


def _defer(state, task, now, kwargs):
    """Defer a task and schedule expiry + record in deferral tracker."""
    if state.defer_task(task, now):
        expiry_scheduler = kwargs.get("expiry_scheduler")
        if expiry_scheduler:
            expiry_scheduler(task)
        deferral_tracker = kwargs.get("deferral_tracker")
        if deferral_tracker:
            deferral_tracker.record_deferral(str(task.id), now, 0.0, "no_k_candidates")


def assign_new_tasks_knlf(state, now, tasks_to_assign, k=15, **_):
    """
    NEW_TASK handler: for each arriving task, query k nearest workers via the
    spatial index and assign to the feasible candidate with the fewest completed
    tasks.  Ties broken by ascending pickup distance (closest wins).
    """
    assignments = []

    drop_km_cache = {}

    for task in tasks_to_assign:
        if not state.available_workers:
            _defer(state, task, now, _)
            continue

        nearest = state.spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
        if not nearest:
            _defer(state, task, now, _)
            continue

        task_key = id(task)
        if task_key not in drop_km_cache:
            drop_km_cache[task_key] = fast_manhattan_km(
                task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
            )
        drop_km = drop_km_cache[task_key]

        best_worker = None
        best_tasks  = float("inf")
        best_dist   = float("inf")

        for worker in nearest:
            pickup_km = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            if not _is_feasible(worker, task, now, pickup_km, drop_km):
                continue

            # Primary key: fewest completed tasks; secondary: closest distance
            if (worker.completed_tasks < best_tasks or
                    (worker.completed_tasks == best_tasks and pickup_km < best_dist)):
                best_tasks  = worker.completed_tasks
                best_dist   = pickup_km
                best_worker = worker

        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            _defer(state, task, now, _)

    return assignments


def match_worker_knlf(state, now, worker, **_):
    """
    FREE_WORKER handler: greedy nearest-task selection (same as Greedy baseline).

    The fairness signal applies at task-assignment time (when a task chooses among
    k workers).  On worker release we simply find the closest pending task —
    this keeps pickup distances low and is symmetric with Greedy's FREE_WORKER.

    Scans both deferred_tasks and active_tasks so no pending task is missed.
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


@register("knlf")
def get_knlf_handlers():
    return {
        "NEW_TASK":    assign_new_tasks_knlf,
        "FREE_WORKER": match_worker_knlf,
    }
