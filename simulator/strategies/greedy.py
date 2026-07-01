"""
Greedy baseline strategy.

Assigns each arriving task to the nearest feasible available worker via an
O(|W|) linear scan of the full fleet.  No fairness signal — purely proximity-
optimal.  Serves as the throughput and wait-time anchor for all comparisons.

NEW_TASK  → O(|W|) scan → nearest feasible worker
FREE_WORKER → nearest feasible pending task (deferred + active)
"""

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km
from simulator.behavior import evaluate_worker_acceptance
from typing import Optional, Dict, Any, List

AVG_SPEED_KMH = 30

def _commit_assignment(task, worker, now):
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    task.start_time = now + (pickup_travel_hours * 3600)
    task.finish_time = task.start_time + (service_travel_hours * 3600)
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _is_feasible_greedy(worker, task, now, pickup_dist: float, drop_dist: float) -> bool:
    pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
    finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)
    return pickup_eta <= task.expire_time and finish_eta <= worker.deadline


def _record_offer(state, accepted: bool) -> None:
    state.offers_made += 1
    if not accepted:
        state.offers_rejected += 1


def _defer(state, task, now, kwargs):
    """Defer a task and schedule expiry + record in deferral tracker."""
    if state.defer_task(task, now):
        expiry_scheduler = kwargs.get("expiry_scheduler")
        if expiry_scheduler:
            expiry_scheduler(task)
        deferral_tracker = kwargs.get("deferral_tracker")
        if deferral_tracker:
            deferral_tracker.record_deferral(str(task.id), now, 0.0, "no_candidates")


def assign_new_tasks_greedy(
    state,
    now,
    tasks_to_assign,
    worker_acceptance: Optional[Dict[str, Any]] = None,
    **_,
):
    """
    Greedy assignment for newly released tasks.

    Performs an O(|W|) linear scan over all available workers to find the
    globally nearest feasible worker for each task. This is the canonical
    definition of greedy dispatch and the version used in the paper evaluation.

    drop_dist is task-only (pickup->dropoff) so it is computed once per task
    outside the worker loop rather than redundantly once per worker.
    """
    assignments = []
    acceptance_enabled = worker_acceptance and worker_acceptance.get("enabled", False)

    for task in tasks_to_assign:
        if not state.available_workers:
            _defer(state, task, now, _)
            continue

        drop_dist = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )

        if acceptance_enabled:
            ranked: List[tuple] = []
            for worker in state.available_workers:
                pickup_dist = fast_manhattan_km(
                    worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
                )
                if not _is_feasible_greedy(worker, task, now, pickup_dist, drop_dist):
                    continue
                ranked.append((pickup_dist, worker))

            ranked.sort(key=lambda x: x[0])
            assigned = False
            for pickup_dist, worker in ranked:
                accepted = evaluate_worker_acceptance(pickup_dist, worker_acceptance)
                _record_offer(state, accepted)
                if accepted:
                    assigned_task = _commit_assignment(task, worker, now)
                    state.assign_task(assigned_task, worker)
                    assignments.append((assigned_task, worker, pickup_dist))
                    assigned = True
                    break

            if not assigned:
                _defer(state, task, now, _)
            continue

        # O(|W|) scan: find the globally nearest feasible worker
        best_worker, best_dist = None, float("inf")
        for worker in state.available_workers:
            pickup_dist = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            if not _is_feasible_greedy(worker, task, now, pickup_dist, drop_dist):
                continue
            if pickup_dist < best_dist:
                best_dist = pickup_dist
                best_worker = worker

        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            _defer(state, task, now, _)

    return assignments


def match_worker_greedy(
    state,
    now,
    worker,
    worker_acceptance: Optional[Dict[str, Any]] = None,
    **_,
):
    """
    Greedy matching for a newly available worker. Finds the nearest task
    from the combined pool of deferred and active unassigned tasks.

    Greedy's NEW_TASK handler explicitly calls state.defer_task() when no
    feasible worker is found, so pending tasks accumulate in deferred_tasks.
    Active_tasks is also included to catch any tasks that arrived at the same
    simulation timestep and have not yet been processed.
    """
    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    acceptance_enabled = worker_acceptance and worker_acceptance.get("enabled", False)

    if acceptance_enabled:
        ranked: List[tuple] = []
        for task in pending:
            pickup_dist = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            drop_dist = fast_manhattan_km(
                task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
            )
            if not _is_feasible_greedy(worker, task, now, pickup_dist, drop_dist):
                continue
            ranked.append((pickup_dist, task))

        ranked.sort(key=lambda x: x[0])
        for pickup_dist, task in ranked:
            accepted = evaluate_worker_acceptance(pickup_dist, worker_acceptance)
            _record_offer(state, accepted)
            if accepted:
                assigned_task = _commit_assignment(task, worker, now)
                state.assign_task(assigned_task, worker)
                return (assigned_task, worker, pickup_dist)

        return None

    best_task, best_dist = None, float("inf")

    for task in pending:
        pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)

        if not _is_feasible_greedy(worker, task, now, pickup_dist, drop_dist):
            continue

        if pickup_dist < best_dist:
            best_dist = pickup_dist
            best_task = task

    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_dist)

    return None


@register("greedy")
def get_greedy_handlers():
    return {
        "NEW_TASK": assign_new_tasks_greedy,
        "FREE_WORKER": match_worker_greedy
    }
