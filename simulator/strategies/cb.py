"""
Cost-Balancing (CB) batch assignment baseline.

Defers tasks and workers until matching cost M is bounded by alpha * waiting cost W,
then runs a greedy longest-wait-first k-NN batch match. Contrasts with immediate
heuristics (greedy, composite) and fixed-window batching (mmd_batch).

Reference: online cost-balancing for on-demand spatial delivery (Section 7.2 style).
2601.21858v1.pdf
"""

from __future__ import annotations

from typing import Any, List, Tuple

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0


def _commit_assignment(task, worker, now):
    pickup_distance = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    drop_distance = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    task.pickup_km = pickup_distance
    task.drop_km = drop_distance

    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH

    task.start_time = now + (pickup_travel_hours * 3600.0)
    task.finish_time = task.start_time + (service_travel_hours * 3600.0)

    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _is_feasible(worker, task, now: float, d_pick: float, drop_distance: float) -> bool:
    pickup_eta = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    finish_eta = now + ((d_pick + drop_distance) / AVG_SPEED_KMH) * 3600.0
    return pickup_eta <= task.expire_time and finish_eta <= worker.deadline


def _evaluate_cb_condition(state, now: float, alpha: float) -> bool:
    """Return True when accumulated waiting cost justifies the current matching cost."""
    if not state.deferred_tasks or not state.available_workers:
        return False

    # W: total elapsed wait across deferred tasks (seconds).
    W = sum(now - task.release_time for task in state.deferred_tasks)

    # M: average nearest-neighbor pickup distance (km) as market-thickness proxy.
    total_d_pick = 0.0
    for task in state.deferred_tasks:
        nearest_workers = state.spatial_index.query_k_nearest(
            task.pickup_lat, task.pickup_lon, 1
        )
        if nearest_workers:
            w = nearest_workers[0]
            total_d_pick += fast_manhattan_km(
                w.start_lat, w.start_lon, task.pickup_lat, task.pickup_lon
            )

    M = total_d_pick / len(state.deferred_tasks)
    return M <= (alpha * W)


def _execute_batch_match(state, now: float, k: int = 10) -> List[Tuple[Any, Any, float]]:
    """Greedy longest-wait-first k-NN batch match when the CB condition fires."""
    assignments: List[Tuple[Any, Any, float]] = []
    sorted_tasks = sorted(state.deferred_tasks, key=lambda t: t.release_time)

    for task in sorted_tasks:
        if not state.available_workers:
            break

        drop_distance = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )
        nearest_workers = state.spatial_index.query_k_nearest(
            task.pickup_lat, task.pickup_lon, k
        )
        best_worker = None
        best_d_pick = float("inf")

        for worker in nearest_workers:
            if worker not in state.available_workers:
                continue
            d_pick = fast_manhattan_km(
                worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
            )
            if d_pick < best_d_pick and _is_feasible(worker, task, now, d_pick, drop_distance):
                best_d_pick = d_pick
                best_worker = worker

        if best_worker is not None:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_d_pick))

    return assignments


def _maybe_dispatch(state, now: float, alpha: float, k: int) -> List[Tuple[Any, Any, float]]:
    if _evaluate_cb_condition(state, now, alpha):
        return _execute_batch_match(state, now, k=k)
    return []


def assign_new_tasks_cb(
    state,
    now,
    tasks_to_assign,
    alpha: float = 0.5,
    k: int = 10,
    expiry_scheduler=None,
    deferral_tracker=None,
    **_,
):
    """Defer incoming tasks, then dispatch if the cost-balance condition is met."""
    for task in tasks_to_assign:
        if state.defer_task(task, now):
            if expiry_scheduler:
                expiry_scheduler(task)
            if deferral_tracker:
                deferral_tracker.record_deferral(str(task.id), now, 0.0, "no_candidates")

    return _maybe_dispatch(state, now, alpha, k)


def match_worker_cb(state, now, worker, alpha: float = 0.5, k: int = 10, **_):
    """Re-evaluate the CB condition when a worker becomes available."""
    out = _maybe_dispatch(state, now, alpha, k)
    return out if out else None


@register("cost_balancing")
def get_cb_handlers():
    return {
        "NEW_TASK": assign_new_tasks_cb,
        "FREE_WORKER": match_worker_cb,
    }
