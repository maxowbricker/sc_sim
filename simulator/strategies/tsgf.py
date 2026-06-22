"""
Two-Sided Group Fairness (TSGF) randomized policy sampling baseline.

Online adaptation of the TSGF-KIID sampling framework: instead of a scalarized
composite score (as in composite.py), each dispatch event samples one of three
pure sub-heuristics — operator profit, max-min worker fairness, max-min task
fairness — or explicitly defers to accumulate market thickness.

Reference: TSGF framework for three-sided spatial matching (operator / offline /
online). Full offline LP + dependent rounding is omitted; this module implements
the paper's runtime sampling mechanism for the event-driven simulator.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0

_rng = random.Random()
_dispatch_counter = 0


def _next_roll(seed: int) -> float:
    """Deterministic, monotonic sampling without resetting RNG state each event."""
    global _dispatch_counter
    _dispatch_counter += 1
    _rng.seed(seed + _dispatch_counter)
    return _rng.random()


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


def _worker_idle_seconds(worker, now: float) -> float:
    ref = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
    return now - ref


def _best_feasible_pair_for_task(
    state,
    task,
    now: float,
    k: int,
) -> Optional[Tuple[Any, Any, float]]:
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

    if best_worker is None:
        return None
    return task, best_worker, best_d_pick


def _best_feasible_pair_for_worker(
    state,
    worker,
    now: float,
    k: int,
) -> Optional[Tuple[Any, Any, float]]:
    nearest_tasks = state.deferred_task_index.query_k_nearest(
        worker.start_lat, worker.start_lon, k
    )

    best_task = None
    best_d_pick = float("inf")
    for task in nearest_tasks:
        if task not in state.deferred_tasks:
            continue
        drop_distance = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )
        d_pick = fast_manhattan_km(
            worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
        )
        if d_pick < best_d_pick and _is_feasible(worker, task, now, d_pick, drop_distance):
            best_d_pick = d_pick
            best_task = task

    if best_task is None:
        return None
    return best_task, worker, best_d_pick


def _execute_pure_greedy_match(
    state, now: float, k: int
) -> List[Tuple[Any, Any, float]]:
    """Heuristic A: minimize pickup distance across the deferred pool."""
    best: Optional[Tuple[Any, Any, float]] = None

    for task in state.deferred_tasks:
        candidate = _best_feasible_pair_for_task(state, task, now, k)
        if candidate is None:
            continue
        if best is None or candidate[2] < best[2]:
            best = candidate

    if best is None:
        return []

    task, worker, score = best
    assigned_task = _commit_assignment(task, worker, now)
    state.assign_task(assigned_task, worker)
    return [(assigned_task, worker, score)]


def _execute_worker_fairness_match(
    state, now: float, k: int
) -> List[Tuple[Any, Any, float]]:
    """Heuristic B: max-min Rawlsian — serve the most idle (starved) worker."""
    if not state.available_workers:
        return []

    starved_worker = max(state.available_workers, key=lambda w: _worker_idle_seconds(w, now))
    candidate = _best_feasible_pair_for_worker(state, starved_worker, now, k)
    if candidate is None:
        return []

    task, worker, score = candidate
    assigned_task = _commit_assignment(task, worker, now)
    state.assign_task(assigned_task, worker)
    return [(assigned_task, worker, score)]


def _execute_task_fairness_match(
    state, now: float, k: int
) -> List[Tuple[Any, Any, float]]:
    """Heuristic C: max-min Rawlsian — serve the longest-waiting deferred task."""
    if not state.deferred_tasks:
        return []

    starved_task = max(
        state.deferred_tasks,
        key=lambda t: (now - t.release_time, t.deferral_count),
    )
    candidate = _best_feasible_pair_for_task(state, starved_task, now, k)
    if candidate is None:
        return []

    task, worker, score = candidate
    assigned_task = _commit_assignment(task, worker, now)
    state.assign_task(assigned_task, worker)
    return [(assigned_task, worker, score)]


def _sample_and_dispatch(
    state,
    now: float,
    alpha: float,
    beta: float,
    gamma: float,
    k: int,
    seed: int,
) -> List[Tuple[Any, Any, float]]:
    if not state.deferred_tasks or not state.available_workers:
        return []

    roll = _next_roll(seed)

    if roll < alpha:
        return _execute_pure_greedy_match(state, now, k)
    if roll < alpha + beta:
        return _execute_worker_fairness_match(state, now, k)
    if roll < alpha + beta + gamma:
        return _execute_task_fairness_match(state, now, k)

    # Explicit deferral window (1 - alpha - beta - gamma) for market thickness.
    return []


def assign_new_tasks_tsgf(
    state,
    now,
    tasks_to_assign,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    k: int = 15,
    seed: int = 42,
    expiry_scheduler=None,
    **_,
):
    for task in tasks_to_assign:
        if state.defer_task(task, now) and expiry_scheduler:
            expiry_scheduler(task)

    return _sample_and_dispatch(state, now, alpha, beta, gamma, k, seed)


def match_worker_tsgf(
    state,
    now,
    worker,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
    k: int = 15,
    seed: int = 42,
    **_,
):
    out = _sample_and_dispatch(state, now, alpha, beta, gamma, k, seed)
    return out if out else None


@register("tsgf")
def get_tsgf_handlers():
    return {
        "NEW_TASK": assign_new_tasks_tsgf,
        "FREE_WORKER": match_worker_tsgf,
    }
