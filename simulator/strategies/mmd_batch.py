"""
Batch bipartite matching baseline inspired by CR-11-style MMD-SC (minimize bottleneck delay).

We approximate min-max total delay by raising per-pair delay to a power before running the
classic Hungarian algorithm (linear sum assignment), which minimizes sum — a standard
heuristic for bottleneck problems.

Uses the same feasibility rules and travel model as greedy/composite (30 km/h Manhattan).
Does not use composite deferral; infeasible pairs receive infinite cost.

Reference: CR-11 (MMD-SC); implementation is a practical batch-matching baseline for this
simulator's event interface, not a full HST online algorithm.
"""

from __future__ import annotations

import numpy as np

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30.0
_INFEASIBLE_COST = 1e18
_DELAY_POWER = 3.0  # Sharpen bottleneck emphasis (CR-11 narrative: penalize long delays)


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


def _pair_delay_seconds(worker, task, now: float) -> float | None:
    """
    Total delay from task release until completion if this pair is assigned at `now`.
    Returns None if infeasible (cannot pick up before expiry or finish before worker deadline).
    """
    d_pick = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    d_drop = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    pickup_eta = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    finish_eta = now + ((d_pick + d_drop) / AVG_SPEED_KMH) * 3600.0

    if pickup_eta > task.expire_time or finish_eta > worker.deadline:
        return None

    wait = now - task.release_time
    travel_pick = (d_pick / AVG_SPEED_KMH) * 3600.0
    travel_drop = (d_drop / AVG_SPEED_KMH) * 3600.0
    return wait + travel_pick + travel_drop


def _batch_min_sum_power_delay(state, now: float, **_ignore) -> list:
    """
    Match all available workers to pending tasks (active + deferred) using Hungarian on powered delay costs.
    Commits every feasible matched pair and returns list of (task, worker, float) tuples.
    """
    workers = list(state.available_workers)
    tasks = list(state.active_tasks) + list(state.deferred_tasks)

    if not workers or not tasks:
        return []

    n_w, n_t = len(workers), len(tasks)
    raw = np.full((n_w, n_t), _INFEASIBLE_COST, dtype=np.float64)

    for i, w in enumerate(workers):
        for j, t in enumerate(tasks):
            dsec = _pair_delay_seconds(w, t, now)
            if dsec is not None:
                raw[i, j] = max(0.0, float(dsec))

    # Min-max emphasis: minimize sum of delay^p
    powered = np.where(raw >= _INFEASIBLE_COST / 2, _INFEASIBLE_COST, np.power(raw, _DELAY_POWER))

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(powered)

    assignments: list = []
    used_workers: set = set()
    used_tasks: set = set()

    for i, j in zip(row_ind, col_ind):
        if powered[i, j] >= _INFEASIBLE_COST / 2:
            continue
        wm, tm = workers[i], tasks[j]
        if wm in used_workers or tm in used_tasks:
            continue
        if tm.assigned or (tm not in state.active_tasks and tm not in state.deferred_tasks):
            continue
        if wm not in state.available_workers:
            continue

        _commit_assignment(tm, wm, now)
        state.assign_task(tm, wm)
        used_workers.add(wm)
        used_tasks.add(tm)
        assignments.append((tm, wm, float(raw[i, j])))

    return assignments


def assign_new_tasks_mmd_batch(state, now, tasks_to_assign, **_kwargs):
    """
    After TASK_RELEASE, re-run global batch matching on the full snapshot.
    tasks_to_assign is ignored for matching (all of active_tasks already includes new tasks).
    """
    return _batch_min_sum_power_delay(state, now)


def match_worker_mmd_batch(state, now, worker, **_kwargs):
    """
    After a worker becomes free, re-run global batch matching.
    Returns a list of assignments (simulator iterates); order does not matter.
    """
    out = _batch_min_sum_power_delay(state, now)
    return out if out else None


@register("mmd_batch")
def get_mmd_batch_handlers():
    return {
        "NEW_TASK": assign_new_tasks_mmd_batch,
        "FREE_WORKER": match_worker_mmd_batch,
    }
