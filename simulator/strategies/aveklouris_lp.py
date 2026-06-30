"""
Discrete review LP matching baseline (Aveklouris et al. 2024).

Defers all arrivals until fixed review intervals, then solves a bipartite
assignment that maximizes spatial utility subject to task expiry and worker
deadline constraints (impatient-agent reneging).

Reference: Aveklouris, Fikioris, Trichakis, & Vaidya — "Matching Impatient
and Heterogeneous Demand and Supply". Implements the Section 7 LP-based discrete
review policy with zero holding costs (c_j^D = c_k^S = 0, Eq. 36).

Paper fidelity notes (for WISE methodology):

- Faithful elements: discrete review buffering (review period l), impatient-agent
  reneging via deadline/expiry feasibility checks, and max-weight bipartite
  matching at each review epoch. With unit-capacity workers/tasks, Hungarian
  assignment on v_jk is equivalent to the integer LP in Eq. 36.
- Zero holding cost variant: utility is purely spatial (1/(1 + d_pick)); no
  penalty for time spent waiting in queue. Performance on wait-time-heavy metrics
  should be interpreted in light of this myopic, zero-holding-cost instantiation.
- Review period l: critical hyperparameter (paper Sec. 6.2). Default
  review_period_seconds=60 must be swept/tuned per dataset arrival rate and
  expiry horizons — too long → reneging/expiry; too short → greedy-like behavior.
- Spatial adaptation: patience is modeled via dataset expire_time / worker.deadline
  plus travel-time feasibility (30 km/h Manhattan), not sampled patience draws.
  v_jk = 1/(1 + d_pick) is a spatial-crowdsourcing instantiation of matching value.
- Simulator detail: empty reviews are skipped when no backlog exists (see
  simulation.py _should_schedule_review); strict fixed-interval reviews regardless
  of queue state would differ slightly.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0
_INFEASIBLE_UTILITY = -1e18


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


def _pair_utility(worker, task, now: float) -> float | None:
    """Return v_jk matching value, or None if the impatient constraints bind."""
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

    return 1.0 / (1.0 + d_pick)


def execute_discrete_review(state, now: float, **_) -> List[Tuple[Any, Any, float]]:
    """Run LP-based matching (Hungarian on utility) at a review epoch."""
    workers = list(state.available_workers)
    tasks = list(state.deferred_tasks) + list(state.active_tasks)

    if not workers or not tasks:
        return []

    utility = np.full((len(tasks), len(workers)), _INFEASIBLE_UTILITY, dtype=np.float64)

    for i, task in enumerate(tasks):
        for j, worker in enumerate(workers):
            value = _pair_utility(worker, task, now)
            if value is not None:
                utility[i, j] = value

    # Minimize negative utility to maximize total assignment value.
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-utility)

    assignments: List[Tuple[Any, Any, float]] = []
    used_workers: set = set()
    used_tasks: set = set()

    for i, j in zip(row_ind, col_ind):
        score = utility[i, j]
        if score <= _INFEASIBLE_UTILITY / 2:
            continue

        task, worker = tasks[i], workers[j]
        if worker in used_workers or task in used_tasks:
            continue
        if task.assigned or worker not in state.available_workers:
            continue
        if task not in state.deferred_tasks and task not in state.active_tasks:
            continue

        assigned_task = _commit_assignment(task, worker, now)
        state.assign_task(assigned_task, worker)
        used_workers.add(worker)
        used_tasks.add(task)
        assignments.append((assigned_task, worker, float(score)))

    return assignments


def _schedule_review_if_needed(review_scheduler, review_period_seconds: float) -> None:
    if review_scheduler is not None:
        review_scheduler(review_period_seconds)


def assign_new_tasks_discrete_review(
    state,
    now,
    tasks_to_assign,
    review_period_seconds: float = 60.0,
    expiry_scheduler=None,
    deferral_tracker=None,
    review_scheduler=None,
    **_,
):
    """Defer arrivals and ensure a review epoch is queued."""
    for task in tasks_to_assign:
        if state.defer_task(task, now):
            if expiry_scheduler:
                expiry_scheduler(task)
            if deferral_tracker:
                deferral_tracker.record_deferral(str(task.id), now, 0.0, "no_candidates")

    _schedule_review_if_needed(review_scheduler, review_period_seconds)
    return []


def match_worker_discrete_review(
    state,
    now,
    worker,
    review_period_seconds: float = 60.0,
    review_scheduler=None,
    **_,
):
    """Workers enter the pool; matching waits for the next review epoch."""
    _schedule_review_if_needed(review_scheduler, review_period_seconds)
    return None


@register("discrete_review_lp")
def get_discrete_review_handlers():
    return {
        "NEW_TASK": assign_new_tasks_discrete_review,
        "FREE_WORKER": match_worker_discrete_review,
        "REVIEW_BATCH": execute_discrete_review,
    }
