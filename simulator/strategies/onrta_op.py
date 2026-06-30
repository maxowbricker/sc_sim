"""
ONRTA-OP (Online Non-rejection Task Assignment — Optimal) baseline.

Two-stage random-order policy: Stage 1 greedy immediate matching until the arrival
count crosses half the expected market size; Stage 2 global utility-maximizing
bipartite matching with continuous pruning (commit only if the new arrival is in
the optimal match, otherwise fall back to greedy).

Reference: "Non-Rejection Aware Online Task Assignment in Spatial Crowdsourcing"
(Algorithm 3 / ONRTA-OP built on ONRTA-Base, Sec. V-B).

Paper fidelity notes (for WISE methodology):

- Stage 1 / Stage 2 split: arrival counter increments on every task and worker
  release; Stage 2 begins when arrivals > floor((expected_a + expected_b) / 2).
  expected_a = |R| (task count); expected_b = sum of worker capacities (sum w.c).
  In this simulator workers have unit capacity (c=1), so expected_b = |W| is
  correct. If multi-capacity workers are added, set expected_b to sum(w.c), not
  |W| — otherwise Stage 2 triggers too early (see simulation.py reset() defaults).
- Stage 2 matching: Hungarian max-utility over pending tasks x available workers;
  incoming entity is assigned only if it appears in the optimal matching; otherwise
  ONRTA-Base greedy fallback (Alg. 2) — matches paper non-rejection behavior.
- Continuous pruning: state.assign_task removes workers from available_workers and
  tasks from deferred/active pools, mirroring the pruned subset R<R_delta, W<W_delta.
- Feasibility / impatience: _pair_utility enforces expire_time and worker.deadline
  at current simulation time now (spatial travel at 30 km/h Manhattan).
- Spatial instantiation: utility v = 1/(1 + d_pick); not the paper's raw distance
  formulation but consistent with other baselines in this codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0
_INFEASIBLE_UTILITY = -1e18


def _pair_utility(worker, task, now: float) -> Tuple[bool, float, float]:
    """Return (feasible, utility, d_pick)."""
    d_pick = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    d_drop = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    pickup_eta = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    finish_eta = now + ((d_pick + d_drop) / AVG_SPEED_KMH) * 3600.0
    feasible = pickup_eta <= task.expire_time and finish_eta <= worker.deadline
    utility = 1.0 / (1.0 + d_pick) if feasible else _INFEASIBLE_UTILITY
    return feasible, utility, d_pick


def _commit_assignment(task, worker, now, d_pick: float | None = None):
    if d_pick is None:
        d_pick = fast_manhattan_km(
            worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
        )
    drop_distance = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    task.pickup_km = d_pick
    task.drop_km = drop_distance

    pickup_travel_hours = d_pick / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH

    task.start_time = now + (pickup_travel_hours * 3600.0)
    task.finish_time = task.start_time + (service_travel_hours * 3600.0)

    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _pending_tasks(state) -> List[Any]:
    return list(state.deferred_tasks) + list(state.active_tasks)


def _get_greedy_match(
    entity,
    targets,
    now: float,
    *,
    is_task: bool,
) -> Tuple[Any | None, float]:
    """Highest-utility immediate match (Stage 1 / fallback)."""
    best_target = None
    best_score = float("-inf")
    best_d_pick = 0.0

    for target in targets:
        worker = target if is_task else entity
        task = entity if is_task else target
        feasible, score, d_pick = _pair_utility(worker, task, now)
        if feasible and score > best_score:
            best_score = score
            best_target = target
            best_d_pick = d_pick

    return best_target, best_d_pick


def _solve_global_optimal(tasks, workers, now: float) -> Dict[Any, Any]:
    """Stage 2 bipartite matching maximizing total utility."""
    if not tasks or not workers:
        return {}

    utility = np.full((len(tasks), len(workers)), _INFEASIBLE_UTILITY, dtype=np.float64)

    for i, task in enumerate(tasks):
        for j, worker in enumerate(workers):
            feasible, value, _ = _pair_utility(worker, task, now)
            if feasible:
                utility[i, j] = value

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-utility)

    optimal_pairs: Dict[Any, Any] = {}
    for i, j in zip(row_ind, col_ind):
        if utility[i, j] > _INFEASIBLE_UTILITY / 2:
            optimal_pairs[tasks[i]] = workers[j]
    return optimal_pairs


def _stage_two_threshold(expected_a: float, expected_b: float) -> float:
    # Paper: floor((a + b) / 2) where a = |R|, b = sum(w.c) — tune expected_b accordingly.
    return (expected_a + expected_b) / 2.0


def _process_arrival(
    state,
    now: float,
    entity,
    *,
    is_task: bool,
    expected_a: float,
    expected_b: float,
    onrta_tracker: Dict[str, int],
    expiry_scheduler=None,
    deferral_tracker=None,
) -> List[Tuple[Any, Any, float]]:
    onrta_tracker["arrivals"] += 1
    is_stage_two = onrta_tracker["arrivals"] > _stage_two_threshold(expected_a, expected_b)
    assignments: List[Tuple[Any, Any, float]] = []
    assigned = False

    if is_stage_two:
        # Alg. 3: global optimal over pruned pending sets; commit only if arrival in match.
        tasks = _pending_tasks(state)
        if is_task and entity not in tasks:
            tasks.append(entity)

        workers = list(state.available_workers)
        if not is_task and entity not in workers:
            workers.append(entity)

        optimal_pairs = _solve_global_optimal(tasks, workers, now)

        if is_task and entity in optimal_pairs:
            worker = optimal_pairs[entity]
            _, score, d_pick = _pair_utility(worker, entity, now)
            assigned_task = _commit_assignment(entity, worker, now, d_pick)
            state.assign_task(assigned_task, worker)
            assignments.append((assigned_task, worker, score))
            assigned = True
        elif not is_task and entity in optimal_pairs.values():
            task = next(t for t, w in optimal_pairs.items() if w is entity)
            _, score, d_pick = _pair_utility(entity, task, now)
            assigned_task = _commit_assignment(task, entity, now, d_pick)
            state.assign_task(assigned_task, entity)
            assignments.append((assigned_task, entity, score))
            assigned = True

    if not assigned:
        # ONRTA-Base greedy fallback (Stage 1, or Stage 2 when not in optimal matching).
        targets = state.available_workers if is_task else _pending_tasks(state)
        best_target, d_pick = _get_greedy_match(entity, targets, now, is_task=is_task)

        if best_target is not None:
            worker = best_target if is_task else entity
            task = entity if is_task else best_target
            _, score, _ = _pair_utility(worker, task, now)
            assigned_task = _commit_assignment(task, worker, now, d_pick)
            state.assign_task(assigned_task, worker)
            assignments.append((assigned_task, worker, score))
        elif is_task:
            if state.defer_task(entity, now):
                if expiry_scheduler:
                    expiry_scheduler(entity)
                if deferral_tracker:
                    deferral_tracker.record_deferral(str(entity.id), now, 0.0, "no_candidates")

    return assignments


def assign_new_tasks_onrta(
    state,
    now,
    tasks_to_assign,
    expected_a: float = 1000.0,   # |R|; reset() defaults to len(tasks) if None in config
    expected_b: float = 1000.0,   # sum(w.c); reset() defaults to len(workers) when c=1
    onrta_tracker: Optional[Dict[str, int]] = None,
    expiry_scheduler=None,
    deferral_tracker=None,
    **_,
):
    if onrta_tracker is None:
        raise RuntimeError("onrta_tracker must be injected by EventSimulator.reset()")

    assignments: List[Tuple[Any, Any, float]] = []
    for task in tasks_to_assign:
        assignments.extend(
            _process_arrival(
                state,
                now,
                task,
                is_task=True,
                expected_a=expected_a,
                expected_b=expected_b,
                onrta_tracker=onrta_tracker,
                expiry_scheduler=expiry_scheduler,
                deferral_tracker=deferral_tracker,
            )
        )
    return assignments


def match_worker_onrta(
    state,
    now,
    worker,
    expected_a: float = 1000.0,
    expected_b: float = 1000.0,
    onrta_tracker: Optional[Dict[str, int]] = None,
    **_,
):
    if onrta_tracker is None:
        raise RuntimeError("onrta_tracker must be injected by EventSimulator.reset()")

    out = _process_arrival(
        state,
        now,
        worker,
        is_task=False,
        expected_a=expected_a,
        expected_b=expected_b,
        onrta_tracker=onrta_tracker,
    )
    return out if out else None


@register("onrta_op")
def get_onrta_handlers():
    return {
        "NEW_TASK": assign_new_tasks_onrta,
        "FREE_WORKER": match_worker_onrta,
    }
