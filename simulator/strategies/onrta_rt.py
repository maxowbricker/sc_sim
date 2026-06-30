"""
ONRTA-RT (Online Non-rejection Task Assignment — Randomized Threshold) baseline.

Draws a random utility threshold theta once per episode from {e^0, ..., e^(lambda-1)}
where lambda = ceil(ln(U_max + 1)). Spatial scores are scaled by UTILITY_SCALE so
fractional 1/(1+d_pick) utilities produce a meaningful logarithmic threshold set.

Reference: "Non-Rejection Aware Online Task Assignment in Spatial Crowdsourcing"
(Algorithm 1 / ONRTA-RT, Sec. IV).

Paper fidelity notes (for WISE methodology):

- Threshold generation (Alg. 1 L1–2): lambda = ceil(ln(U_max + 1)); theta drawn
  uniformly from {e^0, ..., e^(lambda-1)} once per episode. UTILITY_SCALE=100 sets
  U_max for spatial utility 100/(1+d_pick); without scaling, U_max~1 degenerates
  lambda to 1 and collapses the randomized policy (see _ensure_theta).
- Candidate pool (Alg. 1 L4–6): uniform random choice among all feasible pairs
  with scaled utility >= theta — exact match to paper Cand sampling.
- Non-rejection fallback (Alg. 1 L8–10): if Cand empty, assign highest-utility
  feasible pair containing the arriving entity (best_score_fallback loop).
- Returned score vs threshold: threshold compares UTILITY_SCALE/(1+d_pick); returned
  assignment score is unscaled 1/(1+d_pick) for cross-baseline metric comparability.
- Expiration pruning (Alg. 1 L11): tasks removed from deferred pool via TASK_EXPIRE
  events (simulation.py); infeasible pairs skipped via _is_feasible at current now.
  Workers past deadline are not explicitly removed from available_workers but cannot
  match (finish_eta > worker.deadline). Same pattern as other strategies.
- Worker-side pool: FREE_WORKER scans state.deferred_tasks only; unmatched tasks
  are deferred on arrival so this matches ONRTA-RT in practice (onrta_op also
  includes active_tasks for safety).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0
# U_max for Alg. 1 threshold set; spatial 1/(1+d) ~ 1 would collapse lambda to 1.
UTILITY_SCALE = 100.0


def _is_feasible(worker, task, now: float) -> Tuple[bool, float]:
    d_pick = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    drop_distance = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    pickup_eta = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    finish_eta = now + ((d_pick + drop_distance) / AVG_SPEED_KMH) * 3600.0
    feasible = pickup_eta <= task.expire_time and finish_eta <= worker.deadline
    return feasible, d_pick


def _commit_assignment(task, worker, now, d_pick: float):
    drop_distance = fast_manhattan_km(
        task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
    )

    task.pickup_km = d_pick
    task.drop_km = drop_distance

    task.start_time = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    task.finish_time = task.start_time + (drop_distance / AVG_SPEED_KMH) * 3600.0

    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _get_rng(onrta_rt_state: Dict[str, Any], seed: int) -> random.Random:
    if "_rng" not in onrta_rt_state:
        onrta_rt_state["_rng"] = random.Random(seed)
    return onrta_rt_state["_rng"]


def _ensure_theta(onrta_rt_state: Dict[str, Any], seed: int) -> float:
    """Alg. 1 L1–2: draw theta in {e^0, ..., e^(lambda-1)} once per episode."""
    if "theta" in onrta_rt_state:
        return onrta_rt_state["theta"]

    rng = _get_rng(onrta_rt_state, seed)
    lam = max(1, math.ceil(math.log(UTILITY_SCALE + 1)))
    exponent = rng.choice(range(lam))
    onrta_rt_state["theta"] = math.exp(exponent)
    return onrta_rt_state["theta"]


def _process_entity(
    state,
    now: float,
    entity,
    *,
    is_task: bool,
    onrta_rt_state: Dict[str, Any],
    seed: int,
    expiry_scheduler=None,
    deferral_tracker=None,
) -> List[Tuple[Any, Any, float]]:
    theta = _ensure_theta(onrta_rt_state, seed)
    targets = state.available_workers if is_task else state.deferred_tasks

    candidates_above_theta: List[Tuple[Any, float, float]] = []
    best_target_fallback = None
    best_score_fallback = float("-inf")
    best_dpick_fallback = None

    for target in targets:
        worker = target if is_task else entity
        task = entity if is_task else target

        feasible, d_pick = _is_feasible(worker, task, now)
        if not feasible:
            continue

        score = UTILITY_SCALE / (1.0 + d_pick)
        if score >= theta:
            candidates_above_theta.append((target, score, d_pick))
        if score > best_score_fallback:
            best_score_fallback = score
            best_target_fallback = target
            best_dpick_fallback = d_pick

    assigned_target = None
    final_d_pick = None

    if candidates_above_theta:
        # Alg. 1 L4–6: uniform random choice from Cand (utility >= theta).
        assigned_target, _, final_d_pick = _get_rng(onrta_rt_state, seed).choice(
            candidates_above_theta
        )
    elif best_target_fallback is not None:
        # Alg. 1 L8–10: non-rejection greedy fallback when Cand is empty.
        assigned_target = best_target_fallback
        final_d_pick = best_dpick_fallback

    assignments: List[Tuple[Any, Any, float]] = []
    if assigned_target is not None and final_d_pick is not None:
        worker = assigned_target if is_task else entity
        task = entity if is_task else assigned_target
        assigned_task = _commit_assignment(task, worker, now, final_d_pick)
        state.assign_task(assigned_task, worker)
        # Unscaled utility for cross-baseline metrics (threshold uses scaled score).
        assignments.append((assigned_task, worker, 1.0 / (1.0 + final_d_pick)))
    elif is_task:
        if state.defer_task(entity, now):
            if expiry_scheduler:
                expiry_scheduler(entity)
            if deferral_tracker:
                deferral_tracker.record_deferral(str(entity.id), now, 0.0, "no_candidates")

    return assignments


def assign_new_tasks_onrta_rt(
    state,
    now,
    tasks_to_assign,
    onrta_rt_state: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    expiry_scheduler=None,
    deferral_tracker=None,
    **_,
):
    if onrta_rt_state is None:
        raise RuntimeError("onrta_rt_state must be injected by EventSimulator.reset()")

    assignments: List[Tuple[Any, Any, float]] = []
    for task in tasks_to_assign:
        assignments.extend(
            _process_entity(
                state,
                now,
                task,
                is_task=True,
                onrta_rt_state=onrta_rt_state,
                seed=seed,
                expiry_scheduler=expiry_scheduler,
                deferral_tracker=deferral_tracker,
            )
        )
    return assignments


def match_worker_onrta_rt(
    state,
    now,
    worker,
    onrta_rt_state: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    **_,
):
    if onrta_rt_state is None:
        raise RuntimeError("onrta_rt_state must be injected by EventSimulator.reset()")

    out = _process_entity(
        state,
        now,
        worker,
        is_task=False,
        onrta_rt_state=onrta_rt_state,
        seed=seed,
    )
    return out if out else None


@register("onrta_rt")
def get_onrta_rt_handlers():
    return {
        "NEW_TASK": assign_new_tasks_onrta_rt,
        "FREE_WORKER": match_worker_onrta_rt,
    }
