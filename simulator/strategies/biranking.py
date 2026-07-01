"""
Bipartite Ranking (BRK) baseline — KVV-style random priority matching.

Assigns a permanent uniform rank y in [0, 1) to each worker and task at first
appearance. Feasible matches select the neighbor with the lowest rank rather
than the closest spatial distance, spreading assignments across the map.

Reference: "Two-sided Online Stable Task Assignment with Incomplete Lists and
Ties in Spatial Crowdsourcing" (Algorithm 3, BRK variant).

ADAPTATION NOTE (mention in paper):
This implementation preserves the core mechanism of the original algorithm —
permanent random ranks and lowest-rank-first selection — but omits two
constructs that are incompatible with this simulator's unit-capacity,
instantaneous-matching model:

  1. Stability constraints (Definition 4 / Algorithm 3, Lines 11 & 20):
     The paper requires that a matched pair (u, v) satisfies a stability
     condition (no unmatched neighbour offers both shorter distance AND lower
     total travel weight).      Implementing this correctly requires simultaneous knowledge of all
     available neighbours at the same timestamp, which conflicts with this
     simulator's sequential, irrevocable event processing model.

  2. Multi-capacity exhaustion loop (Algorithm 3, Line 3):
     The paper wraps matching in a `while i.cap > 0` loop, allowing a single
     arriving entity to claim multiple partners per event. This simulator uses
     a strict unit-capacity model (one task per worker at a time); the concept
     of capacity > 1 does not exist in the worker or task state.

The behavioural property that matters for baseline comparison — random
persistent priority rather than distance or fairness priority — is fully
preserved. Disclose the adaptation when citing BRK results in the paper.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from simulator.spatial_index import fast_manhattan_km
from simulator.strategies import register

AVG_SPEED_KMH = 30.0


def _is_feasible(worker, task, now: float, drop_distance=None) -> Tuple[bool, float, float]:
    d_pick = fast_manhattan_km(
        worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon
    )
    # drop_distance depends only on the task; accept a precomputed value so the
    # caller can hoist it out of the per-worker loop and reuse it on commit.
    if drop_distance is None:
        drop_distance = fast_manhattan_km(
            task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon
        )

    pickup_eta = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    finish_eta = now + ((d_pick + drop_distance) / AVG_SPEED_KMH) * 3600.0
    feasible = pickup_eta <= task.expire_time and finish_eta <= worker.deadline
    return feasible, d_pick, drop_distance


def _commit_assignment(task, worker, now, d_pick: float, drop_distance: float):
    task.pickup_km = d_pick
    task.drop_km = drop_distance

    task.start_time = now + (d_pick / AVG_SPEED_KMH) * 3600.0
    task.finish_time = task.start_time + (drop_distance / AVG_SPEED_KMH) * 3600.0

    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def _entity_key(entity) -> Tuple[str, int]:
    if hasattr(entity, "pickup_lat"):
        return ("task", entity.id)
    return ("worker", entity.id)


def _get_rng(rank_tracker: Dict[str, Any], seed: int) -> random.Random:
    if "_rng" not in rank_tracker:
        rank_tracker["_rng"] = random.Random(seed)
    return rank_tracker["_rng"]


def _get_or_create_rank(entity, rank_tracker: Dict[str, Any], seed: int) -> float:
    key = _entity_key(entity)
    if key not in rank_tracker:
        rank_tracker[key] = _get_rng(rank_tracker, seed).random()
    return rank_tracker[key]


def _process_entity(
    state,
    now: float,
    entity,
    *,
    is_task: bool,
    rank_tracker: Dict[str, Any],
    seed: int,
    expiry_scheduler=None,
    deferral_tracker=None,
) -> List[Tuple[Any, Any, float]]:
    _get_or_create_rank(entity, rank_tracker, seed)
    targets = state.available_workers if is_task else state.deferred_tasks

    # When the arriving entity is the task, drop distance is constant across all
    # worker candidates — compute it once and reuse it for every feasibility test
    # and the final commit.
    entity_drop = None
    if is_task:
        entity_drop = fast_manhattan_km(
            entity.pickup_lat, entity.pickup_lon, entity.dropoff_lat, entity.dropoff_lon
        )

    best_target = None
    lowest_target_rank = float("inf")
    final_d_pick = None
    final_d_drop = None

    for target in targets:
        worker = target if is_task else entity
        task = entity if is_task else target

        feasible, d_pick, d_drop = _is_feasible(worker, task, now, entity_drop)
        if not feasible:
            continue

        target_rank = _get_or_create_rank(target, rank_tracker, seed)
        if target_rank < lowest_target_rank:
            lowest_target_rank = target_rank
            best_target = target
            final_d_pick = d_pick
            final_d_drop = d_drop

    assignments: List[Tuple[Any, Any, float]] = []
    if best_target is not None and final_d_pick is not None:
        worker = best_target if is_task else entity
        task = entity if is_task else best_target
        assigned_task = _commit_assignment(task, worker, now, final_d_pick, final_d_drop)
        state.assign_task(assigned_task, worker)
        assignments.append((assigned_task, worker, 1.0 / (1.0 + final_d_pick)))
    elif is_task:
        if state.defer_task(entity, now):
            if expiry_scheduler:
                expiry_scheduler(entity)
            if deferral_tracker:
                deferral_tracker.record_deferral(str(entity.id), now, 0.0, "no_candidates")

    return assignments


def assign_new_tasks_biranking(
    state,
    now,
    tasks_to_assign,
    rank_tracker: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    expiry_scheduler=None,
    deferral_tracker=None,
    **_,
):
    if rank_tracker is None:
        raise RuntimeError("rank_tracker must be injected by EventSimulator.reset()")

    assignments: List[Tuple[Any, Any, float]] = []
    for task in tasks_to_assign:
        assignments.extend(
            _process_entity(
                state,
                now,
                task,
                is_task=True,
                rank_tracker=rank_tracker,
                seed=seed,
                expiry_scheduler=expiry_scheduler,
                deferral_tracker=deferral_tracker,
            )
        )
    return assignments


def match_worker_biranking(
    state,
    now,
    worker,
    rank_tracker: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    **_,
):
    if rank_tracker is None:
        raise RuntimeError("rank_tracker must be injected by EventSimulator.reset()")

    out = _process_entity(
        state,
        now,
        worker,
        is_task=False,
        rank_tracker=rank_tracker,
        seed=seed,
    )
    return out if out else None


@register("biranking")
def get_biranking_handlers():
    return {
        "NEW_TASK": assign_new_tasks_biranking,
        "FREE_WORKER": match_worker_biranking,
    }
