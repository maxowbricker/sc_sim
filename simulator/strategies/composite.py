from simulator.strategies import register
from math import log
from typing import Optional, Tuple, List
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30

def _normalize_components(components: List[float]) -> List[float]:
    """Apply min-max normalization to component values."""
    if not components:
        return []
    
    min_val = min(components)
    max_val = max(components)
    
    if max_val - min_val < 1e-10:
        return [0.5] * len(components) 
    
    return [(v - min_val) / (max_val - min_val) for v in components]


def _find_best_assignment_for_task(
    task,
    workers,
    now,
    fairness_weight,
    starvation_weight,
    utility_weight,
    k,
    normalize_scores=False,
    gamma=0.3,
    *,
    spatial_index,
) -> Tuple[Optional[object], float]:
    """OPTIMIZED: Advanced Nearest Neighbor (ANN) single-pass evaluation."""
    nearest_workers = spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
    if not nearest_workers:
        return None, float("-inf")

    # CONSTANTS
    drop_distance_const = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    starvation_raw = log(1 + (now - task.release_time))
    starvation_score = starvation_weight * starvation_raw 

    candidate_data = []
    best_worker, best_score, best_fairness_val = None, float("-inf"), None

    for worker in nearest_workers:
        d_pick = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)

        if (now + (d_pick / AVG_SPEED_KMH) * 3600) > task.expire_time or (
            now + ((d_pick + drop_distance_const) / AVG_SPEED_KMH) * 3600
        ) > worker.deadline:
            continue

        ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
        fairness_raw = (1 - gamma) * (now - ref_time) + gamma * worker.fairness_ewma
        utility_raw = 1.0 / (1.0 + d_pick)

        if normalize_scores:
            candidate_data.append((worker, fairness_raw, utility_raw))
        else:
            # FAST PATH: Avoid list memory allocation
            s = (fairness_weight * fairness_raw) + (utility_weight * utility_raw)
            if s > best_score:
                best_score, best_worker, best_fairness_val = s, worker, fairness_raw

    # NORMALIZATION PATH
    if normalize_scores and candidate_data:
        f_norm = _normalize_components([c[1] for c in candidate_data])
        u_norm = _normalize_components([c[2] for c in candidate_data])

        for i, (worker, f_raw, u_raw) in enumerate(candidate_data):
            s = (fairness_weight * f_norm[i]) + (utility_weight * u_norm[i])
            if s > best_score:
                best_score, best_worker, best_fairness_val = s, worker, f_raw

    if best_worker is not None:
        best_worker.fairness_ewma = best_fairness_val
        return best_worker, best_score + starvation_score

    return None, float("-inf")


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


def assign_new_tasks_composite(
    state,
    now,
    tasks_to_assign,
    fairness_weight=1.0,
    starvation_weight=1.0,
    utility_weight=1.0,
    k=15,
    soft_threshold=0.2,
    normalize_scores=False,
    disable_soft_threshold=False,
    gamma=0.3,
    **_,
):
    assignments = []
    for task in tasks_to_assign:
        best_worker, best_score = _find_best_assignment_for_task(
            task, 
            state.available_workers, 
            now, 
            fairness_weight, 
            starvation_weight, 
            utility_weight, 
            k,
            normalize_scores=normalize_scores,
            gamma=gamma,
            spatial_index=state.spatial_index
        )
        
        threshold_passed = True if (disable_soft_threshold or soft_threshold == 0) else (best_score >= soft_threshold)
        
        if best_worker and threshold_passed:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_score))
        else:
            if state.defer_task(task, now):
                expiry_scheduler = _.get('expiry_scheduler')
                if expiry_scheduler:
                    expiry_scheduler(task)

    return assignments


def match_worker_composite(
    state, 
    now, 
    worker, 
    fairness_weight=1.0, 
    starvation_weight=1.0, 
    utility_weight=1.0, 
    k=15, 
    soft_threshold=0.2,
    normalize_scores=False,
    disable_soft_threshold=False,
    gamma=0.3,
    **_
):
    if not state.deferred_tasks:
        return None

    nearby_tasks = state.deferred_task_index.query_k_nearest(
        worker.start_lat, 
        worker.start_lon, 
        k=k
    )
    
    if not nearby_tasks:
        return None

    candidate_data = []
    best_task, best_ranking_score = None, float("-inf")
    
    for task in nearby_tasks:
        drop_distance_const = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        d_pick = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const

        pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
        finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
        
        if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
            continue

        starvation_raw = log(1 + (now - task.release_time))
        utility_raw = 1.0 / (1.0 + d_pick)
        
        if normalize_scores:
            # Use lightweight tuples instead of heavy dicts
            candidate_data.append((task, starvation_raw, utility_raw))
        else:
            # FAST PATH: Score immediately, avoiding list memory allocation entirely
            s = (starvation_weight * starvation_raw) + (utility_weight * utility_raw)
            if s > best_ranking_score:
                best_ranking_score, best_task = s, task
    
    # NORMALIZATION PATH
    if normalize_scores and candidate_data:
        s_norm = _normalize_components([c[1] for c in candidate_data])
        u_norm = _normalize_components([c[2] for c in candidate_data])
        
        for i, (task, s_raw, u_raw) in enumerate(candidate_data):
            s = (starvation_weight * s_norm[i]) + (utility_weight * u_norm[i])
            if s > best_ranking_score:
                best_ranking_score, best_task = s, task
                
    if best_task is None:
        return None

    # Calculate final score with fairness contribution
    ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
    T_idle_seconds = now - ref_time
    updated_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
    fairness_contribution = fairness_weight * updated_ewma
    
    best_score = best_ranking_score + fairness_contribution

    threshold_passed = True if (disable_soft_threshold or soft_threshold == 0) else (best_score >= soft_threshold)

    if not threshold_passed:
        return None

    worker.fairness_ewma = updated_ewma
    assigned_task = _commit_assignment(best_task, worker, now)
    state.assign_task(assigned_task, worker)
    
    return (assigned_task, worker, best_score)


@register("composite")
def get_composite_handlers():
    return {
        "NEW_TASK": assign_new_tasks_composite,
        "FREE_WORKER": match_worker_composite
    }