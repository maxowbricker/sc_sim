from simulator.strategies import register
from math import log
import pandas as pd
from typing import Optional, Tuple, List, Dict
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30

def _normalize_components(components: List[float]) -> List[float]:
    """Apply min-max normalization to component values.
    
    Parameters
    ----------
    components : List[float]
        List of component values to normalize
        
    Returns
    -------
    List[float]
        Normalized values in [0, 1] range
    """
    if not components:
        return []
    
    min_val = min(components)
    max_val = max(components)
    
    # Avoid division by zero
    if max_val - min_val < 1e-10:
        return [0.5] * len(components)  # All values are the same
    
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
    spatial_index
) -> Tuple[Optional[object], float]:
    """
    OPTIMIZED: Advanced Nearest Neighbor (ANN) approach from FATP paper.
    
    Instead of checking ALL workers O(|W|), we:
    1. Find k nearest workers O(k log k) 
    2. Score only those k workers O(k)
    
    This reduces complexity from O(|W|) to O(k) where k=15 << |W|=38,000.
    Massive performance improvement: 38,000 -> 15 workers checked per task!
    
    Two execution paths:
    FAST PATH: When normalize_scores=False - single-pass, no data collection.
    MIDDLE PATH: When normalize_scores=True - collects candidate data for normalization only.
    
    Parameters
    ----------
    normalize_scores : bool, default False
        If True, apply min-max normalization to F, S, U across candidates.
        
    Returns
    -------
    best_worker : Worker or None
        Best candidate worker
    best_score : float
        Best composite score achieved
    """
    if not workers:
        return None, float("-inf")

    nearest_workers = spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
    
    if not nearest_workers:
        return None, float("-inf")
    
    # OPTIMIZATION 2: Pre-calculate task drop distance (constant for all workers)
    drop_distance_const = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    # FAST PATH: When no normalization needed
    # OPTIMIZED: On-demand EWMA calculation - only for k=15 candidates, only update winner
    # gamma is now passed as parameter (from strategy_params) for consistency
    if not normalize_scores:
        
        best_worker, best_score, best_fairness_val = None, float("-inf"), None
        
        for worker in nearest_workers:
            # Check feasibility constraints: pickup before expiry, finish before worker shift ends
            d_pick = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            total_km_tmp = d_pick + drop_distance_const
            
            # Use float math instead of Timedelta (much faster)
            pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
            
            # Compare timestamps
            if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
                continue
            
            # --- ON-DEMAND FAIRNESS CALCULATION ---
            # 1. Determine reference time (when did they last finish?)
            ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
            
            # 2. Calculate continuous idle duration
            current_idle_seconds = now - ref_time
            
            # 3. Calculate POTENTIAL EWMA (without mutating state)
            #    Formula: (1-gamma) * NewIdle + gamma * OldEWMA
            potential_fairness = (1 - gamma) * current_idle_seconds + gamma * worker.fairness_ewma
            
            # 4. Score using this potential fairness
            starvation = log(1 + (now - task.release_time))
            utility = 1.0 / (1.0 + d_pick)
            
            s = fairness_weight * potential_fairness + starvation_weight * starvation + utility_weight * utility
            
            if s > best_score:
                best_score, best_worker = s, worker
                best_fairness_val = potential_fairness  # Store for winner update
        
        # UPDATE WINNER STATE (only the selected worker)
        if best_worker is not None:
            best_worker.fairness_ewma = best_fairness_val
        
        return best_worker, best_score
    
    # MIDDLE PATH: Normalization only (normalize_scores is True)
    # OPTIMIZED FOR RL: Lazy EWMA calculation, minimal state mutation
    else:
        
        candidate_data = []
        
        for worker in nearest_workers:
            # Check feasibility constraints
            d_pick = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            total_km_tmp = d_pick + drop_distance_const
            pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
            if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
                continue
            
            # LAZY EWMA: Calculate without mutating worker state
            if worker.last_active_ts is None:
                T_idle_seconds = now - worker.release_time
            else:
                T_idle_seconds = now - worker.last_active_ts
            
            fairness_raw = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
            utility_raw = 1.0 / (1.0 + d_pick)
            
            candidate_data.append({
                'worker': worker,
                'fairness_raw': fairness_raw,
                'utility_raw': utility_raw
            })
        
        if not candidate_data:
            return None, float("-inf")
        
        # Extract component values for normalization
        fairness_values = [c['fairness_raw'] for c in candidate_data]
        utility_values = [c['utility_raw'] for c in candidate_data]
        
        fairness_norm = _normalize_components(fairness_values)
        utility_norm = _normalize_components(utility_values)
        
        # Find best candidate using ranking score (fairness + utility only)
        best_worker, best_index, best_ranking_score = None, -1, float("-inf")
        for i, candidate in enumerate(candidate_data):
            ranking_score = (
                fairness_weight * fairness_norm[i] +
                utility_weight * utility_norm[i]
            )
            
            if ranking_score > best_ranking_score:
                best_ranking_score = ranking_score
                best_index = i
                best_worker = candidate['worker']
        
        # ASSIGNMENT CONFIRMED: Update only the assigned worker's EWMA
        if best_worker is not None:
            best_candidate = candidate_data[best_index]
            # Update EWMA state for assigned worker only
            best_worker.fairness_ewma = best_candidate['fairness_raw']
            
            # Calculate full score with starvation (for return value)
            starvation_raw = log(1 + (now - task.release_time))
            starvation_contribution = starvation_weight * starvation_raw
            best_score = best_ranking_score + starvation_contribution
        else:
            best_score = float("-inf")
        
        return best_worker, best_score

def _commit_assignment(task, worker, now):
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # Travel time (hours): worker→pickup, then pickup→dropoff at constant speed
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    
    # Convert hours to seconds and add to float timestamp
    task.start_time = now + (pickup_travel_hours * 3600)  # When worker arrives at pickup
    task.finish_time = task.start_time + (service_travel_hours * 3600)  # When task completes
    
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
    """Assign new tasks to available workers using composite scoring.
    
    Parameters
    ----------
    normalize_scores : bool
        If True, normalize F, S, U components before scoring (default: False)
    disable_soft_threshold : bool
        If True, bypass threshold check and assign immediately (default: False)
    """
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
        
        if disable_soft_threshold or soft_threshold == 0:
            threshold_passed = True
        else:
            threshold_passed = best_score >= soft_threshold
        
        if best_worker and threshold_passed:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_score))
        else:
            # Defer task and schedule expiry event if not already expired
            if state.defer_task(task, now):
                # Schedule expiry event if callback provided (from simulation)
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
    """Match a free worker to nearby deferred tasks using spatial index.
    Fairness is not calculated unless Assignment occurs.
    Ranking uses only starvation + utility
    
    Parameters
    ----------
    normalize_scores : bool
        If True, normalize S, U components before scoring (default: False)
    disable_soft_threshold : bool
        If True, bypass threshold check and assign immediately (default: False)
    """
    if not state.deferred_tasks:
        return None

    # Query the task index using the WORKER'S location
    nearby_tasks = state.deferred_task_index.query_k_nearest(
        worker.start_lat, 
        worker.start_lon, 
        k=k
    )
    
    if not nearby_tasks:
        return None

    # Collect candidate data
    candidate_data = []
    
    # Iterate over nearby_tasks
    # Expired tasks are removed via TASK_EXPIRE events before matching
    for task in nearby_tasks:
        drop_distance_const = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        d_pick = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const

        pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
        finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
        
        if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
            continue

        # Calculate Utility and Starvation metrics
        starvation_raw = log(1 + (now - task.release_time))
        utility_raw = 1.0 / (1.0 + d_pick)
        
        candidate_data.append({
            'task': task,
            'starvation_raw': starvation_raw,
            'utility_raw': utility_raw
        })
    
    if not candidate_data:
        return None
    
    # Apply normalization if true
    if normalize_scores:
        starvation_values = [c['starvation_raw'] for c in candidate_data]
        utility_values = [c['utility_raw'] for c in candidate_data]
        
        starvation_norm = _normalize_components(starvation_values)
        utility_norm = _normalize_components(utility_values)
        
        for i, candidate in enumerate(candidate_data):
            candidate['starvation_norm'] = starvation_norm[i]
            candidate['utility_norm'] = utility_norm[i]
            # Ranking score
            candidate['ranking_score'] = (
                starvation_weight * starvation_norm[i] +
                utility_weight * utility_norm[i]
            )
    else:
        for candidate in candidate_data:
            # Ranking score
            candidate['ranking_score'] = (
                starvation_weight * candidate['starvation_raw'] +
                utility_weight * candidate['utility_raw']
            )
            candidate['starvation_norm'] = None
            candidate['utility_norm'] = None
    
    # Find best candidate using ranking score
    best_candidate = max(candidate_data, key=lambda c: c['ranking_score'])
    best_task = best_candidate['task']

    ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
    T_idle_seconds = now - ref_time
    updated_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
    fairness_contribution = fairness_weight * updated_ewma
    best_score = best_candidate['ranking_score'] + fairness_contribution

    if disable_soft_threshold or soft_threshold == 0:
        threshold_passed = True
    else:
        threshold_passed = best_score >= soft_threshold

    if not threshold_passed:
        return None

    worker.fairness_ewma = updated_ewma

    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_score)
    
    return None

@register("composite")
def get_composite_handlers():
    return {
        "NEW_TASK": assign_new_tasks_composite,
        "FREE_WORKER": match_worker_composite
    }
