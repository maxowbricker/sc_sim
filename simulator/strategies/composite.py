from simulator.strategies import register
from math import log, fabs, cos, radians
import pandas as pd
from typing import Optional, Tuple, List, Dict

def calculate_fairness_signal(worker, current_time, fairness_metric='ewma', all_workers=None):
    """Calculate fairness signal for a worker based on research proposal methodology."""
    if fairness_metric == 'ewma':
        # RESEARCH PROPOSAL: Implement EWMA as described
        # Fairness(w_i) = (1 - γ) · T_idle(w_i) + γ · Previous EWMA
        
        gamma = getattr(worker, 'gamma', 0.3)
        
        # Calculate current idle time in seconds
        if worker.last_active_ts is None:
            # Worker has never been active - use time since release
            T_idle_seconds = (current_time - worker.release_time).total_seconds()
        else:
            # Time since last task completion
            T_idle_seconds = (current_time - worker.last_active_ts).total_seconds()
        
        # Apply EWMA formula from research proposal
        current_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        
        # Update worker's stored EWMA for next calculation
        worker.fairness_ewma = current_ewma
        
        # Return EWMA
        return current_ewma
        
    elif fairness_metric == 'idle_time':
        # Direct idle time approach (simpler alternative)
        if worker.last_active_ts is None:
            idle_seconds = (current_time - worker.release_time).total_seconds()
        else:
            idle_seconds = (current_time - worker.last_active_ts).total_seconds()
        return idle_seconds
        
    elif fairness_metric == 'task_count':
        # Inverse of completed tasks (higher signal = fewer tasks completed)
        return 1.0 / (1.0 + worker.completed_tasks)
        
    else:
        raise ValueError(
            f"Invalid fairness_metric: '{fairness_metric}'. "
            f"Must be one of: 'ewma', 'idle_time', 'task_count'"
        )

AVG_SPEED_KMH = 30

def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

def score(task, worker, fairness_weight, starvation_weight, utility_weight, now, fairness_metric='ewma', all_workers=None):
    distance = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    fairness = calculate_fairness_signal(worker, now, fairness_metric, all_workers)
    starvation = log(1 + (now - task.release_time).total_seconds())
    utility = 1.0 / (1.0 + distance)
    score_val = fairness_weight * fairness + starvation_weight * starvation + utility_weight * utility
    
    return score_val

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
    diagnostic_tracker=None,
    *,
    spatial_index
) -> Tuple[Optional[object], float, Optional[Dict]]:
    """
    OPTIMIZED: Advanced Nearest Neighbor (ANN) approach from FATP paper.
    
    Instead of checking ALL workers O(|W|), we:
    1. Find k nearest workers O(k log k) 
    2. Score only those k workers O(k)
    
    This reduces complexity from O(|W|) to O(k) where k=15 << |W|=38,000.
    Massive performance improvement: 38,000 -> 15 workers checked per task!
    
    Performance Optimization: Three Execution Paths
    -----------------------------------------------
    FAST PATH: When normalize_scores=False and diagnostic_tracker=None
        - Uses single-pass algorithm (like Experiments 006/007)
        - Scores candidates inline, no data collection overhead
        - Typical performance: 3-4 hours for 20k tasks
    
    MIDDLE PATH: When normalize_scores=True and diagnostic_tracker=None
        - Collects candidate data for normalization only
        - Applies normalization, scores with normalized values
        - Skips diagnostic info preparation (reduces overhead)
        - Performance: ~1.5-2x slower than fast path
    
    SLOW PATH: When diagnostic_tracker is provided
        - Collects all candidate data for normalization and/or diagnostics
        - Prepares diagnostic_info for tracking
        - Multiple passes through candidate list
        - Performance: 2-3x slower than fast path, enables advanced features
    
    Parameters
    ----------
    normalize_scores : bool, default False
        If True, apply min-max normalization to F, S, U across candidates.
        Forces SLOW PATH. Use to test if mis-scaled components cause issues.
    diagnostic_tracker : DiagnosticTracker, optional
        If provided, record score component details for analysis.
        Forces SLOW PATH. Enable via config: enable_diagnostics=True.
        
    Returns
    -------
    best_worker : Worker or None
        Best candidate worker
    best_score : float
        Best composite score achieved
    diagnostic_info : dict or None
        Component values for diagnostic tracking (None for fast path)
    """
    if not workers:
        return None, float("-inf"), None

    # OPTIMIZATION 1: Use Spatial Index for efficient nearest neighbor search
    # This reduces from 38,000 workers to 15 workers checked per task!
    nearest_workers = spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
    
    if not nearest_workers:
        return None, float("-inf"), None
    
    # OPTIMIZATION 2: Pre-calculate task drop distance (constant for all workers)
    drop_distance_const = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    # FAST PATH: When no normalization or diagnostics needed
    # Uses original single-pass algorithm for maximum performance
    if not normalize_scores and diagnostic_tracker is None:
        # OPTIMIZED: Pre-calculate timestamps once (avoid Pandas Timedelta in loop)
        now_ts = now.timestamp()
        task_expire_ts = task.expire_time.timestamp()
        
        best_worker, best_score = None, float("-inf")
        
        for worker in nearest_workers:
            # Check feasibility constraints: pickup before expiry, finish before worker shift ends
            d_pick = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            total_km_tmp = d_pick + drop_distance_const
            
            # Use float math instead of Timedelta (much faster)
            pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
            
            # Compare timestamps (floats) - avoids Pandas object creation
            if (now_ts + pickup_eta_seconds) > task_expire_ts or (now_ts + finish_eta_seconds) > worker.deadline.timestamp():
                continue
            
            # Score inline using original score() function
            s = score(task, worker, fairness_weight, starvation_weight, utility_weight, now)
            
            if s > best_score:
                best_score, best_worker = s, worker
        
        return best_worker, best_score, None
    
    # MIDDLE PATH: Normalization only (no diagnostics)
    # OPTIMIZED FOR RL: Lazy EWMA calculation, minimal state mutation
    # Collects candidate data for normalization, but skips diagnostic info preparation
    if normalize_scores and diagnostic_tracker is None:
        # Pre-calculate gamma once (assumes all workers have same gamma)
        gamma = getattr(nearest_workers[0], 'gamma', 0.3) if nearest_workers else 0.3
        
        now_ts = now.timestamp()
        task_expire_ts = task.expire_time.timestamp()
        
        candidate_data = []
        
        for worker in nearest_workers:
            # Check feasibility constraints
            d_pick = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            total_km_tmp = d_pick + drop_distance_const
            pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
            if (now_ts + pickup_eta_seconds) > task_expire_ts or (now_ts + finish_eta_seconds) > worker.deadline.timestamp():
                continue
            
            # LAZY EWMA: Calculate without mutating worker state
            if worker.last_active_ts is None:
                T_idle_seconds = (now - worker.release_time).total_seconds()
            else:
                T_idle_seconds = (now - worker.last_active_ts).total_seconds()
            
            fairness_raw = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
            utility_raw = 1.0 / (1.0 + d_pick)
            
            candidate_data.append({
                'worker': worker,
                'fairness_raw': fairness_raw,
                'utility_raw': utility_raw
            })
        
        if not candidate_data:
            return None, float("-inf"), None
        
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
            starvation_raw = log(1 + (now - task.release_time).total_seconds())
            starvation_contribution = starvation_weight * starvation_raw
            best_score = best_ranking_score + starvation_contribution
        else:
            best_score = float("-inf")
        
        return best_worker, best_score, None
    
    # SLOW PATH: Diagnostics enabled (with or without normalization)
    # Collects all candidate data for normalization and/or diagnostic tracking
    now_ts = now.timestamp()
    task_expire_ts = task.expire_time.timestamp()
    
    candidate_data = []
    
    for worker in nearest_workers:
        # Check feasibility constraints: pickup before expiry, finish before worker shift ends
        d_pick = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const
        
        pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
        finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
        
        # Compare timestamps (floats) - avoids Pandas object creation
        if (now_ts + pickup_eta_seconds) > task_expire_ts or (now_ts + finish_eta_seconds) > worker.deadline.timestamp():
            continue
        
        # Calculate raw component values
        distance = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        fairness_raw = calculate_fairness_signal(worker, now)
        starvation_raw = log(1 + (now - task.release_time).total_seconds())
        utility_raw = 1.0 / (1.0 + distance)
        
        candidate_data.append({
            'worker': worker,
            'fairness_raw': fairness_raw,
            'starvation_raw': starvation_raw,
            'utility_raw': utility_raw
        })
    
    if not candidate_data:
        return None, float("-inf"), None
    
    # Apply normalization if requested
    if normalize_scores:
        # Extract component values
        fairness_values = [c['fairness_raw'] for c in candidate_data]
        starvation_values = [c['starvation_raw'] for c in candidate_data]
        utility_values = [c['utility_raw'] for c in candidate_data]
        
        # Normalize each component across candidates
        fairness_norm = _normalize_components(fairness_values)
        starvation_norm = _normalize_components(starvation_values)
        utility_norm = _normalize_components(utility_values)
        
        # Add normalized values to candidate data
        for i, candidate in enumerate(candidate_data):
            candidate['fairness_norm'] = fairness_norm[i]
            candidate['starvation_norm'] = starvation_norm[i]
            candidate['utility_norm'] = utility_norm[i]
            
            # Calculate score with normalized values
            candidate['score'] = (
                fairness_weight * fairness_norm[i] +
                starvation_weight * starvation_norm[i] +
                utility_weight * utility_norm[i]
            )
    else:
        # Calculate scores with raw values (original behavior)
        for candidate in candidate_data:
            candidate['score'] = (
                fairness_weight * candidate['fairness_raw'] +
                starvation_weight * candidate['starvation_raw'] +
                utility_weight * candidate['utility_raw']
            )
            candidate['fairness_norm'] = None
            candidate['starvation_norm'] = None
            candidate['utility_norm'] = None
    
    # Find best candidate
    best_candidate = max(candidate_data, key=lambda c: c['score'])
    best_worker = best_candidate['worker']
    best_score = best_candidate['score']
    
    # Prepare diagnostic info for the best candidate
    diagnostic_info = {
        'fairness_raw': best_candidate['fairness_raw'],
        'starvation_raw': best_candidate['starvation_raw'],
        'utility_raw': best_candidate['utility_raw'],
        'fairness_norm': best_candidate.get('fairness_norm'),
        'starvation_norm': best_candidate.get('starvation_norm'),
        'utility_norm': best_candidate.get('utility_norm'),
    }
            
    return best_worker, best_score, diagnostic_info

def _commit_assignment(task, worker, now):
    pickup_distance = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # FIXED: Realistic timing - task starts after worker travels to pickup location
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    
    task.start_time = now + pd.to_timedelta(pickup_travel_hours, unit="h")  # When worker arrives at pickup
    task.finish_time = task.start_time + pd.to_timedelta(service_travel_hours, unit="h")  # When task completes
    
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
    fairness_metric='ewma',
    normalize_scores=False,
    disable_soft_threshold=False,
    diagnostic_tracker=None,
    **_
):
    """Assign new tasks to available workers using composite scoring.
    
    EXPERIMENT 008: Enhanced with score normalization and threshold ablation.
    
    Parameters
    ----------
    normalize_scores : bool
        If True, normalize F, S, U components before scoring (default: False)
    disable_soft_threshold : bool
        If True, bypass threshold check and assign immediately (default: False)
    diagnostic_tracker : DiagnosticTracker, optional
        If provided, record assignment diagnostics for analysis
    """
    assignments = []
    for task in tasks_to_assign:
        best_worker, best_score, diagnostic_info = _find_best_assignment_for_task(
            task, 
            state.available_workers, 
            now, 
            fairness_weight, 
            starvation_weight, 
            utility_weight, 
            k,
            normalize_scores=normalize_scores,
            diagnostic_tracker=diagnostic_tracker,
            spatial_index=state.spatial_index  # Spatial index is always initialized in StateManager
        )
        
        # OPTIMIZATION: Skip threshold check if disabled or threshold is 0
        # This avoids unnecessary comparison when threshold check always passes
        if disable_soft_threshold or soft_threshold == 0:
            threshold_passed = True
        else:
            threshold_passed = best_score >= soft_threshold
        
        if best_worker and threshold_passed:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_score))
            
            # EXPERIMENT 008: Record assignment to diagnostic tracker
            if diagnostic_tracker and diagnostic_info:
                diagnostic_tracker.record_assignment(
                    task_id=task.id,
                    worker_id=best_worker.id,
                    fairness_raw=diagnostic_info['fairness_raw'],
                    starvation_raw=diagnostic_info['starvation_raw'],
                    utility_raw=diagnostic_info['utility_raw'],
                    fairness_norm=diagnostic_info.get('fairness_norm'),
                    starvation_norm=diagnostic_info.get('starvation_norm'),
                    utility_norm=diagnostic_info.get('utility_norm'),
                    fairness_weight=fairness_weight,
                    starvation_weight=starvation_weight,
                    utility_weight=utility_weight,
                    final_score=best_score,
                    was_deferred_before=False,
                    timestamp=now
                )
            
            # EXPERIMENT 019: Record assignment to deferral tracker (RQ3.3)
            deferral_tracker = _.get('deferral_tracker') if _ else None
            if deferral_tracker:
                deferral_tracker.record_assignment(
                    task_id=task.id,
                    timestamp=now,
                    was_deferred=(task.deferral_count > 0),
                    deferral_count=task.deferral_count
                )
        else:
            # Defer task and schedule expiry event if not already expired
            if state.defer_task(task, now):
                # Schedule expiry event if callback provided (from simulation)
                expiry_scheduler = _.get('expiry_scheduler')
                if expiry_scheduler:
                    expiry_scheduler(task)
            
            # EXPERIMENT 008: Record deferral to diagnostic tracker
            if diagnostic_tracker:
                reason = "no_candidates" if not best_worker else "below_threshold"
                diagnostic_tracker.record_task_deferred(
                    task_id=task.id,
                    best_score=best_score,
                    threshold=soft_threshold,
                    reason=reason,
                    timestamp=now,
                    best_worker_id=best_worker.id if best_worker else None
                )
            
            # EXPERIMENT 019: Record deferral to deferral tracker (RQ3.3)
            deferral_tracker = _.get('deferral_tracker') if _ else None
            if deferral_tracker:
                reason = "no_candidates" if not best_worker else "below_threshold"
                deferral_tracker.record_deferral(
                    task_id=task.id,
                    timestamp=now,
                    score=best_score,
                    reason=reason
                )
            
            # Monitor deferred task behavior (if monitoring enabled)
            if hasattr(state, 'deferred_monitor') and state.deferred_monitor:
                state.deferred_monitor.record_task_deferred(task.id, now)
            
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
    diagnostic_tracker=None,
    **_
):
    """Match a free worker to nearby deferred tasks using spatial index.
    
    OPTIMIZED: Uses spatial index to find only k nearest deferred tasks instead of
    checking all deferred tasks. This reduces complexity from O(|D|) to O(k) where
    k=15 << |D| (number of deferred tasks).
    
    Fairness is not calculated unless Assignment occurs.
    Ranking uses only starvation + utility (fairness is constant as the worker stays the same).
    
    Parameters
    ----------
    normalize_scores : bool
        If True, normalize S, U components before scoring (default: False)
    disable_soft_threshold : bool
        If True, bypass threshold check and assign immediately (default: False)
    diagnostic_tracker : DiagnosticTracker, optional
        If provided, record assignment diagnostics for analysis
    """
    if not state.deferred_tasks:
        return None

    # Monitor computational impact (if monitoring enabled)
    deferred_count = len(state.deferred_tasks)
    if hasattr(state, 'deferred_monitor') and state.deferred_monitor:
        state.deferred_monitor.record_deferred_iteration(deferred_count)

    # OPTIMIZATION: Use spatial index to find nearby deferred tasks
    # instead of iterating through ALL deferred tasks.
    # Query the task index using the WORKER'S location
    nearby_tasks = state.deferred_task_index.query_k_nearest(
        worker.start_lat, 
        worker.start_lon, 
        k=k
    )
    
    if not nearby_tasks:
        return None

    # Collect candidate data
    now_ts = now.timestamp()
    worker_deadline_ts = worker.deadline.timestamp()
    
    candidate_data = []
    
    # Iterate over nearby_tasks instead of state.deferred_tasks
    # Expired tasks are removed via TASK_EXPIRE events before matching
    for task in nearby_tasks:
        drop_distance_const = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        d_pick = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const

        pickup_eta_seconds = (d_pick / AVG_SPEED_KMH) * 3600
        finish_eta_seconds = (total_km_tmp / AVG_SPEED_KMH) * 3600
        
        if (now_ts + pickup_eta_seconds) > task.expire_time.timestamp() or (now_ts + finish_eta_seconds) > worker_deadline_ts:
            continue

        # Calculate Utility and Starvation metrics
        starvation_raw = log(1 + (now - task.release_time).total_seconds())
        utility_raw = 1.0 / (1.0 + d_pick)
        
        candidate_data.append({
            'task': task,
            'starvation_raw': starvation_raw,
            'utility_raw': utility_raw
        })
    
    if not candidate_data:
        return None
    
    # Apply normalization if requested
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
    
    # OPTIMIZATION: Skip threshold check if disabled or threshold is 0
    # This avoids unnecessary EWMA calculation when threshold check always passes
    if disable_soft_threshold or soft_threshold == 0:
        # Threshold check always passes - assign immediately
        # Calculate EWMA only when assignment is confirmed (lazy evaluation)
        gamma = getattr(worker, 'gamma', 0.3)
        if worker.last_active_ts is None:
            T_idle_seconds = (now - worker.release_time).total_seconds()
        else:
            T_idle_seconds = (now - worker.last_active_ts).total_seconds()
        
        updated_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        fairness_contribution = fairness_weight * updated_ewma
        
        # Calculate full score for return value and diagnostics
        best_score = best_candidate['ranking_score'] + fairness_contribution
        
        # ASSIGNMENT CONFIRMED: Update worker's EWMA state
        worker.fairness_ewma = updated_ewma
    else:
        # Threshold check required - calculate EWMA for threshold evaluation
        gamma = getattr(worker, 'gamma', 0.3)
        if worker.last_active_ts is None:
            T_idle_seconds = (now - worker.release_time).total_seconds()
        else:
            T_idle_seconds = (now - worker.last_active_ts).total_seconds()
        
        updated_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        fairness_contribution = fairness_weight * updated_ewma
        
        # Calculate full score with updated fairness for threshold check
        best_score = best_candidate['ranking_score'] + fairness_contribution
        
        # Check threshold
        threshold_passed = best_score >= soft_threshold
        
        if not threshold_passed:
            # Threshold not met: do nothing, worker keeps current EWMA, task stays deferred
            return None
        
        # ASSIGNMENT CONFIRMED: Update worker's EWMA state
        worker.fairness_ewma = updated_ewma
            
    if best_task:
        
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        
        # EXPERIMENT 008: Record assignment to diagnostic tracker
        if diagnostic_tracker:
            diagnostic_tracker.record_assignment(
                task_id=best_task.id,
                worker_id=worker.id,
                fairness_raw=updated_ewma,
                starvation_raw=best_candidate['starvation_raw'],
                utility_raw=best_candidate['utility_raw'],
                fairness_norm=None,
                starvation_norm=best_candidate.get('starvation_norm'),
                utility_norm=best_candidate.get('utility_norm'),
                fairness_weight=fairness_weight,
                starvation_weight=starvation_weight,
                utility_weight=utility_weight,
                final_score=best_score,
                was_deferred_before=True,
                timestamp=now
            )
        
        # EXPERIMENT 019: Record assignment to deferral tracker (RQ3.3)
        # Note: Need to get deferral_tracker from strategy_params (passed via _)
        deferral_tracker = _.get('deferral_tracker') if _ else None
        if deferral_tracker:
            deferral_tracker.record_assignment(
                task_id=best_task.id,
                timestamp=now,
                was_deferred=True,  # Always true in match_worker path
                deferral_count=best_task.deferral_count
            )
        
        # Monitor successful assignment from deferred state (if monitoring enabled)
        if hasattr(state, 'deferred_monitor') and state.deferred_monitor:
            state.deferred_monitor.record_task_assigned_from_deferred(best_task.id, now)
            
        return (assigned_task, worker, best_score)
    
    return None

@register("composite")
def get_composite_handlers():
    return {
        "NEW_TASK": assign_new_tasks_composite,
        "FREE_WORKER": match_worker_composite
    }
