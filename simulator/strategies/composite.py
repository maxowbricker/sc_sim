from simulator.strategies import register
from math import log, fabs, cos, radians
import random
import pandas as pd

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
        
        # Convert to hours for more reasonable scale and add small differentiation
        # to prevent identical values when all workers have similar idle times
        import random
        random.seed(hash(worker.id) % 2**31)  # Deterministic per worker
        worker_bias = random.uniform(0.95, 1.05)  # Small 5% variation per worker
        
        return (current_ewma / 3600.0) * worker_bias
        
    elif fairness_metric == 'idle_time':
        # Direct idle time approach (simpler alternative)
        if worker.last_active_ts is None:
            idle_seconds = (current_time - worker.release_time).total_seconds()
        else:
            idle_seconds = (current_time - worker.last_active_ts).total_seconds()
        return idle_seconds / 3600.0  # Convert to hours
        
    elif fairness_metric == 'task_count':
        # Inverse of completed tasks (higher signal = fewer tasks completed)
        return 1.0 / (1.0 + worker.completed_tasks)
        
    else:
        # Default to EWMA as per research proposal
        gamma = getattr(worker, 'gamma', 0.3)
        
        if worker.last_active_ts is None:
            T_idle_seconds = (current_time - worker.release_time).total_seconds()
        else:
            T_idle_seconds = (current_time - worker.last_active_ts).total_seconds()
        
        current_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        worker.fairness_ewma = current_ewma
        
        # Add small differentiation
        import random
        random.seed(hash(worker.id) % 2**31)
        worker_bias = random.uniform(0.95, 1.05)
        
        return (current_ewma / 3600.0) * worker_bias

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

def _find_best_assignment_for_task(task, workers, now, fairness_weight, starvation_weight, utility_weight, k):
    if not workers:
        return None, float("-inf")

    drop_distance_const = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    candidates = []
    for w in workers:
        d_pick = manhattan_km(w.start_lat, w.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const
        finish_eta = now + pd.to_timedelta(total_km_tmp / AVG_SPEED_KMH, unit="h")
        if finish_eta > w.deadline or finish_eta > task.expire_time:
            continue
        candidates.append((d_pick, w))

    candidates.sort(key=lambda tup: tup[0])
    if k > 0:
        candidates = candidates[:k]

    if not candidates:
        return None, float("-inf")

    best_worker, best_score = None, float("-inf")
    for dist, w in candidates:
        s = score(task, w, fairness_weight, starvation_weight, utility_weight, now)
        if s > best_score:
            best_score, best_worker = s, w
            
    return best_worker, best_score

def _commit_assignment(task, worker, now):
    pickup_distance = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    hours = (pickup_distance + drop_distance) / AVG_SPEED_KMH
    task.finish_time = now + pd.to_timedelta(hours, unit="h")
    task.start_time = now
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task

def assign_new_tasks_composite(state, now, tasks_to_assign, fairness_weight=1.0, starvation_weight=1.0, utility_weight=1.0, k=15, soft_threshold=0.5, fairness_metric='ewma', **_):
    assignments = []
    for task in tasks_to_assign:
        best_worker, best_score = _find_best_assignment_for_task(task, state.available_workers, now, fairness_weight, starvation_weight, utility_weight, k)
        
        if best_worker and best_score >= soft_threshold:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_score))
            # Log assignment for analysis
            if hasattr(state, 'assignment_log'):
                state.assignment_log.append({
                    'timestamp': now,
                    'task_id': task.id,
                    'worker_id': best_worker.id,
                    'score': best_score,
                    'fairness': best_worker.fairness_ewma,
                    'starvation': log(1 + (now - task.release_time).total_seconds()),
                    'utility': 1.0 / (1.0 + manhattan_km(best_worker.start_lat, best_worker.start_lon, task.pickup_lat, task.pickup_lon)),
                    'decision': 'assigned'
                })
        else:
            state.defer_task(task)
            # Log deferral for analysis
            if hasattr(state, 'assignment_log'):
                state.assignment_log.append({
                    'timestamp': now,
                    'task_id': task.id,
                    'worker_id': best_worker.id if best_worker else None,
                    'score': best_score,
                    'decision': 'deferred'
                })
            
    return assignments

def match_worker_composite(state, now, worker, fairness_weight=1.0, starvation_weight=1.0, utility_weight=1.0, k=15, soft_threshold=0.5, **_):
    if not state.deferred_tasks:
        return None

    best_task, best_score = None, float("-inf")
    for task in list(state.deferred_tasks):
        drop_distance_const = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        d_pick = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        total_km_tmp = d_pick + drop_distance_const
        finish_eta = now + pd.to_timedelta(total_km_tmp / AVG_SPEED_KMH, unit="h")
        if finish_eta > worker.deadline or finish_eta > task.expire_time:
            continue

        s = score(task, worker, fairness_weight, starvation_weight, utility_weight, now)
        if s > best_score:
            best_score, best_task = s, task
            
    if best_task and best_score >= soft_threshold:
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
