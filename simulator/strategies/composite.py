from simulator.strategies import register
from math import log, fabs, cos, radians
import random
import pandas as pd

AVG_SPEED_KMH = 30

def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

def score(task, worker, λ1, λ2, λ3, now):
    distance = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    fairness = worker.fairness_ewma
    starvation = log(1 + (now - task.release_time).total_seconds())
    utility = 1.0 / (1.0 + distance)
    score_val = λ1 * fairness + λ2 * starvation + λ3 * utility
    
    return score_val

def _find_best_assignment_for_task(task, workers, now, λ1, λ2, λ3, k):
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
        s = score(task, w, λ1, λ2, λ3, now)
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

def assign_new_tasks_composite(state, now, tasks_to_assign, λ1=1.0, λ2=1.0, λ3=1.0, k=15, soft_threshold=4.0, **_):
    assignments = []
    for task in tasks_to_assign:
        best_worker, best_score = _find_best_assignment_for_task(task, state.available_workers, now, λ1, λ2, λ3, k)
        
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

def match_worker_composite(state, now, worker, λ1=1.0, λ2=1.0, λ3=1.0, k=15, soft_threshold=4.0, **_):
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

        s = score(task, worker, λ1, λ2, λ3, now)
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
