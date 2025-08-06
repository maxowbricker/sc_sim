from simulator.strategies import register
import pandas as pd
from math import fabs, cos, radians

AVG_SPEED_KMH = 30

def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

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

def assign_new_tasks_greedy(state, now, tasks_to_assign, **_):
    """
    Greedy assignment for new tasks. Assigns to nearest feasible worker.
    In an event-driven model, a task that cannot be assigned is effectively deferred.
    """
    assignments = []
    for task in tasks_to_assign:
        best_worker, best_dist = None, float("inf")
        
        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)

        for worker in state.available_workers:
            pickup_dist = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            
            # Feasibility check
            finish_eta = now + pd.to_timedelta((pickup_dist + drop_dist) / AVG_SPEED_KMH, unit="h")
            if finish_eta > worker.deadline or finish_eta > task.expire_time:
                continue

            if pickup_dist < best_dist:
                best_dist = pickup_dist
                best_worker = worker
        
        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            # No feasible worker found, task remains in active_tasks to be matched later
            pass
            
    return assignments

def match_worker_greedy(state, now, worker, **_):
    """
    Greedy matching for a newly available worker. Finds the nearest task
    from the pool of active (unassigned) tasks.
    """
    if not state.active_tasks:
        return None

    best_task, best_dist = None, float("inf")
    
    for task in list(state.active_tasks): # Iterate over a copy
        pickup_dist = manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        
        # Feasibility check
        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        finish_eta = now + pd.to_timedelta((pickup_dist + drop_dist) / AVG_SPEED_KMH, unit="h")
        if finish_eta > worker.deadline or finish_eta > task.expire_time:
            continue

        if pickup_dist < best_dist:
            best_dist = pickup_dist
            best_task = task
            
    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_dist)
        
    return None

@register("greedy")
def get_greedy_handlers():
    return {
        "NEW_TASK": assign_new_tasks_greedy,
        "FREE_WORKER": match_worker_greedy
    }
