from simulator.strategies import register
import pandas as pd
from math import fabs, cos, radians
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30

def _ensure_timestamp(now):
    """Convert float timestamp to pd.Timestamp if needed."""
    if isinstance(now, (int, float)):
        return pd.Timestamp.fromtimestamp(now)
    return now

def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

def _commit_assignment(task, worker, now):
    # Ensure now is a float timestamp (timestamps are already floats from data loader)
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # OPTIMIZED: Use float math instead of Timedelta (much faster)
    # FIXED: Realistic timing - task starts after worker travels to pickup location  
    pickup_travel_seconds = (pickup_distance / AVG_SPEED_KMH) * 3600
    service_travel_seconds = (drop_distance / AVG_SPEED_KMH) * 3600
    
    task.start_time = now + pickup_travel_seconds  # When worker arrives at pickup
    task.finish_time = task.start_time + service_travel_seconds  # When task completes
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task

def assign_new_tasks_greedy(state, now, tasks_to_assign, k=100, **_):
    """
    Greedy assignment for new tasks. Assigns to nearest feasible worker.
    
    OPTIMIZED: Uses spatial index to find k nearest workers instead of checking all workers.
    This reduces complexity from O(|W|) to O(k) where k=100 << |W| (e.g., 35,449 workers).
    Even with k=100, we get the truly nearest worker in almost all cases while being 350x faster.
    
    OPTIMIZED: Uses pure float math instead of pandas Timedelta for timestamp calculations.
    This avoids object creation overhead in the hot loop (much faster).
    
    Args:
        k: Number of nearest workers to consider (default: 100). Increase for larger datasets
           or if you need absolute guarantee of finding the nearest worker.
    """
    # Ensure now is a float timestamp (timestamps are already floats from data loader)
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    assignments = []
    
    # Pre-calculate drop distance for each task (constant for all workers)
    for task in tasks_to_assign:
        best_worker, best_dist = None, float("inf")
        
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)

        # OPTIMIZATION: Use spatial index to find k nearest workers instead of all workers
        # This reduces from 35,449 workers to ~100 workers checked per task!
        nearest_workers = state.spatial_index.query_k_nearest(task.pickup_lat, task.pickup_lon, k)
        
        for worker in nearest_workers:
            pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            
            # OPTIMIZATION: Use float math instead of Timedelta (much faster)
            # Convert km to seconds: (km / kmh) * 3600 seconds/hour
            pickup_eta_seconds = (pickup_dist / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = ((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600
            
            # Compare timestamps directly (all floats)
            if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
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

def match_worker_greedy(state, now, worker, k=100, **_):
    """
    Greedy matching for a newly available worker. Finds the nearest task
    from the pool of active (unassigned) tasks.
    
    NOTE: Active tasks don't have a spatial index, so we iterate through all active tasks.
    However, active_tasks is typically much smaller than available_workers, so this is
    less of a bottleneck. If active_tasks grows large, consider adding an active_task_index
    to StateManager similar to deferred_task_index.
    
    OPTIMIZED: Uses pure float math instead of pandas Timedelta for timestamp calculations.
    
    Args:
        k: Currently unused (active tasks don't use spatial index). Kept for API consistency.
    """
    # Ensure now is a float timestamp (timestamps are already floats from data loader)
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    if not state.active_tasks:
        return None

    best_task, best_dist = None, float("inf")
    
    # NOTE: Active tasks don't have a spatial index, so we check all of them.
    # This is usually fine since active_tasks is much smaller than available_workers.
    # If this becomes a bottleneck, we could add an active_task_index to StateManager.
    for task in list(state.active_tasks): # Iterate over a copy
        pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        
        # OPTIMIZATION: Use float math instead of Timedelta (much faster)
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        pickup_eta_seconds = (pickup_dist / AVG_SPEED_KMH) * 3600
        finish_eta_seconds = ((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600
        
        # Compare timestamps directly (all floats)
        if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
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
