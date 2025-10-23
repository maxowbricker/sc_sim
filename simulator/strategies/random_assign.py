"""
Random assignment strategy for spatial crowdsourcing.
Randomly selects from k nearest feasible workers.

For RQ4.2: Tests the impact of purely random (but spatially-constrained) assignment
as a null hypothesis baseline.
"""

import random
import pandas as pd
from simulator.strategies import register
from math import fabs, cos, radians

AVG_SPEED_KMH = 30


def manhattan_km(lat1, lon1, lat2, lon2):
    """Calculate Manhattan distance in kilometers between two points."""
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon


def _commit_assignment(task, worker, now):
    """
    Commit a task assignment to a worker.
    Calculates timing, updates task and worker state.
    """
    pickup_distance = manhattan_km(worker.start_lat, worker.start_lon, 
                                   task.pickup_lat, task.pickup_lon)
    drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon, 
                                 task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # Calculate timing: task starts after worker travels to pickup
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    
    task.start_time = now + pd.to_timedelta(pickup_travel_hours, unit="h")
    task.finish_time = task.start_time + pd.to_timedelta(service_travel_hours, unit="h")
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def assign_new_tasks_random(state, now, tasks_to_assign, **kwargs):
    """
    Randomly assign new tasks to available workers from k=15 nearest candidates.
    
    For RQ4.2: Tests the impact of purely random (but spatially-constrained) assignment
    as a null hypothesis baseline.
    
    Core Logic:
    1. For each task, find k nearest available workers
    2. Filter for feasible workers (can meet pickup/deadline constraints)
    3. RANDOMLY select one worker from feasible set (no optimization)
    4. Defer task if no feasible worker exists
    """
    strategy_params = kwargs.get('strategy_params', {})
    k = strategy_params.get('k', 15)
    assignments = []
    
    for task in tasks_to_assign:
        feasible_workers = []
        
        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon, 
                                task.dropoff_lat, task.dropoff_lon)
        
        # Collect k nearest workers by distance
        worker_distances = []
        for worker in state.available_workers:
            pickup_dist = manhattan_km(worker.start_lat, worker.start_lon,
                                      task.pickup_lat, task.pickup_lon)
            worker_distances.append((worker, pickup_dist))
        
        # Sort by distance and take k nearest
        worker_distances.sort(key=lambda x: x[1])
        nearest_k = worker_distances[:k]
        
        # Check feasibility for k nearest workers
        for worker, pickup_dist in nearest_k:
            # Feasibility check: pickup before expiry, finish before worker shift ends
            pickup_eta = now + pd.to_timedelta(pickup_dist / AVG_SPEED_KMH, unit="h")
            finish_eta = now + pd.to_timedelta((pickup_dist + drop_dist) / AVG_SPEED_KMH, unit="h")
            
            if pickup_eta > task.expire_time or finish_eta > worker.deadline:
                continue
            
            feasible_workers.append((worker, pickup_dist, drop_dist))
        
        # RANDOM SELECTION: Pick random feasible worker (no optimization)
        if feasible_workers:
            selected_worker, pickup_dist, drop_dist = random.choice(feasible_workers)
            assigned_task = _commit_assignment(task, selected_worker, now)
            state.assign_task(assigned_task, selected_worker)
            assignments.append((assigned_task, selected_worker, pickup_dist))
        else:
            # No feasible worker found - task remains in active_tasks for later matching
            pass
    
    return assignments


def match_worker_random(state, now, worker, **kwargs):
    """
    When a worker becomes free, randomly assign from available tasks (if any nearby).
    
    Core Logic:
    1. Find k nearest tasks to the worker's current location
    2. Filter for feasible tasks
    3. RANDOMLY select one task from feasible set
    """
    strategy_params = kwargs.get('strategy_params', {})
    k = strategy_params.get('k', 15)
    
    if not state.active_tasks:
        return None
    
    feasible_tasks = []
    
    # Collect k nearest tasks by distance
    task_distances = []
    for task in list(state.active_tasks):  # Iterate over copy
        pickup_dist = manhattan_km(worker.start_lat, worker.start_lon,
                                  task.pickup_lat, task.pickup_lon)
        task_distances.append((task, pickup_dist))
    
    # Sort by distance and take k nearest
    task_distances.sort(key=lambda x: x[1])
    nearest_k = task_distances[:k]
    
    # Check feasibility for k nearest tasks
    for task, pickup_dist in nearest_k:
        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon,
                                task.dropoff_lat, task.dropoff_lon)
        pickup_eta = now + pd.to_timedelta(pickup_dist / AVG_SPEED_KMH, unit="h")
        finish_eta = now + pd.to_timedelta((pickup_dist + drop_dist) / AVG_SPEED_KMH, unit="h")
        
        if pickup_eta > task.expire_time or finish_eta > worker.deadline:
            continue
        
        feasible_tasks.append((task, pickup_dist, drop_dist))
    
    # RANDOM SELECTION: Pick random feasible task (no optimization)
    if feasible_tasks:
        task, pickup_dist, drop_dist = random.choice(feasible_tasks)
        assigned_task = _commit_assignment(task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, pickup_dist)
    
    return None


@register("random_assign")
def random_assign_strategy():
    """Factory function for random assignment strategy."""
    return {
        "NEW_TASK": assign_new_tasks_random,
        "FREE_WORKER": match_worker_random
    }
