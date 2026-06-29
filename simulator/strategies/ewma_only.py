"""
EWMA-Only Strategy

Advanced fairness baseline that assigns tasks to workers with the highest EWMA
fairness signal (most under-served). Uses the same sophisticated EWMA metric
as the composite strategy but without utility or starvation components.

Research Context:
- Represents "Advanced Fairness-Only" baseline for comparison with LAF and composite
- Uses time-weighted EWMA metric: (1-γ)×T_idle + γ×Previous_EWMA
- No spatial optimization (assigns to most under-served worker by EWMA)
- Demonstrates that sophisticated fairness metric alone is insufficient
"""

from simulator.strategies import register
import pandas as pd
import random
from math import fabs, cos, radians

AVG_SPEED_KMH = 30


def manhattan_km(lat1, lon1, lat2, lon2):
    """Calculate Manhattan distance in kilometers between two points."""
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon


def calculate_fairness_signal(worker, current_time, fairness_metric='ewma', gamma=0.3):
    """
    Calculate fairness signal for a worker based on EWMA methodology.

    Copied from composite.py to ensure identical metric calculation.
    Higher score = more under-served = higher priority for assignment.

    Args:
        worker: Worker object
        current_time: Current simulation timestamp
        fairness_metric: Type of fairness metric ('ewma' is default)
        gamma: EWMA smoothing factor (from strategy_params)

    Returns:
        Float representing fairness signal (higher = more under-served)
    """
    if fairness_metric == 'ewma':
        # EWMA Formula: Fairness(w_i) = (1 - γ) · T_idle(w_i) + γ · Previous EWMA
        # Timestamps are plain floats (seconds since epoch) — arithmetic gives seconds directly.
        ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
        T_idle_seconds = current_time - ref_time

        current_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        worker.fairness_ewma = current_ewma

        random.seed(hash(worker.id) % 2**31)
        worker_bias = random.uniform(0.95, 1.05)
        return (current_ewma / 3600.0) * worker_bias

    elif fairness_metric == 'idle_time':
        ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
        return (current_time - ref_time) / 3600.0

    elif fairness_metric == 'task_count':
        return 1.0 / (1.0 + worker.completed_tasks)

    else:
        # Default to EWMA
        ref_time = worker.last_active_ts if worker.last_active_ts is not None else worker.release_time
        T_idle_seconds = current_time - ref_time
        current_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
        worker.fairness_ewma = current_ewma
        random.seed(hash(worker.id) % 2**31)
        worker_bias = random.uniform(0.95, 1.05)
        return (current_ewma / 3600.0) * worker_bias


def _commit_assignment(task, worker, now):
    """
    Commit a task assignment to a worker.
    Calculates timing, updates task and worker state.
    
    Args:
        task: Task object to assign
        worker: Worker object to assign to
        now: Current simulation timestamp
    
    Returns:
        The assigned task object
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
    
    task.start_time = now + (pickup_travel_hours * 3600)
    task.finish_time = task.start_time + (service_travel_hours * 3600)
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def assign_new_tasks_ewma_only(state, now, tasks_to_assign, gamma=0.3, **_):
    """
    EWMA-Only assignment for new tasks arriving in the system.
    
    Core Logic:
    1. For each task, find all feasible workers (can meet pickup/deadline constraints)
    2. Calculate EWMA fairness signal for each feasible worker
    3. From feasible set, select worker with MAXIMUM fairness signal (most under-served)
    4. Use distance as tie-breaker if multiple workers have similar EWMA scores
    5. Defer task if no feasible worker exists
    
    This enforces fairness using sophisticated time-weighted metric.
    
    Args:
        state: Current simulation state
        now: Current timestamp
        tasks_to_assign: List of tasks to assign
    
    Returns:
        List of (task, worker, distance) tuples for successful assignments
    """
    assignments = []
    
    for task in tasks_to_assign:
        best_worker = None
        best_fairness_signal = float("-inf")
        best_dist = float("inf")  # Tie-breaker
        
        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon, 
                                task.dropoff_lat, task.dropoff_lon)
        
        for worker in state.available_workers:
            pickup_dist = manhattan_km(worker.start_lat, worker.start_lon,
                                      task.pickup_lat, task.pickup_lon)
            
            # Feasibility check: pickup before expiry, finish before worker shift ends
            pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
            finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)
            
            if pickup_eta > task.expire_time or finish_eta > worker.deadline:
                continue
            
            # EWMA-ONLY CORE LOGIC: Calculate fairness signal for each worker
            fairness_signal = calculate_fairness_signal(worker, now, 'ewma', gamma=gamma)
            
            if fairness_signal > best_fairness_signal:
                # Found worker with higher fairness signal (more under-served)
                best_fairness_signal = fairness_signal
                best_worker = worker
                best_dist = pickup_dist
            elif abs(fairness_signal - best_fairness_signal) < 0.001 and pickup_dist < best_dist:
                # Tie on fairness signal - use distance as tie-breaker
                best_worker = worker
                best_dist = pickup_dist
        
        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            # No feasible worker found - task remains in active_tasks for later matching
            pass
    
    return assignments


def match_worker_ewma_only(state, now, worker, **_):
    """
    EWMA-Only matching when a worker becomes available.

    Design Decision: Worker-side matching uses GREEDY (nearest task) approach.
    Rationale: EWMA-Only enforces fairness when tasks choose workers (NEW_TASK event),
    not when workers choose tasks. On worker release, spatial efficiency is acceptable.

    Scans both deferred_tasks (tasks deferred by the simulator when no workers were
    available at arrival, or deferred by other strategies) and active_tasks (tasks
    that arrived while workers existed but found no feasible match) to ensure no
    pending task is missed.
    """
    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    best_task = None
    best_dist = float("inf")

    for task in pending:
        pickup_dist = manhattan_km(worker.start_lat, worker.start_lon,
                                  task.pickup_lat, task.pickup_lon)

        drop_dist = manhattan_km(task.pickup_lat, task.pickup_lon,
                                task.dropoff_lat, task.dropoff_lon)
        pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
        finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)

        if pickup_eta > task.expire_time or finish_eta > worker.deadline:
            continue

        if pickup_dist < best_dist:
            best_dist = pickup_dist
            best_task = task

    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_dist)

    return None


@register("ewma_only")
def get_ewma_only_handlers():
    """
    Register EWMA-Only strategy with event handlers.
    
    Returns:
        Dictionary mapping event types to handler functions
    """
    return {
        "NEW_TASK": assign_new_tasks_ewma_only,
        "FREE_WORKER": match_worker_ewma_only
    }

