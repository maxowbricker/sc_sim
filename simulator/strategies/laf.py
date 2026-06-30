"""
LAF (Least Allocated Worker First) Strategy

Pure fairness baseline that assigns tasks to workers with the fewest completed tasks.
Prioritizes equitable workload distribution over spatial efficiency.

Research Context:
- Represents "Simple Fairness-Only" baseline for comparison with composite strategy
- Uses cumulative task count as fairness metric
- No spatial optimization (assigns to least-loaded worker regardless of distance)
"""

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30


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
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                        task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
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


def assign_new_tasks_laf(state, now, tasks_to_assign, **_):
    """
    LAF assignment for new tasks arriving in the system.
    
    Core Logic:
    1. For each task, find all feasible workers (can meet pickup/deadline constraints)
    2. From feasible set, select worker with MINIMUM completed_tasks
    3. Use distance as tie-breaker if multiple workers have same task count
    4. Defer task if no feasible worker exists
    
    This enforces fairness by prioritizing under-utilized workers.
    
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
        best_task_count = float("inf")
        best_dist = float("inf")  # Tie-breaker
        
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                      task.dropoff_lat, task.dropoff_lon)

        for worker in state.available_workers:
            pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                            task.pickup_lat, task.pickup_lon)
            
            # Feasibility check: pickup before expiry, finish before worker shift ends
            pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
            finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)
            
            if pickup_eta > task.expire_time or finish_eta > worker.deadline:
                continue
            
            # LAF CORE LOGIC: Select worker with fewest completed tasks
            worker_task_count = worker.completed_tasks
            
            if worker_task_count < best_task_count:
                # Found worker with fewer tasks
                best_task_count = worker_task_count
                best_worker = worker
                best_dist = pickup_dist
            elif worker_task_count == best_task_count and pickup_dist < best_dist:
                # Tie on task count - use distance as tie-breaker
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


def match_worker_laf(state, now, worker, **_):
    """
    LAF matching when a worker becomes available.

    Design Decision: Worker-side matching uses GREEDY (nearest task) approach.
    Rationale: LAF enforces fairness when tasks choose workers (NEW_TASK event),
    not when workers choose tasks. On worker release, spatial efficiency is acceptable.

    Scans both deferred_tasks (tasks deferred by the simulator when no workers were
    available at arrival) and active_tasks (tasks that arrived while workers existed
    but found no feasible match) to ensure no pending task is missed.
    """
    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    best_task = None
    best_dist = float("inf")

    for task in pending:
        pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                        task.pickup_lat, task.pickup_lon)

        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
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


@register("laf")
def get_laf_handlers():
    """
    Register LAF strategy with event handlers.
    
    Returns:
        Dictionary mapping event types to handler functions
    """
    return {
        "NEW_TASK": assign_new_tasks_laf,
        "FREE_WORKER": match_worker_laf
    }

