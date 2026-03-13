from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30

def _commit_assignment(task, worker, now):
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # Realistic timing - task starts after worker travels to pickup location  
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    
    # Pure float math (hours to seconds)
    task.start_time = now + (pickup_travel_hours * 3600)  
    task.finish_time = task.start_time + (service_travel_hours * 3600)  
    
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def assign_new_tasks_greedy(state, now, tasks_to_assign, **_):
    """
    Greedy assignment for newly released tasks. Finds the nearest available worker.
    """
    assignments = []
    
    for task in tasks_to_assign:
        if not state.available_workers:
            state.defer_task(task, now)
            continue
            
        best_worker, best_dist = None, float("inf")
        
        for worker in state.available_workers:
            pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
            drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
            
            # Pure float math
            pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
            finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)
            
            if pickup_eta > task.expire_time or finish_eta > worker.deadline:
                continue

            if pickup_dist < best_dist:
                best_dist = pickup_dist
                best_worker = worker
                
        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))
        else:
            state.defer_task(task, now)
            
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
        pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, task.pickup_lat, task.pickup_lon)
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, task.dropoff_lat, task.dropoff_lon)
        
        # Pure float math
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


@register("greedy")
def get_greedy_handlers():
    return {
        "NEW_TASK": assign_new_tasks_greedy,
        "FREE_WORKER": match_worker_greedy
    }