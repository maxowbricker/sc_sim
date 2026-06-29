"""
FATP-ANN: Fairness-Aware Task Planning with Approximate Nearest Neighbor

Implementation of the FATP-ANN algorithm from the paper:
"Fair Task Allocation in Crowdsourced Delivery"

Key Features:
- Fairness cap (c_hat) to balance task distribution across workers
- Task-Process (TP): Assigns new tasks to nearest eligible worker
- Worker-Process (WP): Assigns multiple tasks to free workers based on utility
- Exponentially decayed utility function for task selection

Event Handlers:
- assign_new_tasks_fatp_ann: Handles TASK_RELEASE events (TP)
- match_worker_fatp_ann: Handles WORKER_RELEASE/TASK_COMPLETE events (WP)
"""

from simulator.strategies import register
from math import fabs, cos, radians, exp

AVG_SPEED_KMH = 30

def _ensure_timestamp(now):
    """Force timestamp to be a raw float for the optimized simulator."""
    if hasattr(now, "timestamp"):
        return now.timestamp()
    return float(now)


# ============================================================================
# FAIRNESS CAP TRACKER
# ============================================================================

class FairnessCapTracker:
    """
    Tracks the fairness cap (c_hat) using O(1) incremental updates.
    
    The fairness cap is calculated as:
        c_hat = sum(Count_i^2) / sum(Count_i)
    
    Where Count_i is the number of tasks completed by worker i.
    """
    
    def __init__(self):
        self.total_task_count_sum = 0
        self.total_task_count_squared_sum = 0
        self.worker_count = 0
    
    def initialize(self, workers):
        """Initialize tracker from current worker task counts."""
        self.total_task_count_sum = 0
        self.total_task_count_squared_sum = 0
        self.worker_count = len(workers)
        
        for w in workers:
            count = w.completed_tasks
            self.total_task_count_sum += count
            self.total_task_count_squared_sum += count ** 2
    
    def get_cap(self):
        """
        Calculate and return current fairness cap.
        Returns float('inf') if no tasks assigned yet.
        """
        if self.total_task_count_sum == 0:
            return float('inf')  # No cap when no tasks assigned
        return self.total_task_count_squared_sum / self.total_task_count_sum
    
    def update_worker_count(self, old_count, new_count):
        """
        Update tracker when a worker's task count changes.
        
        Args:
            old_count: Worker's previous task count
            new_count: Worker's new task count
        """
        self.total_task_count_sum += (new_count - old_count)
        self.total_task_count_squared_sum += (new_count ** 2 - old_count ** 2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def manhattan_km(lat1, lon1, lat2, lon2):
    """Calculate Manhattan distance in kilometers between two lat/lon points."""
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon


def _calculate_utility(task, worker_lat, worker_lon, worker_time, mu, alpha_scale):
    """
    Calculate utility for assigning a task from a worker's current position/time.
    
    Formula: u_r = alpha_r * exp(-mu * (completion_time - release_time))
    
    Args:
        task: Task object
        worker_lat, worker_lon: Worker's current location
        worker_time: Worker's current time (float in seconds)
        mu: Decay factor
        alpha_scale: Scaling factor for base utility
    
    Returns:
        float: Calculated utility value
    """
    # Calculate completion time
    pickup_dist = manhattan_km(worker_lat, worker_lon, task.pickup_lat, task.pickup_lon)
    service_dist = manhattan_km(task.pickup_lat, task.pickup_lon, 
                                task.dropoff_lat, task.dropoff_lon)
    
    total_hours = (pickup_dist + service_dist) / AVG_SPEED_KMH
    completion_time = worker_time + (total_hours * 3600)
    
    # Calculate time penalty (wait time since release)
    # Convert task.release_time to float if it's a Timestamp
    release_time_float = task.release_time if isinstance(task.release_time, (int, float)) else task.release_time.timestamp()
    wait_time_hours = (completion_time - release_time_float) / 3600.0
    
    # Calculate utility with exponential decay
    base_utility = task.base_utility * alpha_scale
    utility = base_utility * exp(-mu * wait_time_hours)
    
    return utility


def _is_valid_assignment(worker, task, now):
    """
    Check if a worker can feasibly complete a task from their current position.
    
    Checks:
    1. Can reach pickup before task expiry
    2. Can complete task before worker's deadline
    
    Args:
        worker: Worker object
        task: Task object
        now: Current simulation time
    
    Returns:
        bool: True if assignment is valid
    """
    now = _ensure_timestamp(now)
    pickup_dist = manhattan_km(worker.start_lat, worker.start_lon, 
                               task.pickup_lat, task.pickup_lon)
    service_dist = manhattan_km(task.pickup_lat, task.pickup_lon,
                               task.dropoff_lat, task.dropoff_lon)
    
    # Calculate ETAs
    pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
    finish_eta = now + (((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600)
    
    # Check constraints
    if pickup_eta > task.expire_time:
        return False
    if finish_eta > worker.deadline:
        return False
    
    return True


def _is_valid_from_shadow(task, shadow_location, shadow_time, worker):
    """
    Check if a task is valid from a shadow (hypothetical) worker state.
    Used in Worker-Process loop for multi-task assignment.
    
    Args:
        task: Task object
        shadow_location: Tuple (lat, lon) of worker's hypothetical location
        shadow_time: float timestamp of worker's hypothetical time
        worker: Worker object (for deadline)
    
    Returns:
        bool: True if assignment is valid from shadow state
    """
    shadow_lat, shadow_lon = shadow_location
    
    pickup_dist = manhattan_km(shadow_lat, shadow_lon, 
                               task.pickup_lat, task.pickup_lon)
    service_dist = manhattan_km(task.pickup_lat, task.pickup_lon,
                               task.dropoff_lat, task.dropoff_lon)
    
    # Calculate ETAs from shadow state
    pickup_eta = shadow_time + ((pickup_dist / AVG_SPEED_KMH) * 3600)
    finish_eta = shadow_time + (((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600)
    
    # Check constraints
    if pickup_eta > task.expire_time:
        return False
    if finish_eta > worker.deadline:
        return False
    
    return True


def _commit_assignment(task, worker, now):
    """
    Commit a task assignment to a worker.
    Calculates timing, updates task and worker state.
    
    Args:
        task: Task object to assign
        worker: Worker object to assign to
        now: Current simulation timestamp (float in seconds)
    
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


# ============================================================================
# TASK-PROCESS (TP) - NEW_TASK EVENT HANDLER
# ============================================================================

def assign_new_tasks_fatp_ann(state, now, tasks_to_assign, 
                              fairness_cap_tracker=None,
                              mu=0.5, 
                              alpha_scale=0.5,
                              use_k_nearest=False,
                              k=15,
                              **_):
    """
    Task-Process (TP): Handle newly arriving tasks.
    
    Algorithm:
    1. For each new task, scan all available workers (or k-nearest if enabled)
    2. Filter candidates: worker.completed_tasks_count < cap AND is_valid
    3. If candidates exist, assign to nearest worker
    4. Else, task remains in pool (simulation handles automatically)
    
    Args:
        state: StateManager with available workers and tasks
        now: Current simulation time
        tasks_to_assign: List of newly released tasks
        fairness_cap_tracker: FairnessCapTracker instance
        mu: Decay factor for utility calculation
        alpha_scale: Scaling factor for base utility
        use_k_nearest: If True, only consider k nearest workers (optimization)
        k: Number of nearest workers to consider if use_k_nearest=True
    
    Returns:
        List of (task, worker, distance) tuples for assignments made
    """
    now = _ensure_timestamp(now)
    if fairness_cap_tracker is None:
        raise ValueError("fairness_cap_tracker must be provided to FATP-ANN strategy")
    
    assignments = []
    cap = fairness_cap_tracker.get_cap()
    
    for task in tasks_to_assign:
        # Step 1: Get candidate workers
        if use_k_nearest:
            # Optimization: Only consider k nearest workers
            worker_distances = []
            for worker in state.available_workers:
                dist = manhattan_km(worker.start_lat, worker.start_lon, 
                                   task.pickup_lat, task.pickup_lon)
                worker_distances.append((worker, dist))
            worker_distances.sort(key=lambda x: x[1])
            candidate_pool = [w for w, _ in worker_distances[:k]]
        else:
            # Original algorithm: Consider all available workers
            candidate_pool = list(state.available_workers)
        
        # Step 2: Filter eligible candidates (fairness cap + validity check)
        eligible_candidates = []
        for worker in candidate_pool:
            # Check fairness cap
            if worker.completed_tasks >= cap:
                continue
            
            # Check validity (feasibility)
            if not _is_valid_assignment(worker, task, now):
                continue
            
            # Calculate distance for nearest selection
            dist = manhattan_km(worker.start_lat, worker.start_lon,
                               task.pickup_lat, task.pickup_lon)
            eligible_candidates.append((worker, dist))
        
        # Step 3: Assign to nearest eligible worker
        if eligible_candidates:
            # Find nearest candidate
            eligible_candidates.sort(key=lambda x: x[1])
            best_worker, best_dist = eligible_candidates[0]
            
            # Commit assignment
            old_count = best_worker.completed_tasks
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            
            # Update fairness cap tracker
            fairness_cap_tracker.update_worker_count(old_count, best_worker.completed_tasks)
            
            assignments.append((assigned_task, best_worker, best_dist))
        
        # Step 4: If no eligible worker, task remains in pool (automatic)
    
    return assignments


# ============================================================================
# WORKER-PROCESS (WP) - FREE_WORKER EVENT HANDLER
# ============================================================================

def match_worker_fatp_ann(state, now, worker, 
                         fairness_cap_tracker=None,
                         mu=0.5,
                         alpha_scale=0.5,
                         **_):
    """
    Worker-Process (WP): Handle worker becoming available.

    Algorithm with shadow state tracking:
    1. Initialize shadow state (location, time)
    2. While worker.completed_tasks_count < cap:
        a. Find all valid tasks from combined deferred + active task pools
        b. Calculate utility for each valid task
        c. Select task with max utility
        d. Assign task and update shadow state
    3. Return first assigned task (simulation expects single return)

    Scans both deferred_tasks and active_tasks so that tasks deferred by the
    simulator (no workers at arrival time) are also eligible for recovery.
    """
    if fairness_cap_tracker is None:
        raise ValueError("fairness_cap_tracker must be provided to FATP-ANN strategy")

    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    now = _ensure_timestamp(now)
    shadow_location = (worker.start_lat, worker.start_lon)
    shadow_time = now
    tasks_assigned_in_loop = []

    cap = fairness_cap_tracker.get_cap()

    while worker.completed_tasks < cap:
        # Rebuild pending each iteration — tasks may have been assigned in prior loops
        pending = list(state.deferred_tasks) + list(state.active_tasks)
        valid_tasks = []
        for task in pending:
            if _is_valid_from_shadow(task, shadow_location, shadow_time, worker):
                utility = _calculate_utility(task, shadow_location[0], shadow_location[1],
                                            shadow_time, mu, alpha_scale)
                valid_tasks.append((task, utility))

        if not valid_tasks:
            break

        best_task, best_utility = max(valid_tasks, key=lambda x: x[1])

        old_count = worker.completed_tasks
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        tasks_assigned_in_loop.append(assigned_task)

        fairness_cap_tracker.update_worker_count(old_count, worker.completed_tasks)

        pickup_dist = manhattan_km(shadow_location[0], shadow_location[1],
                                   best_task.pickup_lat, best_task.pickup_lon)
        service_dist = manhattan_km(best_task.pickup_lat, best_task.pickup_lon,
                                    best_task.dropoff_lat, best_task.dropoff_lon)

        shadow_time = shadow_time + (((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600)
        shadow_location = (best_task.dropoff_lat, best_task.dropoff_lon)

    if tasks_assigned_in_loop:
        return (tasks_assigned_in_loop[0], worker, None)

    return None


# ============================================================================
# STRATEGY REGISTRATION
# ============================================================================

@register("fatp_ann")
def get_fatp_ann_handlers():
    """
    Return event handlers for FATP-ANN strategy.
    
    Returns:
        Dict mapping event types to handler functions
    """
    return {
        "NEW_TASK": assign_new_tasks_fatp_ann,
        "FREE_WORKER": match_worker_fatp_ann
    }

