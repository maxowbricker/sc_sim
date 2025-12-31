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
from math import exp
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30


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

def _calculate_utility(task, worker_lat, worker_lon, worker_time, mu, alpha_scale):
    """
    Calculate utility for assigning a task from a worker's current position/time.
    
    Formula: u_r = alpha_r * exp(-mu * (completion_time - release_time))
    
    OPTIMIZED: Uses float math instead of pandas Timedelta for better performance.
    
    Args:
        task: Task object
        worker_lat, worker_lon: Worker's current location
        worker_time: Worker's current time (float Unix timestamp)
        mu: Decay factor
        alpha_scale: Scaling factor for base utility
    
    Returns:
        float: Calculated utility value
    """
    # Ensure worker_time is a float timestamp
    if not isinstance(worker_time, (int, float)):
        worker_time = worker_time.timestamp() if hasattr(worker_time, 'timestamp') else float(worker_time)
    
    # OPTIMIZED: Use fast_manhattan_km and float math
    pickup_dist = fast_manhattan_km(worker_lat, worker_lon, task.pickup_lat, task.pickup_lon)
    service_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, 
                                    task.dropoff_lat, task.dropoff_lon)
    
    # OPTIMIZED: Use float math instead of Timedelta
    total_seconds = ((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600
    completion_time = worker_time + total_seconds
    
    # Calculate time penalty (wait time since release) - all floats
    wait_time_hours = (completion_time - task.release_time) / 3600.0
    
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
    
    OPTIMIZED: Uses float math instead of pandas Timedelta for better performance.
    
    Args:
        worker: Worker object
        task: Task object
        now: Current simulation time (float Unix timestamp)
    
    Returns:
        bool: True if assignment is valid
    """
    # Ensure now is a float timestamp
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    # OPTIMIZED: Use fast_manhattan_km
    pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon, 
                                   task.pickup_lat, task.pickup_lon)
    service_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                   task.dropoff_lat, task.dropoff_lon)
    
    # OPTIMIZED: Use float math instead of Timedelta
    pickup_eta_seconds = (pickup_dist / AVG_SPEED_KMH) * 3600
    finish_eta_seconds = ((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600
    
    # Compare timestamps directly (all floats)
    if (now + pickup_eta_seconds) > task.expire_time:
        return False
    if (now + finish_eta_seconds) > worker.deadline:
        return False
    
    return True


def _is_valid_from_shadow(task, shadow_location, shadow_time, worker):
    """
    Check if a task is valid from a shadow (hypothetical) worker state.
    Used in Worker-Process loop for multi-task assignment.
    
    OPTIMIZED: Uses float math instead of pandas Timedelta for better performance.
    
    Args:
        task: Task object
        shadow_location: Tuple (lat, lon) of worker's hypothetical location
        shadow_time: float Unix timestamp of worker's hypothetical time
        worker: Worker object (for deadline)
    
    Returns:
        bool: True if assignment is valid from shadow state
    """
    # Ensure shadow_time is a float timestamp
    if not isinstance(shadow_time, (int, float)):
        shadow_time = shadow_time.timestamp() if hasattr(shadow_time, 'timestamp') else float(shadow_time)
    
    shadow_lat, shadow_lon = shadow_location
    
    # OPTIMIZED: Use fast_manhattan_km
    pickup_dist = fast_manhattan_km(shadow_lat, shadow_lon, 
                                   task.pickup_lat, task.pickup_lon)
    service_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                   task.dropoff_lat, task.dropoff_lon)
    
    # OPTIMIZED: Use float math instead of Timedelta
    pickup_eta_seconds = (pickup_dist / AVG_SPEED_KMH) * 3600
    finish_eta_seconds = ((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600
    
    # Compare timestamps directly (all floats)
    if (shadow_time + pickup_eta_seconds) > task.expire_time:
        return False
    if (shadow_time + finish_eta_seconds) > worker.deadline:
        return False
    
    return True


def _commit_assignment(task, worker, now):
    """
    Commit a task assignment to a worker.
    Calculates timing, updates task and worker state.
    
    OPTIMIZED: Uses float math instead of pandas Timedelta for better performance.
    
    Args:
        task: Task object to assign
        worker: Worker object to assign to
        now: Current simulation timestamp (float Unix timestamp)
    
    Returns:
        The assigned task object
    """
    # Ensure now is a float timestamp
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    # OPTIMIZED: Use fast_manhattan_km
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon, 
                                       task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon, 
                                     task.dropoff_lat, task.dropoff_lon)
    
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    
    # OPTIMIZED: Use float math instead of Timedelta
    pickup_travel_seconds = (pickup_distance / AVG_SPEED_KMH) * 3600
    service_travel_seconds = (drop_distance / AVG_SPEED_KMH) * 3600
    
    task.start_time = now + pickup_travel_seconds
    task.finish_time = task.start_time + service_travel_seconds
    
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
                              use_k_nearest=True,
                              k=100,
                              **_):
    """
    Task-Process (TP): Handle newly arriving tasks.
    
    Algorithm:
    1. For each new task, scan k nearest workers using spatial index (or all if disabled)
    2. Filter candidates: worker.completed_tasks_count < cap AND is_valid
    3. If candidates exist, assign to nearest worker
    4. Else, task remains in pool (simulation handles automatically)
    
    OPTIMIZED: Uses spatial index for efficient k-nearest neighbor search.
    This reduces complexity from O(|W|) to O(k) where k=100 << |W|.
    
    Args:
        state: StateManager with available workers and tasks
        now: Current simulation time (float Unix timestamp)
        tasks_to_assign: List of newly released tasks
        fairness_cap_tracker: FairnessCapTracker instance
        mu: Decay factor for utility calculation
        alpha_scale: Scaling factor for base utility
        use_k_nearest: If True, use spatial index to find k nearest workers (default: True)
        k: Number of nearest workers to consider if use_k_nearest=True (default: 100)
    
    Returns:
        List of (task, worker, distance) tuples for assignments made
    """
    # Ensure now is a float timestamp
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    if fairness_cap_tracker is None:
        raise ValueError("fairness_cap_tracker must be provided to FATP-ANN strategy")
    
    assignments = []
    cap = fairness_cap_tracker.get_cap()
    
    for task in tasks_to_assign:
        # OPTIMIZATION 1: Pre-calculate drop distance (constant for all workers)
        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon, 
                                      task.dropoff_lat, task.dropoff_lon)
        
        # Step 1: Get candidate workers
        if use_k_nearest:
            # OPTIMIZATION 2: Use spatial index for efficient k-nearest search
            # This reduces from 35,449 workers to ~100 workers checked per task!
            candidate_pool = state.spatial_index.query_k_nearest(
                task.pickup_lat, task.pickup_lon, k
            )
        else:
            # Original algorithm: Consider all available workers (slow!)
            candidate_pool = list(state.available_workers)
        
        # Step 2: Filter eligible candidates (fairness cap + validity check)
        eligible_candidates = []
        for worker in candidate_pool:
            # Check fairness cap
            if worker.completed_tasks >= cap:
                continue
            
            # OPTIMIZATION 3: Use fast_manhattan_km and pre-calculated drop_dist
            pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                           task.pickup_lat, task.pickup_lon)
            
            # Check validity (feasibility) using optimized function
            # We can inline the check here for better performance
            pickup_eta_seconds = (pickup_dist / AVG_SPEED_KMH) * 3600
            finish_eta_seconds = ((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600
            
            if (now + pickup_eta_seconds) > task.expire_time or (now + finish_eta_seconds) > worker.deadline:
                continue
            
            eligible_candidates.append((worker, pickup_dist))
        
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
        a. Find all valid tasks from active_tasks
        b. Calculate utility for each valid task
        c. Select task with max utility
        d. Assign task and update shadow state
    3. Return first assigned task (simulation expects single return)
    
    OPTIMIZED: Uses float math instead of pandas Timedelta for shadow state updates.
    
    NOTE: Active tasks don't have a spatial index, so we iterate through all active tasks.
    This is usually fine since active_tasks is typically smaller than available_workers.
    
    Args:
        state: StateManager with available workers and tasks
        now: Current simulation time (float Unix timestamp)
        worker: Newly available worker
        fairness_cap_tracker: FairnessCapTracker instance
        mu: Decay factor for utility calculation
        alpha_scale: Scaling factor for base utility
    
    Returns:
        (task, worker, None) tuple if assignment made, else None
    """
    if fairness_cap_tracker is None:
        raise ValueError("fairness_cap_tracker must be provided to FATP-ANN strategy")
    
    if not state.active_tasks:
        return None
    
    # Ensure now is a float timestamp
    if not isinstance(now, (int, float)):
        now = now.timestamp() if hasattr(now, 'timestamp') else float(now)
    
    # Initialize shadow state
    shadow_location = (worker.start_lat, worker.start_lon)
    shadow_time = now  # Already a float
    tasks_assigned_in_loop = []
    
    cap = fairness_cap_tracker.get_cap()
    
    # Multi-task assignment loop
    while worker.completed_tasks < cap:
        # Step 2a: Find all valid tasks from active_tasks
        valid_tasks = []
        for task in list(state.active_tasks):  # Iterate over copy
            if _is_valid_from_shadow(task, shadow_location, shadow_time, worker):
                # Step 2b: Calculate utility
                utility = _calculate_utility(task, shadow_location[0], shadow_location[1],
                                            shadow_time, mu, alpha_scale)
                valid_tasks.append((task, utility))
        
        if not valid_tasks:
            break  # No more valid tasks
        
        # Step 2c: Select task with max utility
        best_task, best_utility = max(valid_tasks, key=lambda x: x[1])
        
        # Step 2d: Assign task
        old_count = worker.completed_tasks
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        tasks_assigned_in_loop.append(assigned_task)
        
        # Update fairness cap tracker
        fairness_cap_tracker.update_worker_count(old_count, worker.completed_tasks)
        
        # OPTIMIZED: Update shadow state using float math
        pickup_dist = fast_manhattan_km(shadow_location[0], shadow_location[1],
                                       best_task.pickup_lat, best_task.pickup_lon)
        service_dist = fast_manhattan_km(best_task.pickup_lat, best_task.pickup_lon,
                                        best_task.dropoff_lat, best_task.dropoff_lon)
        
        # OPTIMIZED: Use float math instead of Timedelta
        travel_seconds = ((pickup_dist + service_dist) / AVG_SPEED_KMH) * 3600
        shadow_time = shadow_time + travel_seconds
        shadow_location = (best_task.dropoff_lat, best_task.dropoff_lon)
    
    # Step 3: Return first task (simulation expects single task return)
    # Other tasks are already committed via state.assign_task
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

