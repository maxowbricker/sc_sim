from datetime import datetime
from simulator.spatial_index import GridSpatialIndex


class StateManager:
    def __init__(self, all_workers=None, all_tasks=None):
        """
        Parameters
        ----------
        all_workers : list[Worker]
        all_tasks   : list[Task]
        """
        self.all_workers_map = {w.id: w for w in all_workers} if all_workers else {}
        self.all_tasks_map = {t.id: t for t in all_tasks} if all_tasks else {}
        
        # Dynamic pools - Using sets for O(1) operations instead of O(n) list operations
        self.available_workers = set()
        self.active_tasks = set()          # Released but unassigned
        self.deferred_tasks = set()        # Scored below threshold, waiting
        
        self.assigned_tasks = set()
        self.assigned_workers = set()
        self.completed_tasks = set()
        
        # Initialize the Spatial Index for efficient nearest neighbor search
        self.spatial_index = GridSpatialIndex()
        
        # PERFORMANCE FIX: Assignment logging removed - was causing memory bloat without any benefit
        # Real statistics are collected via simulation summary and fairness tracker
        
        # Deferred tasks monitoring (optional)
        self.deferred_monitor = None  # Set by simulation if monitoring enabled

    def get_worker(self, worker_id):
        return self.all_workers_map.get(worker_id)

    def get_task(self, task_id):
        return self.all_tasks_map.get(task_id)

    def release_worker(self, worker):
        self.available_workers.add(worker)
        # ADD TO INDEX
        self.spatial_index.add(worker)

    def release_task(self, task):
        self.active_tasks.add(task)

    def assign_task(self, task, worker):
        # Remove from active/deferred pools (O(1) operations with sets)
        self.active_tasks.discard(task)      # discard() won't raise error if not present
        self.deferred_tasks.discard(task)
        
        # Move worker from available to assigned
        self.available_workers.discard(worker)
        
        # REMOVE FROM INDEX
        # Note: Worker must be at the same location as when they were added
        self.spatial_index.remove(worker)
        
        self.assigned_tasks.add(task)
        self.assigned_workers.add(worker)

    def defer_task(self, task, current_time=None):
        """
        Defer a task to the deferred pool.
        
        Args:
            task: Task to defer
            current_time: Current simulation time (optional, for expiry check)
        
        Returns:
            bool: True if task was deferred, False if already expired
        """
        # Check if task is already expired (if current_time provided)
        if current_time is not None and current_time > task.expire_time:
            return False  # Don't defer already-expired tasks
        
        # Move from active to deferred (O(1) operations)
        self.active_tasks.discard(task)
        self.deferred_tasks.add(task)
        task.deferral_count += 1  # Track number of times this task was deferred
        return True

    def complete_task(self, task, worker, current_time):
        # Remove from assigned pools (O(1) operations with sets)
        self.assigned_tasks.discard(task)
        self.assigned_workers.discard(worker)

        task.is_completed = True
        self.completed_tasks.add(task)
        worker.record_completion(current_time)

        # The worker is now at the drop-off location of the completed task.
        # This state update is crucial for subsequent assignments.
        worker.start_lat = task.dropoff_lat
        worker.start_lon = task.dropoff_lon
            
        # Worker becomes available again
        self.available_workers.add(worker)
        
        # ADD TO INDEX (at new location)
        self.spatial_index.add(worker)
