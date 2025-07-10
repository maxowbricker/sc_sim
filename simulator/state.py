from datetime import datetime
from config import SIM_CONFIG


class StateManager:
    def __init__(self, all_workers=None, all_tasks=None):
        """
        Parameters
        ----------
        all_workers : list[Worker]
        all_tasks   : list[Task]
        """
        self.all_workers = all_workers if all_workers is not None else []
        self.all_tasks = all_tasks if all_tasks is not None else []

        # Dynamic pools
        self.available_workers = []     # workers that can take a task now
        self.active_tasks = []          # released but unassigned tasks

        self.assigned_tasks = []        # currently assigned tasks
        self.assigned_workers = []      # workers currently busy
        self.completed_tasks = []       # tasks finished during simulation

    # ------------------------------------------------------------------
    # Release logic (called each timestep)
    # ------------------------------------------------------------------
    def release_workers(self, current_time):
        """
        Move workers whose release_time <= current_time into the available pool.
        """
        to_release = [w for w in self.all_workers if w.release_time <= current_time]
        self.available_workers.extend(to_release)
        self.all_workers = [w for w in self.all_workers if w not in to_release]

    def release_tasks(self, current_time):
        """
        Move tasks whose release_time <= current_time into the active pool.
        """
        to_release = [t for t in self.all_tasks if t.release_time <= current_time]
        self.active_tasks.extend(to_release)
        self.all_tasks = [t for t in self.all_tasks if t not in to_release]

    # ------------------------------------------------------------------
    # Assignment helpers
    # ------------------------------------------------------------------
    def assign_task(self, task, worker):
        """
        Mark a task as assigned to the given worker.
        """
        self.active_tasks.remove(task)
        self.available_workers.remove(worker)

        self.assigned_tasks.append(task)
        self.assigned_workers.append(worker)

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------
    def complete_task(self, task, worker, current_time):
        """
        Mark a task as completed and free the worker so they can accept new tasks.
        """
        # Remove from assigned tracking if present
        if task in self.assigned_tasks:
            self.assigned_tasks.remove(task)
        if worker in self.assigned_workers:
            self.assigned_workers.remove(worker)

        # Record completion
        self.completed_tasks.append(task)

        # Let worker update personal counters
        worker.record_completion(current_time)

        # Optional teleport to drop-off location
        if SIM_CONFIG.get("teleport_on_complete", False):
            worker.start_lat = task.dropoff_lat
            worker.start_lon = task.dropoff_lon

        # Pools maintenance
        self.available_workers.append(worker)

    # ------------------------------------------------------------------
    # Per-timestep update
    # ------------------------------------------------------------------
    def step(self, current_time):
        """
        Perform release updates for the current timestep.
        """
        self.release_workers(current_time)
        self.release_tasks(current_time)

        # Update idle time counters for all currently available workers
        for w in self.available_workers:
            w.update_idle(current_time)

        # Check assigned tasks for completion (distance-based service mode)
        completed_now = [t for t in self.assigned_tasks if getattr(t, "finish_time", None) is not None and t.finish_time <= current_time]
        for task in completed_now:
            worker = task.assigned_worker  # set during assignment
            self.complete_task(task, worker, task.finish_time)