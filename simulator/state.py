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
        self.all_workers_map = {w.id: w for w in all_workers} if all_workers else {}
        self.all_tasks_map = {t.id: t for t in all_tasks} if all_tasks else {}
        
        # Dynamic pools
        self.available_workers = []
        self.active_tasks = []          # Released but unassigned
        self.deferred_tasks = []        # Scored below threshold, waiting
        
        self.assigned_tasks = []
        self.assigned_workers = []
        self.completed_tasks = []
        
        # Logging for analysis
        self.assignment_log = []

    def get_worker(self, worker_id):
        return self.all_workers_map.get(worker_id)

    def get_task(self, task_id):
        return self.all_tasks_map.get(task_id)

    def release_worker(self, worker):
        if worker not in self.available_workers:
            self.available_workers.append(worker)

    def release_task(self, task):
        if task not in self.active_tasks:
            self.active_tasks.append(task)

    def assign_task(self, task, worker):
        if task in self.active_tasks:
            self.active_tasks.remove(task)
        if task in self.deferred_tasks:
            self.deferred_tasks.remove(task)
        
        self.available_workers.remove(worker)
        self.assigned_tasks.append(task)
        self.assigned_workers.append(worker)

    def defer_task(self, task):
        if task in self.active_tasks:
            self.active_tasks.remove(task)
        if task not in self.deferred_tasks:
            self.deferred_tasks.append(task)

    def complete_task(self, task, worker, current_time):
        if task in self.assigned_tasks:
            self.assigned_tasks.remove(task)
        if worker in self.assigned_workers:
            self.assigned_workers.remove(worker)

        task.is_completed = True
        self.completed_tasks.append(task)
        worker.record_completion(current_time)

        # The worker is now at the drop-off location of the completed task.
        # This state update is crucial for subsequent assignments.
        worker.start_lat = task.dropoff_lat
        worker.start_lon = task.dropoff_lon
            
        self.available_workers.append(worker)
