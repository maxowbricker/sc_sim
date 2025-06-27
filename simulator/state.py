from datetime import datetime
import pandas as pd

class StateManager:
    def __init__(self, all_workers, all_tasks):
        self.all_workers = all_workers
        self.all_tasks = all_tasks
        self.available_workers = []
        self.active_tasks = []
        self.assigned_tasks = []
        self.assigned_workers = []

    def release_workers(self, current_time):
        """
        Moves workers whose release_time <= current_time to the available pool.
        """
        to_release = [w for w in self.all_workers if pd.to_datetime(w["release_time"]) <= current_time]
        self.available_workers.extend(to_release)
        self.all_workers = [w for w in self.all_workers if w not in to_release]

    def release_tasks(self, current_time):
        """
        Moves tasks whose release_time <= current_time to the active task pool.
        """
        to_release = [t for t in self.all_tasks if pd.to_datetime(t["release_time"]) <= current_time]
        self.active_tasks.extend(to_release)
        self.all_tasks = [t for t in self.all_tasks if t not in to_release]

    def assign_task(self, task, worker):
        """
        Assign a task to a worker and update internal tracking.
        """
        self.active_tasks.remove(task)
        self.available_workers.remove(worker)
        self.assigned_tasks.append(task)
        self.assigned_workers.append(worker)

    def step(self, current_time):
        """
        Performs state update for current timestep.
        """
        self.release_workers(current_time)
        self.release_tasks(current_time)
