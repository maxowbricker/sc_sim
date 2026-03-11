from __future__ import annotations

from datetime import datetime
import pandas as pd


class Worker:
    """Domain model for a spatial-crowdsourcing worker.

    Augmented with bookkeeping fields that will be required for metrics such as
    EWMA fairness, revenue distribution, etc.
    """

    def __init__(self, worker_dict):
        # Static attributes (from dataset)
        self.id = worker_dict["worker_id"]
        self.start_lat = float(worker_dict["start_lat"])
        self.start_lon = float(worker_dict["start_lon"])
        self.release_time = float(worker_dict["release_time"])
        self.deadline = float(worker_dict["deadline"])

        # Dynamic state
        self.assigned_task = None
        self.available = True

        self.total_idle_time = 0.0  # cumulative idle duration in seconds
        self.last_state_ts: float | None = self.release_time

        # EWMA fairness tracking
        self.fairness_ewma: float = 0.0  # starts at 0 (no under-service yet)

        self.completed_tasks: int = 0
        self.revenue: float = 0.0  # placeholder – will depend on task info
        self.last_active_ts: float | None = None  # when last task finished

    def assign_task(self, task):
        """Mark worker as busy with *task*."""
        self.assigned_task = task
        self.available = False

    def record_completion(self, now, task_revenue: float = 0.0):
        """Update counters when a task is completed.
        
        Args:
            now: Current time (pd.Timestamp or float Unix timestamp)
            task_revenue: Revenue from completed task
        """
        self.completed_tasks += 1
        self.revenue += float(task_revenue)
        self.last_active_ts = now
        
        # Update last_state_ts for idle time tracking
        self.last_state_ts = now

        # Worker becomes available again; `StateManager` will also toggle pools.
        self.available = True

    def is_available(self, current_time):
        return self.available and self.release_time <= current_time <= self.deadline

    def update_idle_time(self, time_delta_seconds: float):
        """Update cumulative idle time for reporting purposes.
        
        Args:
            time_delta_seconds: Time delta in seconds (float)
        """
        if not self.available:
            return

        self.total_idle_time += time_delta_seconds

