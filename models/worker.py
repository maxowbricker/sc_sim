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

        self.assigned_task = None     # Core link between worker and task; used to determine current busy state.
        self.available = True         # Fast boolean lookup; works with release_time and deadline in is_available().

        self.total_idle_time = 0.0    # Cumulative metric used by MetricsManager for RL rewards and final stats.
        self.last_state_ts = self.release_time  # The "lap button"; used to calculate idle deltas between RL steps.
        self.fairness_ewma = 0.0      # Persists the decaying fairness score between assignments.
        self.last_active_ts = None    # The "Fairness Anchor"; records last completion to calculate wait time for next assignment.

        self.completed_tasks = 0      # Used for JFI stats and as a physical constraint/cap in FATP-ANN strategy.

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

    # ------------------------------------------------------------------
    # Oracle state serialisation (primitives only — no object refs)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Return all mutable fields as a dict of primitives for fast snapshot."""
        return {
            'start_lat': self.start_lat,
            'start_lon': self.start_lon,
            'available': self.available,
            'total_idle_time': self.total_idle_time,
            'last_state_ts': self.last_state_ts,
            'fairness_ewma': self.fairness_ewma,
            'last_active_ts': self.last_active_ts,
            'completed_tasks': self.completed_tasks,
            'assigned_task_id': self.assigned_task.id if self.assigned_task is not None else None,
        }

    def restore_from_dict(self, d: dict) -> None:
        """Overwrite mutable fields from a snapshot dict.

        Note: assigned_task is restored at the StateManager level (because
        it requires a task object reference), so it is not set here.
        """
        self.start_lat = d['start_lat']
        self.start_lon = d['start_lon']
        self.available = d['available']
        self.total_idle_time = d['total_idle_time']
        self.last_state_ts = d['last_state_ts']
        self.fairness_ewma = d['fairness_ewma']
        self.last_active_ts = d['last_active_ts']
        self.completed_tasks = d['completed_tasks']

