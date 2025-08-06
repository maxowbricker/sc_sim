from __future__ import annotations

from datetime import datetime
import pandas as pd
from config import SIM_CONFIG


class Worker:
    """Domain model for a spatial-crowdsourcing worker.

    Augmented with bookkeeping fields that will be required for metrics such as
    EWMA fairness, revenue distribution, etc.
    """

    # ------------------------------------------------------------------ #
    # Construction & basic properties
    # ------------------------------------------------------------------ #

    def __init__(self, worker_dict):
        # Static attributes (from dataset)
        self.id = worker_dict["worker_id"]
        self.start_lat = float(worker_dict["start_lat"])
        self.start_lon = float(worker_dict["start_lon"])
        self.release_time = pd.to_datetime(worker_dict["release_time"])
        self.deadline = pd.to_datetime(worker_dict["deadline"])

        # Dynamic state
        self.assigned_task = None
        self.available = True

        # ------------------------------------------------------------------ #
        # Metrics-related counters
        # ------------------------------------------------------------------ #
        self.total_idle_time = pd.Timedelta(0)  # cumulative idle duration
        self.last_state_ts: pd.Timestamp | None = self.release_time  # last time idle counter updated

        # EWMA fairness tracking
        self.gamma: float = SIM_CONFIG.get("strategy_params", {}).get("gamma", 0.3)
        self.fairness_ewma: float = 0.0  # starts at 0 (no under-service yet)

        self.completed_tasks: int = 0
        self.revenue: float = 0.0  # placeholder – will depend on task info
        self.last_active_ts: pd.Timestamp | None = None  # when last task finished

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #

    def assign_task(self, task):
        """Mark worker as busy with *task*."""
        self.assigned_task = task
        self.available = False

    def record_completion(self, now: pd.Timestamp, task_revenue: float = 0.0):
        """Update counters when a task is completed."""
        self.completed_tasks += 1
        self.revenue += float(task_revenue)
        self.last_active_ts = now

        # Worker becomes available again; `StateManager` will also toggle pools.
        self.available = True

    # ------------------------------------------------------------------ #
    # Availability helpers
    # ------------------------------------------------------------------ #

    def is_available(self, current_time):
        return self.available and self.release_time <= current_time <= self.deadline

    def update_idle_time(self, time_delta_seconds: float):
        """Update idle time by a fixed duration and recalculate EWMA fairness."""
        if not self.available:
            return

        # Update cumulative idle duration
        self.total_idle_time += pd.to_timedelta(time_delta_seconds, unit='s')

        # EWMA update using the time delta (not cumulative)
        # This follows the methodology: (1-γ)*T_idle(w_i) + γ*Previous_EWMA
        self.fairness_ewma = (1 - self.gamma) * time_delta_seconds + self.gamma * self.fairness_ewma
