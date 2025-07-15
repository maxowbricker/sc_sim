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

    def update_idle(self, current_time: pd.Timestamp):
        """Accumulate idle time up to *current_time* if worker is available."""
        if self.last_state_ts is None:
            self.last_state_ts = current_time

        if self.available and current_time >= self.last_state_ts:
            delta_td = current_time - self.last_state_ts
            # Update cumulative idle duration
            self.total_idle_time += delta_td

            # Current measurement: total idle seconds so far
            current_idle_sec = self.total_idle_time.total_seconds()

            # EWMA update: (1-γ)*current + γ*previous
            self.fairness_ewma = (1 - self.gamma) * current_idle_sec + self.gamma * self.fairness_ewma

        # Whether idle or busy, advance the marker so next call measures from here.
        self.last_state_ts = current_time