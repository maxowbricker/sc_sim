import pandas as pd
from simulator.spatial_index import fast_manhattan_km
import math

class Task:
    def __init__(self, task_dict):
        self.id = task_dict["task_id"]
        self.pickup_lat = float(task_dict["pickup_lat"])
        self.pickup_lon = float(task_dict["pickup_lon"])
        self.dropoff_lat = float(task_dict["dropoff_lat"])
        self.dropoff_lon = float(task_dict["dropoff_lon"])
        self.release_time = float(task_dict["release_time"])
        self.expire_time = float(task_dict["expire_time"])
        self.assigned_worker = None
        self.assigned = False
        self.is_completed = False

        # Service-time bookkeeping
        self.finish_time = None  # float Unix timestamp when task completes
        self.start_time = None   # float Unix timestamp when service starts
        self.pickup_km = None
        self.drop_km = None
        self.deferral_count = 0  # Number of times this task was deferred

        # Base utility for FATP-ANN strategy (lazy-loaded; only computed when strategy uses it)
        self._base_utility = None

    @property
    def base_utility(self):
        """Pickup→dropoff distance (km). Lazy-loaded; only computed when FATP-ANN strategy uses it."""
        if self._base_utility is None:
            self._base_utility = self._calculate_base_utility()
        return self._base_utility

    def _calculate_base_utility(self):
            """
            Calculate base utility for FATP-ANN strategy.
            Base utility is proportional to task distance (pickup to dropoff).
            """
            if math.isnan(self.pickup_lat) or math.isnan(self.dropoff_lat):
                raise ValueError(f"Task {self.id} has NaN coordinates. Cannot calculate spatial utility.")
                
            return fast_manhattan_km(
                self.pickup_lat, self.pickup_lon,
                self.dropoff_lat, self.dropoff_lon
            )
    
    def is_available(self, current_time):
        return (not self.assigned) and self.release_time <= current_time <= self.expire_time

    def assign_to_worker(self, worker):
        self.assigned_worker = worker
        self.assigned = True

    # ------------------------------------------------------------------
    # Oracle state serialisation (primitives only — no object refs)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Return all mutable fields as a dict of primitives for fast snapshot."""
        return {
            'assigned': self.assigned,
            'is_completed': self.is_completed,
            'finish_time': self.finish_time,
            'start_time': self.start_time,
            'pickup_km': self.pickup_km,
            'drop_km': self.drop_km,
            'deferral_count': self.deferral_count,
            'assigned_worker_id': self.assigned_worker.id if self.assigned_worker is not None else None,
        }

    def restore_from_dict(self, d: dict) -> None:
        """Overwrite mutable fields from a snapshot dict.

        Note: assigned_worker is restored at the StateManager level (because
        it requires a worker object reference), so it is not set here.
        """
        self.assigned = d['assigned']
        self.is_completed = d['is_completed']
        self.finish_time = d['finish_time']
        self.start_time = d['start_time']
        self.pickup_km = d['pickup_km']
        self.drop_km = d['drop_km']
        self.deferral_count = d['deferral_count']