from __future__ import annotations

import math

from config import get_platform_revenue_config
from simulator.spatial_index import fast_manhattan_km


def core_movement_cost_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon) -> float:
    """α in Basık et al.: geospatial distance between pickup and dropoff."""
    return fast_manhattan_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)


def platform_revenue_from_alpha(alpha_km: float, base_fare: float | None = None, per_km_rate: float | None = None) -> float:
    """Platform monetary value t_j.m = base_fare + per_km_rate × α."""
    cfg = get_platform_revenue_config()
    base = cfg["base_fare"] if base_fare is None else base_fare
    rate = cfg["per_km_rate"] if per_km_rate is None else per_km_rate
    return base + rate * alpha_km


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

        # Platform revenue (Basık et al.): intrinsic task value from trip distance.
        # Requires set_city_constants() before Task construction (data loader does this).
        if math.isnan(self.pickup_lat) or math.isnan(self.dropoff_lat):
            raise ValueError(f"Task {self.id} has NaN coordinates. Cannot calculate revenue.")
        self.core_movement_cost_km = core_movement_cost_km(
            self.pickup_lat, self.pickup_lon, self.dropoff_lat, self.dropoff_lon
        )
        self.revenue = platform_revenue_from_alpha(self.core_movement_cost_km)

    @property
    def base_utility(self):
        """Pickup→dropoff distance (km). Alias for FATP-ANN strategy compatibility."""
        return self.core_movement_cost_km

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
