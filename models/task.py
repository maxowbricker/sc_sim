import pandas as pd
from math import fabs, cos, radians

class Task:
    def __init__(self, task_dict):
        self.id = task_dict["task_id"]
        self.pickup_lat = float(task_dict["pickup_lat"])
        self.pickup_lon = float(task_dict["pickup_lon"])
        self.dropoff_lat = float(task_dict["dropoff_lat"])
        self.dropoff_lon = float(task_dict["dropoff_lon"])
        self.release_time = pd.to_datetime(task_dict["release_time"])
        self.expire_time = pd.to_datetime(task_dict["expire_time"])
        self.assigned_worker = None
        self.assigned = False
        self.is_completed = False

        # Service-time bookkeeping
        self.finish_time = None  # pd.Timestamp when task completes
        self.start_time = None   # when service starts
        self.pickup_km = None
        self.drop_km = None
        self.deferral_count = 0  # Number of times this task was deferred
        
        # Base utility for FATP-ANN strategy (proportional to task distance)
        self.base_utility = self._calculate_base_utility()

    def _calculate_base_utility(self):
        """
        Calculate base utility for FATP-ANN strategy.
        Base utility is proportional to task distance (pickup to dropoff).
        """
        if self.pickup_lat and self.dropoff_lat:
            # Use manhattan distance
            km_per_deg = 111
            d_lat = fabs(self.pickup_lat - self.dropoff_lat) * km_per_deg
            avg_lat = (self.pickup_lat + self.dropoff_lat) / 2
            d_lon = fabs(self.pickup_lon - self.dropoff_lon) * km_per_deg * cos(radians(avg_lat))
            distance_km = d_lat + d_lon
            return distance_km
        return 1.0  # Default if coordinates are invalid
    
    def is_available(self, current_time):
        return (not self.assigned) and self.release_time <= current_time <= self.expire_time

    def assign_to_worker(self, worker):
        self.assigned_worker = worker
        self.assigned = True