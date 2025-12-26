import pandas as pd
from simulator.spatial_index import fast_manhattan_km

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
        
        # Base utility for FATP-ANN strategy (proportional to task distance)
        self.base_utility = self._calculate_base_utility()

    def _calculate_base_utility(self):
        """
        Calculate base utility for FATP-ANN strategy.
        Base utility is proportional to task distance (pickup to dropoff).
        
        OPTIMIZED: Uses fast_manhattan_km (Flat Earth) for consistency and speed.
        Requires set_city_constants() to be called before Task objects are created.
        """
        if self.pickup_lat and self.dropoff_lat:
            # Use the global optimized function (no trig calls)
            return fast_manhattan_km(
                self.pickup_lat, self.pickup_lon,
                self.dropoff_lat, self.dropoff_lon
            )
        return 1.0  # Default if coordinates are invalid
    
    def is_available(self, current_time):
        return (not self.assigned) and self.release_time <= current_time <= self.expire_time

    def assign_to_worker(self, worker):
        self.assigned_worker = worker
        self.assigned = True