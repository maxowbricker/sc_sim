import pandas as pd

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

        # Service-time bookkeeping
        self.finish_time = None  # pd.Timestamp when task completes
        self.start_time = None   # when service starts
        self.pickup_km = None
        self.drop_km = None

    def is_available(self, current_time):
        return (not self.assigned) and self.release_time <= current_time <= self.expire_time

    def assign_to_worker(self, worker):
        self.assigned_worker = worker
        self.assigned = True