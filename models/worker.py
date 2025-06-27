from datetime import datetime
import pandas as pd

class Worker:
    def __init__(self, worker_dict):
        self.id = worker_dict["worker_id"]
        self.start_lat = float(worker_dict["start_lat"])
        self.start_lon = float(worker_dict["start_lon"])
        self.release_time = pd.to_datetime(worker_dict["release_time"])
        self.deadline = pd.to_datetime(worker_dict["deadline"])
        self.assigned_task = None
        self.available = True

    def assign_task(self, task):
        self.assigned_task = task
        self.available = False

    def is_available(self, current_time):
        return self.available and self.release_time <= current_time <= self.deadline