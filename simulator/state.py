from simulator.spatial_index import GridSpatialIndex


class StateManager:
    spatial_index: GridSpatialIndex
    deferred_task_index: GridSpatialIndex

    def __init__(self, all_workers=None, all_tasks=None):
        """
        Parameters
        ----------
        all_workers : list[Worker]
        all_tasks   : list[Task]
        """
        self.all_workers_map = {w.id: w for w in all_workers} if all_workers else {}
        self.all_tasks_map = {t.id: t for t in all_tasks} if all_tasks else {}
        
        # Dynamic pools - Using sets for O(1) operations
        self.available_workers = set()
        self.active_tasks = set()          # Released but unassigned
        self.deferred_tasks = set()        # Scored below threshold, waiting
        
        self.assigned_tasks = set()
        self.assigned_workers = set()
        self.completed_tasks = set()

        # Stochastic acceptance counters (O(1) per offer)
        self.offers_made = 0
        self.offers_rejected = 0
        
        # INDEX 1: Available Workers (uses start_lat/lon)
        self.spatial_index = GridSpatialIndex(lat_attr='start_lat', lon_attr='start_lon')
        
        # INDEX 2: Deferred Tasks (uses pickup_lat/lon)
        self.deferred_task_index = GridSpatialIndex(lat_attr='pickup_lat', lon_attr='pickup_lon')

    def get_worker(self, worker_id):
        return self.all_workers_map.get(worker_id)

    def get_task(self, task_id):
        return self.all_tasks_map.get(task_id)

    def release_worker(self, worker):
        self.available_workers.add(worker)
        self.spatial_index.add(worker)

    def release_task(self, task):
        self.active_tasks.add(task)

    def assign_task(self, task, worker):
        self.active_tasks.discard(task)
        self.remove_deferred_task(task)
        
        self.available_workers.discard(worker)
        self.spatial_index.remove(worker)
        
        self.assigned_tasks.add(task)
        self.assigned_workers.add(worker)

    def defer_task(self, task, current_time=None):
        """
        Defer a task to the deferred pool.
        Returns True if deferred, False if already expired.
        """
        if current_time is not None and current_time >= task.expire_time:
            return False 
        
        self.active_tasks.discard(task)
        self.deferred_tasks.add(task)
        self.deferred_task_index.add(task)
        
        task.deferral_count += 1
        return True
    
    def remove_deferred_task(self, task):
        """
        Helper to cleanly remove a task from deferred state (e.g. on expiry or assignment).
        Removes from both the deferred_tasks set and the deferred_task_index.
        Only removes if the task is in the deferred pool (new tasks assigned immediately are not).
        """
        if task in self.deferred_tasks:
            self.deferred_tasks.remove(task)
            self.deferred_task_index.remove(task)

    def complete_task(self, task, worker, current_time):
        self.assigned_tasks.discard(task)
        self.assigned_workers.discard(worker)

        task.is_completed = True
        self.completed_tasks.add(task)
        worker.record_completion(current_time, task.revenue)

        # Worker physically moves to the drop-off location
        worker.start_lat = task.dropoff_lat
        worker.start_lon = task.dropoff_lon
            
        self.available_workers.add(worker)
        self.spatial_index.add(worker)