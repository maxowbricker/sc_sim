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
        worker.record_completion(current_time)

        # Worker physically moves to the drop-off location
        worker.start_lat = task.dropoff_lat
        worker.start_lon = task.dropoff_lon
            
        self.available_workers.add(worker)
        self.spatial_index.add(worker)

    # ------------------------------------------------------------------
    # Oracle state serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Snapshot all mutable StateManager fields as primitive IDs.

        Object references (Worker, Task) are stored as their IDs so the
        snapshot is cheap and independent of the live objects.  Restore
        reconstructs references from the same all_workers_map/all_tasks_map
        dictionaries, which are never mutated during a run.
        """
        return {
            'available_worker_ids': {w.id for w in self.available_workers},
            'active_task_ids': {t.id for t in self.active_tasks},
            'deferred_task_ids': {t.id for t in self.deferred_tasks},
            'assigned_task_ids': {t.id for t in self.assigned_tasks},
            'assigned_worker_ids': {w.id for w in self.assigned_workers},
            'completed_task_ids': {t.id for t in self.completed_tasks},
            # Per-worker primitive state
            'workers': {w_id: w.get_state_dict() for w_id, w in self.all_workers_map.items()},
            # Per-task primitive state
            'tasks': {t_id: t.get_state_dict() for t_id, t in self.all_tasks_map.items()},
        }

    def restore(self, snap: dict) -> None:
        """Overwrite mutable state from a snapshot produced by ``snapshot``.

        Rebuilds both spatial indices from scratch (fast — grid insert is O(1)
        per item) rather than trying to serialise the index structures.
        """
        wmap = self.all_workers_map
        tmap = self.all_tasks_map

        # 1. Restore primitive fields on every Worker and Task
        for w_id, wstate in snap['workers'].items():
            wmap[w_id].restore_from_dict(wstate)
        for t_id, tstate in snap['tasks'].items():
            tmap[t_id].restore_from_dict(tstate)

        # 2. Restore assigned_task / assigned_worker cross-references
        for w_id, wstate in snap['workers'].items():
            at_id = wstate['assigned_task_id']
            wmap[w_id].assigned_task = tmap[at_id] if at_id is not None else None
        for t_id, tstate in snap['tasks'].items():
            aw_id = tstate['assigned_worker_id']
            tmap[t_id].assigned_worker = wmap[aw_id] if aw_id is not None else None

        # 3. Rebuild the dynamic pool sets from snapshotted ID sets
        self.available_workers = {wmap[i] for i in snap['available_worker_ids']}
        self.active_tasks      = {tmap[i] for i in snap['active_task_ids']}
        self.deferred_tasks    = {tmap[i] for i in snap['deferred_task_ids']}
        self.assigned_tasks    = {tmap[i] for i in snap['assigned_task_ids']}
        self.assigned_workers  = {wmap[i] for i in snap['assigned_worker_ids']}
        self.completed_tasks   = {tmap[i] for i in snap['completed_task_ids']}

        # 4. Rebuild spatial indices from restored object state
        self.spatial_index = GridSpatialIndex(lat_attr='start_lat', lon_attr='start_lon')
        for w in self.available_workers:
            self.spatial_index.add(w)

        self.deferred_task_index = GridSpatialIndex(lat_attr='pickup_lat', lon_attr='pickup_lon')
        for t in self.deferred_tasks:
            self.deferred_task_index.add(t)