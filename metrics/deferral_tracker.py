"""
Lightweight Deferral Tracker for task-level starvation analysis.
Maintains O(1) memory footprint via aggregate counters; no per-event history stored.
"""

class DeferralTracker:
    """
    Tracks aggregate task deferral lifecycle metrics without storing historical event logs.
    """
    
    def __init__(self):
        # O(1) Aggregate Counters
        self.total_deferrals = 0
        self.tasks_deferred_at_least_once = 0
        self.tasks_assigned_after_deferral = 0
        self.tasks_expired_after_deferral = 0   # deferred but never served
        self.total_deferral_time_sec = 0.0

        # Temporary tracking (cleaned up on assignment/expiry to prevent memory leaks)
        self.active_deferred_tasks = {}  # task_id -> first_deferral_ts
        
    def record_deferral(self, task_id: str, timestamp: float, score: float, reason: str):
        """
        Record when a task is deferred.
        Note: score and reason are accepted for API compatibility but not stored 
        to save memory during DRL training.
        """
        self.total_deferrals += 1
        
        if task_id not in self.active_deferred_tasks:
            self.active_deferred_tasks[task_id] = timestamp
            self.tasks_deferred_at_least_once += 1
            
    def record_assignment(self, task_id: str, timestamp: float, was_deferred: bool, deferral_count: int):
        """Record when a task is finally assigned (rescued from deferred queue)."""
        if task_id in self.active_deferred_tasks:
            first_def_ts = self.active_deferred_tasks.pop(task_id)
            self.tasks_assigned_after_deferral += 1
            self.total_deferral_time_sec += (timestamp - first_def_ts)

    def record_expiry(self, task_id: str):
        """Record when a deferred task expires without being assigned."""
        if task_id in self.active_deferred_tasks:
            self.active_deferred_tasks.pop(task_id)
            self.tasks_expired_after_deferral += 1

    def get_summary(self) -> dict:
        """Returns the aggregate statistics for the simulation run."""
        assigned = self.tasks_assigned_after_deferral
        expired  = self.tasks_expired_after_deferral
        deferred = self.tasks_deferred_at_least_once
        return {
            'total_deferral_events':             self.total_deferrals,
            'unique_tasks_deferred':             deferred,
            'deferred_then_rescued':             assigned,
            'deferred_then_expired':             expired,
            'deferred_still_pending':            deferred - assigned - expired,
            'avg_deferral_duration_sec':         (self.total_deferral_time_sec / assigned) if assigned > 0 else 0.0,
            'rescue_rate':                       (assigned / deferred) if deferred > 0 else 0.0,
        }