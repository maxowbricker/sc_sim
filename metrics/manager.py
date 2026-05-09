"""
Unified Metrics Manager for Spatial Crowdsourcing Simulation.
Highly optimized to provide $O(1)$ step tracking for Deep Reinforcement Learning.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import datetime  # Built-in for faster timestamp processing

from metrics.fairness import (
    jains_fairness_index,
    utility_difference,
    FairnessMetricsTracker,
)
from metrics.tracker import MetricTracker
from metrics.deferral_tracker import DeferralTracker


class MetricsManager:
    """Central hub for all simulation metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.config = config
        
        # Configuration flags
        self.enable_deferral_tracking = config.get('enable_deferral_tracking', False)
        # Extract the diagnostics flag to silence heavy trackers during RL training
        self.enable_diagnostics = config.get('enable_diagnostics', False)
        
        # Initialize specialized trackers with the diagnostic flag
        self.fairness_tracker = FairnessMetricsTracker(enable_diagnostics=self.enable_diagnostics)
        self.metric_tracker = MetricTracker(enable_diagnostics=self.enable_diagnostics)
        
        self.deferral_tracker = DeferralTracker() if self.enable_deferral_tracking else None
        
        strategy_params = config.get('strategy_params', {})
        if self.deferral_tracker:
            strategy_params['deferral_tracker'] = self.deferral_tracker
        
        # RL State - The "Source of Truth" for the Agent
        self.current_step_stats = {
            'jfi': 1.0,
            'backlog': 0,
            'avg_wait': 0.0,
            'step_avg_assignment_delay': 0.0,
            'utility_diff': 0.0,
            'completed_in_step': 0,
            'deferred_ratio': 0.0,
            'worker_availability_ratio': 0.0,
            'task_worker_ratio': 0.0,
            'mean_worker_idle_min': 0.0,
            'cv_worker_idle': 0.0,
            'pct_deferrals_below_threshold': 0.0,
            'pct_deferrals_no_candidates': 0.0,
        }
        
        # O(1) Step accumulators (reset every step)
        self.step_completed_tasks_count = 0  # Replaced list with simple integer
        self.step_wait_times = []  
        self.step_travel_dist = 0.0  
        self.step_start_time = None  
        self.step_tasks_released = 0  
        self.step_deferrals_below_threshold = 0  # O(1) RL deferral tracking
        self.step_deferrals_no_candidates = 0    # O(1) RL deferral tracking
        self.step_total_deferrals = 0
        
        # Global accumulators (persist across steps)
        self.total_tasks_released = 0
        self._summary_minimal = {
            'service_times': [],
            'pickup_distances': [],
            'expired_tasks': [],
        }
        self._completed_tasks = 0
        self._total_travel_km = 0.0
        self._empty_km = 0.0
        self._passenger_km = 0.0
        self._total_wait_min = 0.0
        self._wait_times = []
        self._backlog_peak = 0
        self._assignment_delays = []
    
    # --- EVENT HANDLERS (Fast, O(1) updates) ---
    
    def on_task_completed(self, task, worker, current_time):
        self._completed_tasks += 1
        pickup_km = task.pickup_km or 0.0
        drop_km = task.drop_km or 0.0
        self._total_travel_km += pickup_km + drop_km
        self._empty_km += pickup_km
        self._passenger_km += drop_km
        
        if pickup_km is not None:
            self._summary_minimal['pickup_distances'].append(pickup_km)
        
        if task.start_time and task.release_time:
            wait_min = (task.start_time - task.release_time) / 60.0  
            self._total_wait_min += wait_min
            self._wait_times.append(wait_min)
            
            # Step-specific tracking
            self.step_completed_tasks_count += 1
            self.step_wait_times.append(wait_min)
        
        if drop_km is not None:
            service_min = (drop_km / 30) * 60  
            self._summary_minimal['service_times'].append(service_min)
        
        self.step_travel_dist += pickup_km + drop_km
        self.fairness_tracker.record_task_assignment(task, worker, current_time)
    
    def on_task_assigned(self, task, worker, current_time,
                        score_components: Optional[Dict] = None,
                        final_score: Optional[float] = None):
        if task.start_time and task.release_time:
            assignment_delay = current_time - task.release_time  
            self._assignment_delays.append(assignment_delay)
        
        if self.deferral_tracker:
            was_deferred = hasattr(task, 'deferral_count') and task.deferral_count > 0
            deferral_count = getattr(task, 'deferral_count', 0)
            self.deferral_tracker.record_assignment(
                task_id=str(task.id),
                timestamp=current_time,
                was_deferred=was_deferred,
                deferral_count=deferral_count
            )
    
    def on_task_released(self, task, available_workers: List, current_time):
        self.total_tasks_released += 1
        self.step_tasks_released += 1
        self.fairness_tracker.record_task_release(task, available_workers, current_time)
    
    def on_task_deferred(self, task, score: float, reason: str, current_time,
                         threshold: Optional[float] = None, best_worker_id: Optional[str] = None):
        
        # O(1) Step-level tracking for RL agent observation space
        self.step_total_deferrals += 1
        if reason == 'below_threshold':
            self.step_deferrals_below_threshold += 1
        elif reason == 'no_candidates':
            self.step_deferrals_no_candidates += 1

        if self.deferral_tracker:
            self.deferral_tracker.record_deferral(
                task_id=str(task.id),
                timestamp=current_time,
                score=score,
                reason=reason
            )
    
    # --- STEP SNAPSHOT (Expensive, runs once per step) ---
    
    def snapshot_step(self, state, current_time, step_start_time=None):
        if step_start_time:
            self.step_start_time = step_start_time
        
        for w in state.all_workers_map.values():
            if w.available:
                if w.last_state_ts is not None and w.last_state_ts < current_time:
                    time_delta = current_time - w.last_state_ts
                    if time_delta > 0:
                        w.update_idle_time(time_delta)
                        w.last_state_ts = current_time
                elif w.last_state_ts is None:
                    w.last_state_ts = current_time
        
        workers = list(state.all_workers_map.values())
        self.fairness_tracker.update_worker_stats(workers)

        # Fairness only over workers who have been online (idle time accrued) or completed work.
        # Avoids inflating n with workers not yet on shift.
        active_workers = [w for w in workers if w.total_idle_time > 0 or w.completed_tasks > 0]
        if not active_workers:
            active_workers = workers

        task_counts = [w.completed_tasks for w in active_workers]
        jfi = jains_fairness_index(task_counts)
        utility_diff = utility_difference(task_counts)
        
        active_backlog = len(state.active_tasks)
        deferred_backlog = len(state.deferred_tasks)
        total_backlog = active_backlog + deferred_backlog
        self._backlog_peak = max(self._backlog_peak, total_backlog)
        
        # If no completions this step, carry forward last measured averages (avoid fake 0.0 → RL reward exploit)
        avg_wait = (
            float(np.mean(self.step_wait_times))
            if self.step_wait_times
            else self.current_step_stats.get('avg_wait', 0.0)
        )
        recent_delays = (
            self._assignment_delays[-max(1, self.step_completed_tasks_count) :]
            if self._assignment_delays
            else []
        )
        step_avg_assignment_delay = (
            float(np.mean(recent_delays))
            if recent_delays
            else self.current_step_stats.get('step_avg_assignment_delay', 0.0)
        )
        deferred_ratio = deferred_backlog / max(1, self.total_tasks_released)
        
        available_workers = len(state.available_workers)
        total_workers = len(state.all_workers_map)
        worker_availability_ratio = available_workers / max(1, total_workers)
        
        if step_start_time and self.step_start_time:
            step_duration_sec = current_time - self.step_start_time
            assigned_workers_count = len(state.assigned_workers)
            total_active_workers = available_workers + assigned_workers_count
            
            if step_duration_sec > 0 and total_active_workers > 0:
                task_release_rate_per_min = (self.step_tasks_released / step_duration_sec) * 60
                task_worker_ratio = task_release_rate_per_min / total_active_workers
            else:
                task_worker_ratio = 0.0
        else:
            task_worker_ratio = 0.0
        
        if workers:
            idle_times_min = [w.total_idle_time / 60.0 for w in workers]
            mean_idle = float(np.mean(idle_times_min))
            std_idle = float(np.std(idle_times_min)) if len(idle_times_min) > 1 else 0.0
            cv_idle = std_idle / mean_idle if mean_idle > 0 else 0.0
        else:
            mean_idle = 0.0
            cv_idle = 0.0
        
        # Calculate deferral breakdown dynamically from O(1) step variables
        if self.step_total_deferrals > 0:
            pct_below_threshold = self.step_deferrals_below_threshold / self.step_total_deferrals
            pct_no_candidates = self.step_deferrals_no_candidates / self.step_total_deferrals
        else:
            pct_below_threshold = 0.0
            pct_no_candidates = 0.0
        
        self.current_step_stats = {
            'jfi': jfi,
            'backlog': total_backlog,
            'avg_wait': avg_wait,
            'step_avg_assignment_delay': step_avg_assignment_delay,
            'utility_diff': utility_diff,
            'completed_in_step': self.step_completed_tasks_count,
            'deferred_ratio': deferred_ratio,
            'worker_availability_ratio': worker_availability_ratio,
            'task_worker_ratio': task_worker_ratio,
            'mean_worker_idle_min': mean_idle,
            'cv_worker_idle': cv_idle,
            'pct_deferrals_below_threshold': pct_below_threshold,
            'pct_deferrals_no_candidates': pct_no_candidates,
        }
        
        self.metric_tracker.snapshot(state, current_time)
        
        # Reset step accumulators
        self.step_completed_tasks_count = 0
        self.step_wait_times = []
        self.step_travel_dist = 0.0
        self.step_tasks_released = 0
        self.step_total_deferrals = 0
        self.step_deferrals_below_threshold = 0
        self.step_deferrals_no_candidates = 0
    
    def get_recent_expirations(self, current_time, window_minutes=30) -> int:
        """Counts how many tasks expired in the trailing time window."""
        window_seconds = window_minutes * 60.0
        cutoff_time = current_time - window_seconds

        count = 0
        # Iterate backwards through the chronological list for blazing speed
        for _, exp_time in reversed(self._summary_minimal['expired_tasks']):
            if exp_time >= cutoff_time:
                count += 1
            else:
                break  # Stop searching once we hit tasks older than 30 mins
        return count

    # ------------------------------------------------------------------
    # Oracle state serialisation
    # ------------------------------------------------------------------

    def snapshot_metrics(self) -> dict:
        """Snapshot all state needed to restore the metrics manager after a
        greedy oracle run.  Copies lists/dicts by value so the snapshot is
        independent of the live object.
        """
        return {
            # Step accumulators (reset each step)
            'step_completed_tasks_count': self.step_completed_tasks_count,
            'step_wait_times': list(self.step_wait_times),
            'step_travel_dist': self.step_travel_dist,
            'step_start_time': self.step_start_time,
            'step_tasks_released': self.step_tasks_released,
            'step_deferrals_below_threshold': self.step_deferrals_below_threshold,
            'step_deferrals_no_candidates': self.step_deferrals_no_candidates,
            'step_total_deferrals': self.step_total_deferrals,
            # Current step stats dict (RL observation source)
            'current_step_stats': dict(self.current_step_stats),
            # Global accumulators (must be preserved so final results remain
            # anchored to the composite run, not the greedy oracle run)
            'total_tasks_released': self.total_tasks_released,
            '_completed_tasks': self._completed_tasks,
            '_total_travel_km': self._total_travel_km,
            '_empty_km': self._empty_km,
            '_passenger_km': self._passenger_km,
            '_total_wait_min': self._total_wait_min,
            '_wait_times': list(self._wait_times),
            '_backlog_peak': self._backlog_peak,
            '_assignment_delays': list(self._assignment_delays),
            '_summary_minimal': {
                'service_times': list(self._summary_minimal['service_times']),
                'pickup_distances': list(self._summary_minimal['pickup_distances']),
                'expired_tasks': list(self._summary_minimal['expired_tasks']),
            },
        }

    def restore_metrics(self, snap: dict) -> None:
        """Overwrite all mutable fields from a snapshot produced by
        ``snapshot_metrics``.  The fairness_tracker is *not* restored here
        because it only affects final diagnostics, not the per-step reward
        signal.  If you need full fairness-tracker fidelity, call
        ``snapshot_metrics`` before and after and handle it separately.
        """
        self.step_completed_tasks_count = snap['step_completed_tasks_count']
        self.step_wait_times = list(snap['step_wait_times'])
        self.step_travel_dist = snap['step_travel_dist']
        self.step_start_time = snap['step_start_time']
        self.step_tasks_released = snap['step_tasks_released']
        self.step_deferrals_below_threshold = snap['step_deferrals_below_threshold']
        self.step_deferrals_no_candidates = snap['step_deferrals_no_candidates']
        self.step_total_deferrals = snap['step_total_deferrals']
        self.current_step_stats = dict(snap['current_step_stats'])
        self.total_tasks_released = snap['total_tasks_released']
        self._completed_tasks = snap['_completed_tasks']
        self._total_travel_km = snap['_total_travel_km']
        self._empty_km = snap['_empty_km']
        self._passenger_km = snap['_passenger_km']
        self._total_wait_min = snap['_total_wait_min']
        self._wait_times = list(snap['_wait_times'])
        self._backlog_peak = snap['_backlog_peak']
        self._assignment_delays = list(snap['_assignment_delays'])
        self._summary_minimal = {
            'service_times': list(snap['_summary_minimal']['service_times']),
            'pickup_distances': list(snap['_summary_minimal']['pickup_distances']),
            'expired_tasks': list(snap['_summary_minimal']['expired_tasks']),
        }

    # --- RL INTERFACE ---
    
    def get_reward_stats(self, current_time) -> Dict[str, float]:
        return {
            'fairness': self.current_step_stats['jfi'],
            'latency': self.current_step_stats['avg_wait'],
            'recent_expirations': self.get_recent_expirations(current_time, window_minutes=30)
        }
    
    def get_observation_data(self, state, current_time) -> Dict[str, Any]:
        stats = self.current_step_stats
        
        # FAST TIME ENCODING
        if isinstance(current_time, (int, float)):
            dt = datetime.datetime.fromtimestamp(current_time)
            hour = dt.hour + dt.minute / 60.0
            weekday = dt.weekday()
        else:
            hour = current_time.hour + current_time.minute / 60.0
            weekday = current_time.weekday()
            
        time_sin = np.sin(2 * np.pi * hour / 24.0)
        time_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # Day Categories (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
        is_weekend = float(weekday >= 5)
        is_midweek = float(1 <= weekday <= 3)
        is_mon_fri = float(weekday == 0 or weekday == 4)

        # Assignment delay (sec): computed in snapshot_step while step counts are valid; do not re-slice here
        # (step_completed_tasks_count is reset to 0 after each snapshot).
        step_avg_delay = float(stats.get('step_avg_assignment_delay', 0.0))

        # Raw Task Arrival Rate (Tasks per minute)
        # We divide by 5.0 minutes (the standard step duration)
        task_arrival_rate = (self.step_tasks_released / 5.0) 
        
        return {
            'deferred_ratio': stats['deferred_ratio'],
            'worker_availability_ratio': stats['worker_availability_ratio'],
            'total_workers': len(state.all_workers_map),
            'jfi': stats['jfi'],
            'step_avg_wait': stats['avg_wait'],
            'step_avg_assignment_delay': step_avg_delay,
            'backlog_peak': self._backlog_peak,
            'task_arrival_rate': task_arrival_rate,
            'is_midweek': is_midweek,
            'is_mon_fri': is_mon_fri,
            'is_weekend': is_weekend,
            'time_sin': time_sin,
            'time_cos': time_cos,
        }
    
    # --- FINAL RESULTS INTERFACE ---
    
    def on_task_expired(self, task_id, current_time=0.0):
        # Store as a tuple: (task_id, expiration_timestamp)
        self._summary_minimal['expired_tasks'].append((task_id, current_time))

    @property
    def summary(self):
        return {
            **self._summary_minimal,
            'completed_tasks': self._completed_tasks,
            'total_travel_km': self._total_travel_km,
            'empty_km': self._empty_km,
            'passenger_km': self._passenger_km,
            'total_wait_min': self._total_wait_min,
            'wait_times': self._wait_times,
            'backlog_peak': self._backlog_peak,
            'assignment_delays': self._assignment_delays,
        }

    def get_final_results(self) -> Dict[str, Any]:
        fairness_summary = self.fairness_tracker.get_fairness_summary()
        
        results = {
            **self._summary_minimal,
            'completed_tasks': self._completed_tasks,
            'total_travel_km': self._total_travel_km,
            'empty_km': self._empty_km,
            'passenger_km': self._passenger_km,
            'total_wait_min': self._total_wait_min,
            'wait_times': self._wait_times,
            'backlog_peak': self._backlog_peak,
            'assignment_delays': self._assignment_delays,
            'final_jains_fairness_index': self.current_step_stats.get('jfi', 1.0),
            'final_utility_difference_tasks': self.current_step_stats.get('utility_diff', 0.0),
            
            **fairness_summary,
            'metric_tracker': self.metric_tracker,
        }
        
        if self.deferral_tracker:
            results['deferral_stats'] = self.deferral_tracker.get_summary()
        
        return results