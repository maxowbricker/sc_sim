"""
Unified Metrics Manager for Spatial Crowdsourcing Simulation.

This module provides a single source of truth for all metrics calculations,
replacing the fragmented logic across simulation.py, tracker.py, and gym_environment.py.

The MetricsManager:
- Orchestrates specialized trackers (deferral, fairness)
- Maintains current step statistics for RL agent
- Provides clean interfaces for simulation events and RL observations
- Eliminates duplicate metric calculations
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from metrics.fairness import (
    jains_fairness_index,
    utility_difference,
    FairnessMetricsTracker
)
from metrics.tracker import MetricTracker
from metrics.deferral_tracker import DeferralTracker


class MetricsManager:
    """
    Central hub for all simulation metrics.
    
    This class replaces the fragmented metric logic in:
    - simulator/simulation.py (self.summary dict)
    - metrics/tracker.py (MetricTracker)
    - rl/gym_environment.py (manual JFI calculations)
    
    Usage:
        manager = MetricsManager(config)
        
        # During simulation events
        manager.on_task_completed(task, worker, current_time)
        manager.on_task_assigned(task, worker, score_components, current_time)
        manager.on_task_deferred(task, score, reason, current_time)
        
        # At end of step (for RL)
        manager.snapshot_step(state, current_time)
        
        # For RL agent
        stats = manager.get_reward_stats()
        obs_data = manager.get_observation_data(state, current_time)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metrics manager.
        
        Args:
            config: Configuration dict with flags like:
                - enable_deferral_tracking: Enable deferral tracking
        """
        config = config or {}
        self.config = config
        
        # Configuration flags
        self.enable_deferral_tracking = config.get('enable_deferral_tracking', False)
        
        # Initialize specialized trackers
        self.fairness_tracker = FairnessMetricsTracker()
        self.metric_tracker = MetricTracker()
        
        # Optional specialized trackers
        self.deferral_tracker = DeferralTracker() if self.enable_deferral_tracking else None
        
        # Pass trackers to strategy params if needed (strategy_params is passed in config)
        strategy_params = config.get('strategy_params', {})
        if self.deferral_tracker:
            strategy_params['deferral_tracker'] = self.deferral_tracker
        
        # RL State - The "Source of Truth" for the Agent
        # These are updated once per step in snapshot_step()
        self.current_step_stats = {
            'jfi': 1.0,  # Jain's Fairness Index
            'backlog': 0,  # Total backlog (active + deferred)
            'avg_wait': 0.0,  # Average wait time in this step
            'utility_diff': 0.0,  # Utility difference
            'completed_in_step': 0,  # Tasks completed in this step
            'deferred_ratio': 0.0,  # Deferred tasks / total tasks released
            'worker_availability_ratio': 0.0,  # Available workers / total workers
            'task_worker_ratio': 0.0,  # Task release rate per worker
            'mean_worker_idle_min': 0.0,  # Mean worker idle time
            'cv_worker_idle': 0.0,  # Coefficient of variation of idle time
            'pct_deferrals_below_threshold': 0.0,  # % deferrals due to low score
            'pct_deferrals_no_candidates': 0.0,  # % deferrals due to no candidates
        }
        
        # Step accumulators (reset every step)
        self.step_completed_tasks = []  # Tasks completed in current step
        self.step_wait_times = []  # Wait times for tasks completed in step
        self.step_travel_dist = 0.0  # Total travel distance in step
        self.step_start_time = None  # When current step started
        self.step_tasks_released = 0  # Tasks released in current step
        
        # Global accumulators (persist across steps)
        self.total_tasks_released = 0  # Total tasks released (for deferred ratio)
        # Minimal summary: only what is not already covered by event handlers / snapshot
        self._summary_minimal = {
            'service_times': [],
            'pickup_distances': [],
            'expired_tasks': [],
        }
        # Accumulators updated by on_task_completed / on_task_assigned / snapshot_step (single source of truth)
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
        """
        Called when a task is completed.
        
        Args:
            task: Completed task object
            worker: Worker who completed the task
            current_time: Current simulation time (pd.Timestamp or float Unix timestamp)
        """
        # Update accumulators (single source of truth; no duplicate summary)
        self._completed_tasks += 1
        pickup_km = task.pickup_km or 0.0
        drop_km = task.drop_km or 0.0
        self._total_travel_km += pickup_km + drop_km
        self._empty_km += pickup_km
        self._passenger_km += drop_km
        
        if pickup_km is not None:
            self._summary_minimal['pickup_distances'].append(pickup_km)
        
        # Calculate wait time
        if task.start_time and task.release_time:
            wait_min = (task.start_time - task.release_time) / 60.0  # Already in seconds
            self._total_wait_min += wait_min
            self._wait_times.append(wait_min)
            
            # Step-specific tracking
            self.step_completed_tasks.append(task)
            self.step_wait_times.append(wait_min)
        
        # Service time
        if drop_km is not None:
            service_min = (drop_km / 30) * 60  # Assuming 30 km/h average speed
            self._summary_minimal['service_times'].append(service_min)
        
        # Update step travel distance
        self.step_travel_dist += pickup_km + drop_km
        
        # Update fairness tracker
        self.fairness_tracker.record_task_assignment(task, worker, current_time)
    
    def on_task_assigned(self, task, worker, current_time,
                        score_components: Optional[Dict] = None,
                        final_score: Optional[float] = None):
        """
        Called when a task is assigned to a worker.
        
        Args:
            task: Assigned task object
            worker: Worker assigned to the task
            current_time: Current simulation time (pd.Timestamp or float Unix timestamp)
            score_components: Optional dict with 'fairness', 'starvation', 'utility' components
            final_score: Optional final composite score
        """
        # Update assignment delay accumulator
        if task.start_time and task.release_time:
            assignment_delay = current_time - task.release_time  # Already in seconds
            self._assignment_delays.append(assignment_delay)
        
        # Update deferral tracker if enabled
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
        """
        Called when a task is released into the system.
        
        Args:
            task: Released task object
            available_workers: List of available workers at release time
            current_time: Current simulation time
        """
        self.total_tasks_released += 1
        self.step_tasks_released += 1
        
        # Update fairness tracker for eligibility tracking
        self.fairness_tracker.record_task_release(task, available_workers, current_time)
    
    def on_task_deferred(self, task, score: float, reason: str, current_time,
                         threshold: Optional[float] = None, best_worker_id: Optional[str] = None):
        """
        Called when a task is deferred.
        
        Args:
            task: Deferred task object
            score: Best score achieved (or 0.0 if no candidates)
            reason: Reason for deferral ('below_threshold' or 'no_candidates')
            current_time: Current simulation time
            threshold: Optional threshold value
            best_worker_id: Optional ID of worker with best score
        """
        if self.deferral_tracker:
            self.deferral_tracker.record_deferral(
                task_id=str(task.id),
                timestamp=current_time,
                score=score,
                reason=reason
            )
    
    # --- STEP SNAPSHOT (Expensive, runs once per step) ---
    
    def snapshot_step(self, state, current_time, step_start_time=None):
        """
        Calculate expensive metrics at the end of a step.
        This updates the data the RL agent will see.
        
        Args:
            state: StateManager instance with current simulation state
            current_time: Current simulation time (pd.Timestamp or float Unix timestamp)
            step_start_time: Optional start time of the step (for calculating step duration)
        """
        # Update step start time if provided
        if step_start_time:
            self.step_start_time = step_start_time
        
        # Sync all workers' idle time stats before reporting
        # Needed for accurate total_idle_time reporting in final metrics.
        for w in state.all_workers_map.values():
            if w.available:
                # Calculate pending idle time since last sync
                if w.last_state_ts is not None and w.last_state_ts < current_time:
                    time_delta = current_time - w.last_state_ts  # Already in seconds
                    if time_delta > 0:
                        w.update_idle_time(time_delta)
                        w.last_state_ts = current_time
                elif w.last_state_ts is None:
                    # Initialize last_state_ts if not set
                    w.last_state_ts = current_time
        
        # Update worker stats for fairness calculation
        self.fairness_tracker.update_worker_stats(state.all_workers_map.values())
        self.fairness_tracker.record_snapshot(current_time)
        
        # Calculate fairness metrics using the fairness tracker
        fairness_metrics = self.fairness_tracker.calculate_current_fairness()
        jfi = fairness_metrics.get('jains_fairness_index_tasks', 1.0)
        utility_diff = fairness_metrics.get('utility_difference_tasks', 0.0)
        
        # Calculate backlog
        active_backlog = len(state.active_tasks)
        deferred_backlog = len(state.deferred_tasks)
        total_backlog = active_backlog + deferred_backlog
        
        # Update peak backlog accumulator
        self._backlog_peak = max(self._backlog_peak, total_backlog)
        
        # Step average wait time (for current_step_stats)
        avg_wait = np.mean(self.step_wait_times) if self.step_wait_times else 0.0
        
        # Calculate deferred ratio
        deferred_ratio = deferred_backlog / max(1, self.total_tasks_released)
        
        # Calculate worker availability ratio
        available_workers = len(state.available_workers)
        total_workers = len(state.all_workers_map)
        worker_availability_ratio = available_workers / max(1, total_workers)
        
        # Calculate task-worker ratio (tasks per minute per worker)
        if step_start_time and self.step_start_time:
            step_duration_sec = current_time - self.step_start_time  # Already in seconds
            assigned_workers_count = len(state.assigned_workers)
            total_active_workers = available_workers + assigned_workers_count
            
            if step_duration_sec > 0 and total_active_workers > 0:
                task_release_rate_per_min = (self.step_tasks_released / step_duration_sec) * 60
                task_worker_ratio = task_release_rate_per_min / total_active_workers
            else:
                task_worker_ratio = 0.0
        else:
            task_worker_ratio = 0.0
        
        # Calculate worker idle time statistics
        workers = list(state.all_workers_map.values())
        if workers:
            idle_times_min = [w.total_idle_time / 60.0 for w in workers]  # Convert seconds to minutes
            mean_idle = float(np.mean(idle_times_min))
            std_idle = float(np.std(idle_times_min)) if len(idle_times_min) > 1 else 0.0
            cv_idle = std_idle / mean_idle if mean_idle > 0 else 0.0
        else:
            mean_idle = 0.0
            cv_idle = 0.0
        
        # Get deferral reason breakdown
        if self.deferral_tracker:
            reason_breakdown = self.deferral_tracker.get_deferral_reason_breakdown()
            pct_below_threshold = reason_breakdown.get('pct_below_threshold', 0.0) / 100.0
            pct_no_candidates = reason_breakdown.get('pct_no_candidates', 0.0) / 100.0
        else:
            pct_below_threshold = 0.0
            pct_no_candidates = 0.0
        
        # Update the "Source of Truth" for RL
        self.current_step_stats = {
            'jfi': jfi,
            'backlog': total_backlog,
            'avg_wait': avg_wait,
            'utility_diff': utility_diff,
            'completed_in_step': len(self.step_completed_tasks),
            'deferred_ratio': deferred_ratio,
            'worker_availability_ratio': worker_availability_ratio,
            'task_worker_ratio': task_worker_ratio,
            'mean_worker_idle_min': mean_idle,
            'cv_worker_idle': cv_idle,
            'pct_deferrals_below_threshold': pct_below_threshold,
            'pct_deferrals_no_candidates': pct_no_candidates,
        }
        
        # Update metric tracker snapshot (for historical tracking)
        self.metric_tracker.snapshot(state, current_time)
        
        # Reset step accumulators
        self.step_completed_tasks = []
        self.step_wait_times = []
        self.step_travel_dist = 0.0
        self.step_tasks_released = 0
    
    # --- RL INTERFACE ---
    
    def get_reward_stats(self) -> Dict[str, float]:
        """
        Return the clean stats for the RL environment reward calculation.
        
        Returns:
            Dict with 'fairness' (JFI), 'throughput' (negative backlog), 'latency' (negative wait)
        """
        return {
            'fairness': self.current_step_stats['jfi'],
            'throughput': -self.current_step_stats['backlog'],
            'latency': -self.current_step_stats['avg_wait']
        }
    
    def get_observation_data(self, state, current_time) -> Dict[str, Any]:
        """
        Get all data needed for RL observation space.
        
        This provides a clean interface for gym_environment.py to get state data
        without recalculating metrics.
        
        Args:
            state: StateManager instance
            current_time: Current simulation time (pd.Timestamp or float Unix timestamp)
            
        Returns:
            Dict with all observation features
        """
        # Get current step stats
        stats = self.current_step_stats
        
        # Time encoding - handle both float and pd.Timestamp
        # OPTIMIZATION: Extract hour from float Unix timestamp
        if isinstance(current_time, (int, float)):
            # Convert Unix timestamp to datetime for hour extraction
            dt = pd.Timestamp.fromtimestamp(current_time)
            hour = dt.hour + dt.minute / 60.0
        else:
            # Fallback for pd.Timestamp (backward compatibility)
            hour = current_time.hour + current_time.minute / 60.0
        time_sin = np.sin(2 * np.pi * hour / 24.0)
        time_cos = np.cos(2 * np.pi * hour / 24.0)
        
        return {
            'deferred_tasks': len(state.deferred_tasks),
            'total_tasks_released': self.total_tasks_released,
            'available_workers': len(state.available_workers),
            'total_workers': len(state.all_workers_map),
            'workers': list(state.all_workers_map.values()),
            'backlog_peak': self._backlog_peak,
            'current_time': current_time,
            'step_avg_wait': stats['avg_wait'],
            'time_sin': time_sin,
            'time_cos': time_cos,
            # All the step stats
            **stats
        }
    
    # --- FINAL RESULTS INTERFACE ---
    
    def on_task_expired(self, task_id):
        """Record an expired task (called from simulation TASK_EXPIRE)."""
        self._summary_minimal['expired_tasks'].append(task_id)

    @property
    def summary(self):
        """Read-only view merging minimal summary and accumulators (backward compat)."""
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
        """
        Get final simulation results for analysis.
        
        Returns:
            Dict with all final metrics, compatible with existing code
        """
        # Get fairness summary
        fairness_summary = self.fairness_tracker.get_fairness_summary()
        
        # Single source of truth: minimal summary + accumulators
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
            **fairness_summary,
            'metric_tracker': self.metric_tracker,
        }
        
        # Add optional tracker summaries
        if self.deferral_tracker:
            results['deferral_stats'] = self.deferral_tracker.get_summary()
        
        return results

