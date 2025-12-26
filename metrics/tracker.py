"""Simple per-tick metrics collection utilities.

This first implementation focuses on the metrics used in the dissertation:

* Jain's Fairness Index (JFI) over task completions per worker.
* Utility Difference (UD) – mean absolute deviation of completions per worker.
* Average task age across *active* + *assigned* tasks.
* Raw backlog size (len(active_tasks)).

MetricTracker.snapshot(state, now) returns a dict which can later be converted
to a DataFrame for plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


class MetricTracker:
    """Collects simulation KPIs every tick and allows exporting them."""

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._worker_fairness_history: Dict[str, List[Dict]] = {}  # Per-worker EWMA fairness over time
        self._wait_time_samples: List[float] = []  # All wait times for distribution analysis
        self._travel_distance_samples: List[float] = []  # All travel distances
        self._completion_time_samples: List[float] = []  # All task completion times

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def snapshot(self, state, now: pd.Timestamp) -> Dict[str, Any]:
        """Capture a snapshot of key metrics for the current timestep.

        Parameters
        ----------
        state : simulator.state.StateManager
            The live simulation state.
        now : pd.Timestamp
            Current simulation time.
        """

        # Fairness metrics use per-worker completion counts
        all_workers = list(state.available_workers) + list(state.assigned_workers)
        completed_counts = [w.completed_tasks for w in all_workers if hasattr(w, "completed_tasks")]

        jfi = _jains_fairness(completed_counts)
        ud = _utility_difference(completed_counts)

        # Enhanced fairness distribution (EWMA idle-time proxy)  
        active_workers = list(state.available_workers) + list(state.assigned_workers)
        fairness_vals = [getattr(w, "fairness_ewma", 0.0) for w in active_workers]

        # Track individual worker EWMA fairness for time-series analysis
        for worker in active_workers:
            if hasattr(worker, 'id') and hasattr(worker, 'fairness_ewma'):
                worker_id = str(worker.id)
                if worker_id not in self._worker_fairness_history:
                    self._worker_fairness_history[worker_id] = []
                
                self._worker_fairness_history[worker_id].append({
                    'time': now,
                    'ewma_fairness': worker.fairness_ewma,
                    'completed_tasks': getattr(worker, 'completed_tasks', 0),
                    'available': getattr(worker, 'available', False)
                })

        if fairness_vals:
            fairness_mean = float(pd.Series(fairness_vals).mean())
            fairness_p90  = float(pd.Series(fairness_vals).quantile(0.9))
            fairness_max  = float(max(fairness_vals))
            fairness_min  = float(min(fairness_vals))
            fairness_std  = float(pd.Series(fairness_vals).std())
        else:
            fairness_mean = fairness_p90 = fairness_max = fairness_min = fairness_std = 0.0

        # Enhanced task age metrics
        task_pool = list(state.active_tasks) + list(state.assigned_tasks)
        ages = [(now - t.release_time) / 60.0 for t in task_pool]  # Already in seconds, convert to minutes
        avg_age = sum(ages) / len(ages) if ages else 0.0
        max_age = max(ages) if ages else 0.0

        # Wait time analysis for completed tasks in this timestep
        current_wait_times = []
        for task in list(state.completed_tasks):
            if hasattr(task, 'start_time') and task.start_time and task.release_time:
                wait_time_min = (task.start_time - task.release_time) / 60.0  # Already in seconds
                current_wait_times.append(wait_time_min)
                if wait_time_min not in self._wait_time_samples:  # Avoid duplicates
                    self._wait_time_samples.append(wait_time_min)

        # Travel distance analysis for assigned tasks
        current_travel_distances = []
        for task in list(state.assigned_tasks):
            if hasattr(task, 'pickup_km') and task.pickup_km is not None:
                current_travel_distances.append(task.pickup_km)
                if task.pickup_km not in self._travel_distance_samples:  # Avoid duplicates
                    self._travel_distance_samples.append(task.pickup_km)

        # Task completion time analysis
        current_completion_times = []
        for task in list(state.completed_tasks):
            if hasattr(task, 'finish_time') and hasattr(task, 'start_time') and task.finish_time and task.start_time:
                completion_time_min = (task.finish_time - task.start_time) / 60.0  # Already in seconds
                current_completion_times.append(completion_time_min)
                if completion_time_min not in self._completion_time_samples:  # Avoid duplicates
                    self._completion_time_samples.append(completion_time_min)

        # Calculate current averages
        avg_wait_time = sum(current_wait_times) / len(current_wait_times) if current_wait_times else 0.0
        max_wait_time = max(current_wait_times) if current_wait_times else 0.0
        min_wait_time = min(current_wait_times) if current_wait_times else 0.0
        
        avg_travel_distance = sum(current_travel_distances) / len(current_travel_distances) if current_travel_distances else 0.0
        max_travel_distance = max(current_travel_distances) if current_travel_distances else 0.0
        
        avg_completion_time = sum(current_completion_times) / len(current_completion_times) if current_completion_times else 0.0
        
        # Overall wait time statistics from all samples
        overall_avg_wait = sum(self._wait_time_samples) / len(self._wait_time_samples) if self._wait_time_samples else 0.0
        overall_max_wait = max(self._wait_time_samples) if self._wait_time_samples else 0.0
        overall_min_wait = min(self._wait_time_samples) if self._wait_time_samples else 0.0
        
        # Overall travel distance statistics
        overall_avg_travel = sum(self._travel_distance_samples) / len(self._travel_distance_samples) if self._travel_distance_samples else 0.0

        # Enhanced backlog tracking: total backlog (active + deferred)
        active_backlog = len(state.active_tasks)
        deferred_backlog = len(state.deferred_tasks)
        total_backlog = active_backlog + deferred_backlog
        
        record = {
            "time": now,
            "backlog": active_backlog,  # Legacy: keep for backward compatibility
            "backlog_active": active_backlog,  # Explicit active backlog
            "backlog_deferred": deferred_backlog,  # Deferred backlog
            "backlog_total": total_backlog,  # Total backlog (active + deferred)
            "assigned": len(state.assigned_tasks),
            "completed_total": len(state.completed_tasks),
            "jfi": jfi,
            "ud": ud,
            
            # Enhanced task age metrics
            "avg_task_age_min": avg_age,
            "max_task_age_min": max_age,
            
            # Enhanced fairness metrics
            "fairness_mean": fairness_mean,
            "fairness_p90": fairness_p90,
            "fairness_max": fairness_max,
            "fairness_min": fairness_min,
            "fairness_std": fairness_std,
            
            # Wait time metrics (current timestep)
            "current_avg_wait_min": avg_wait_time,
            "current_max_wait_min": max_wait_time,
            "current_min_wait_min": min_wait_time,
            "current_wait_samples": len(current_wait_times),
            
            # Travel distance metrics (current timestep)
            "current_avg_travel_km": avg_travel_distance,
            "current_max_travel_km": max_travel_distance,
            "current_travel_samples": len(current_travel_distances),
            
            # Task completion time metrics (current timestep)
            "current_avg_completion_min": avg_completion_time,
            "current_completion_samples": len(current_completion_times),
            
            # Overall simulation statistics
            "overall_avg_wait_min": overall_avg_wait,
            "overall_max_wait_min": overall_max_wait,
            "overall_min_wait_min": overall_min_wait,
            "overall_avg_travel_km": overall_avg_travel,
            "total_wait_samples": len(self._wait_time_samples),
            "total_travel_samples": len(self._travel_distance_samples),
            "total_completion_samples": len(self._completion_time_samples),
        }

        self._records.append(record)
        return record

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Export main time-series metrics as DataFrame."""
        return pd.DataFrame(self._records)
    
    def get_worker_fairness_dataframe(self) -> pd.DataFrame:
        """Export per-worker EWMA fairness time-series for plotting.
        
        Returns DataFrame with columns: time, worker_id, ewma_fairness, completed_tasks, available
        Perfect for creating individual worker fairness trend plots.
        """
        rows = []
        for worker_id, history in self._worker_fairness_history.items():
            for entry in history:
                rows.append({
                    'worker_id': worker_id,
                    'time': entry['time'],
                    'ewma_fairness': entry['ewma_fairness'],
                    'completed_tasks': entry['completed_tasks'],
                    'available': entry['available']
                })
        return pd.DataFrame(rows)
    
    def get_wait_time_distribution(self) -> pd.DataFrame:
        """Export all wait time samples for distribution analysis.
        
        Perfect for creating histograms like the one in your screenshot.
        """
        return pd.DataFrame({
            'wait_time_minutes': self._wait_time_samples,
            'wait_time_seconds': [w * 60 for w in self._wait_time_samples]
        })
    
    def get_travel_distance_distribution(self) -> pd.DataFrame:
        """Export all travel distance samples for analysis."""
        return pd.DataFrame({
            'travel_distance_km': self._travel_distance_samples
        })
    
    def get_completion_time_distribution(self) -> pd.DataFrame:
        """Export all task completion time samples for analysis."""
        return pd.DataFrame({
            'completion_time_minutes': self._completion_time_samples
        })
    
    def get_temporal_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for all temporal metrics."""
        main_df = self.to_dataframe()
        
        if len(main_df) == 0:
            return {}
            
        # Wait time evolution statistics
        wait_evolution = {
            'avg_wait_time_trend': main_df['overall_avg_wait_min'].tolist(),
            'max_wait_time_trend': main_df['overall_max_wait_min'].tolist(),
            'min_wait_time_trend': main_df['overall_min_wait_min'].tolist(),
            'time_points': main_df['time'].tolist(),
        }
        
        # Task age evolution
        task_age_evolution = {
            'avg_task_age_trend': main_df['avg_task_age_min'].tolist(),
            'max_task_age_trend': main_df['max_task_age_min'].tolist(),
        }
        
        # Fairness evolution  
        fairness_evolution = {
            'fairness_mean_trend': main_df['fairness_mean'].tolist(),
            'fairness_max_trend': main_df['fairness_max'].tolist(),
            'fairness_min_trend': main_df['fairness_min'].tolist(),
            'fairness_std_trend': main_df['fairness_std'].tolist(),
            'jfi_trend': main_df['jfi'].tolist(),
        }
        
        # Distribution summaries
        wait_stats = {}
        travel_stats = {}
        completion_stats = {}
        
        if self._wait_time_samples:
            wait_df = pd.Series(self._wait_time_samples)
            wait_stats = {
                'mean': float(wait_df.mean()),
                'std': float(wait_df.std()),
                'min': float(wait_df.min()),
                'max': float(wait_df.max()),
                'median': float(wait_df.median()),
                'p25': float(wait_df.quantile(0.25)),
                'p75': float(wait_df.quantile(0.75)),
                'p90': float(wait_df.quantile(0.9)),
                'p95': float(wait_df.quantile(0.95)),
                'count': len(self._wait_time_samples)
            }
            
        if self._travel_distance_samples:
            travel_df = pd.Series(self._travel_distance_samples)
            travel_stats = {
                'mean': float(travel_df.mean()),
                'std': float(travel_df.std()),
                'min': float(travel_df.min()),
                'max': float(travel_df.max()),
                'median': float(travel_df.median()),
                'p90': float(travel_df.quantile(0.9)),
                'count': len(self._travel_distance_samples)
            }
            
        if self._completion_time_samples:
            completion_df = pd.Series(self._completion_time_samples)
            completion_stats = {
                'mean': float(completion_df.mean()),
                'std': float(completion_df.std()),
                'min': float(completion_df.min()),
                'max': float(completion_df.max()),
                'median': float(completion_df.median()),
                'count': len(self._completion_time_samples)
            }
        
        return {
            'wait_time_evolution': wait_evolution,
            'task_age_evolution': task_age_evolution,
            'fairness_evolution': fairness_evolution,
            'wait_time_stats': wait_stats,
            'travel_distance_stats': travel_stats,
            'completion_time_stats': completion_stats,
            'num_workers_tracked': len(self._worker_fairness_history),
            'simulation_duration': (main_df['time'].iloc[-1] - main_df['time'].iloc[0]).total_seconds() / 3600.0 if len(main_df) > 1 else 0.0
        }

    def save_parquet(self, path: str | Path):
        """Save main time-series data as parquet."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)
        
    def save_csv(self, path: str | Path):
        """Save main time-series data as CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        
    def save_all_data(self, base_path: str | Path):
        """Save all collected data for comprehensive analysis.
        
        Creates multiple files:
        - {base_path}_metrics.csv - main time-series metrics
        - {base_path}_worker_fairness.csv - per-worker EWMA fairness trends  
        - {base_path}_wait_times.csv - wait time distribution
        - {base_path}_travel_distances.csv - travel distance distribution
        - {base_path}_completion_times.csv - completion time distribution
        - {base_path}_summary.json - temporal summary statistics
        """
        import json
        
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        
        # Main metrics
        self.to_dataframe().to_csv(f"{base}_metrics.csv", index=False)
        
        # Worker fairness time-series
        worker_df = self.get_worker_fairness_dataframe()
        if not worker_df.empty:
            worker_df.to_csv(f"{base}_worker_fairness.csv", index=False)
        
        # Distributions
        wait_df = self.get_wait_time_distribution()
        if not wait_df.empty:
            wait_df.to_csv(f"{base}_wait_times.csv", index=False)
            
        travel_df = self.get_travel_distance_distribution()  
        if not travel_df.empty:
            travel_df.to_csv(f"{base}_travel_distances.csv", index=False)
            
        completion_df = self.get_completion_time_distribution()
        if not completion_df.empty:
            completion_df.to_csv(f"{base}_completion_times.csv", index=False)
        
        # Summary statistics
        summary = self.get_temporal_summary()
        with open(f"{base}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _jains_fairness(values: List[int | float]) -> float:
    """Compute Jain's Fairness Index for a list of non-negative values."""
    if not values:
        return 1.0  # perfectly fair when no activity yet

    x = pd.Series(values, dtype="float")
    numerator = x.sum() ** 2
    denominator = len(x) * (x ** 2).sum()
    return float(numerator / denominator) if denominator else 1.0


def _utility_difference(values: List[int | float]) -> float:
    """Utility Difference = mean absolute deviation from the mean.

    Lower is better (0 means perfectly equal completions).
    """
    if not values:
        return 0.0

    x = pd.Series(values, dtype="float")
    return float((x - x.mean()).abs().mean())