"""
Lightweight per-tick metrics collection utilities.
Optimized for DRL throughput by disabling heavy historical tracking by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# SINGLE SOURCE OF TRUTH: Import fast C-optimized math from fairness module
from metrics.fairness import jains_fairness_index, utility_difference


class MetricTracker:
    """Collects simulation KPIs every tick and allows exporting them."""

    def __init__(self, enable_diagnostics: bool = False):
        self.enable_diagnostics = enable_diagnostics
        
        # Historical arrays (Only populated if diagnostics are enabled to save RAM)
        self._records: List[Dict[str, Any]] = [] if enable_diagnostics else None
        self._worker_fairness_history: Dict[str, List[Dict]] = {} if enable_diagnostics else None
        
        # Aggregate samples (Safe to store: max size = total tasks, ~1-2MB RAM)
        self._wait_time_samples: List[float] = []  
        self._travel_distance_samples: List[float] = []  
        self._completion_time_samples: List[float] = []  

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def snapshot(self, state, now: float) -> Dict[str, Any]:
        """
        Capture a snapshot of key metrics for the current timestep.
        Returns the data for RL observation. Only saves history if diagnostics=True.
        """
        workers = list(state.all_workers_map.values())
        task_counts = [w.completed_tasks for w in workers]
        
        # FAST MATH: Uses numpy under the hood via fairness.py
        jfi = jains_fairness_index(task_counts)
        ud = utility_difference(task_counts)
        
        active_workers = _get_active_worker_count(workers, now)
        backlog = len(state.active_tasks) + len(state.deferred_tasks)
        avg_wait = _get_avg_wait(state.active_tasks, state.deferred_tasks, now)
        
        ewma_values = [w.fairness_ewma for w in workers]
        ewma_mean = float(np.mean(ewma_values)) if ewma_values else 0.0

        record = {
            "time": now,
            "jfi": jfi,
            "ud": ud,
            "backlog": backlog,
            "avg_wait_sec": avg_wait,
            "ewma_fairness_mean": ewma_mean,
            "active_workers": active_workers,
            "unassigned_ratio": backlog / len(state.all_tasks_map) if state.all_tasks_map else 0.0
        }

        # HEAVY TRACKING: Lock behind diagnostics flag
        if self.enable_diagnostics:
            self._records.append(record)
            for w in workers:
                if w.id not in self._worker_fairness_history:
                    self._worker_fairness_history[w.id] = []
                self._worker_fairness_history[w.id].append({
                    "time": now, 
                    "ewma": w.fairness_ewma,
                    "completed": w.completed_tasks
                })

        return record

    def record_task_completion(self, wait_time: float, travel_distance: float, completion_time: float):
        """O(1) appending for end-of-simulation aggregate statistics."""
        self._wait_time_samples.append(wait_time)
        self._travel_distance_samples.append(travel_distance)
        self._completion_time_samples.append(completion_time)

    # ------------------------------------------------------------------ #
    # Export API (Only used at end of simulation)
    # ------------------------------------------------------------------ #

    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.enable_diagnostics:
            print("Warning: Diagnostics disabled. No historical records to export.")
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def export_worker_fairness_history(self) -> Dict[str, pd.DataFrame]:
        if not self.enable_diagnostics:
            return {}
        return {w_id: pd.DataFrame(history) for w_id, history in self._worker_fairness_history.items()}

    def get_wait_time_distribution(self) -> pd.DataFrame:
        return pd.DataFrame({"wait_time_sec": self._wait_time_samples})

    def get_travel_distance_distribution(self) -> pd.DataFrame:
        return pd.DataFrame({"travel_distance_km": self._travel_distance_samples})

    def get_completion_time_distribution(self) -> pd.DataFrame:
        return pd.DataFrame({"completion_time_sec": self._completion_time_samples})

    def get_temporal_summary(self) -> Dict[str, float]:
        """Aggregate summary of the collected samples."""
        def safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
        
        return {
            "avg_wait_time_sec": safe_mean(self._wait_time_samples),
            "max_wait_time_sec": float(np.max(self._wait_time_samples)) if self._wait_time_samples else 0.0,
            "avg_travel_distance_km": safe_mean(self._travel_distance_samples),
            "avg_completion_time_sec": safe_mean(self._completion_time_samples),
            "total_completed": len(self._completion_time_samples)
        }

    def save_all_metrics(self, output_dir: str, prefix: str = "sim"):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        base = out_path / prefix
        
        if self.enable_diagnostics:
            df = self.export_to_dataframe()
            if not df.empty:
                df.to_csv(f"{base}_temporal_metrics.csv", index=False)
            
        wait_df = self.get_wait_time_distribution()
        if not wait_df.empty:
            wait_df.to_csv(f"{base}_wait_times.csv", index=False)
            
        travel_df = self.get_travel_distance_distribution()  
        if not travel_df.empty:
            travel_df.to_csv(f"{base}_travel_distances.csv", index=False)
            
        completion_df = self.get_completion_time_distribution()
        if not completion_df.empty:
            completion_df.to_csv(f"{base}_completion_times.csv", index=False)
        
        with open(f"{base}_summary.json", 'w') as f:
            json.dump(self.get_temporal_summary(), f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _get_avg_wait(active_tasks: set, deferred_tasks: set, now: float) -> float:
    """O(N) iteration, using pure python math. Fast enough for snapshots."""
    total_wait = sum(now - t.release_time for t in active_tasks) + \
                 sum(now - t.release_time for t in deferred_tasks)
    count = len(active_tasks) + len(deferred_tasks)
    return total_wait / count if count > 0 else 0.0

def _get_active_worker_count(workers: list, now: float) -> int:
    # A worker is active if their shift has started and hasn't ended
    return sum(1 for w in workers if w.release_time <= now <= w.deadline)