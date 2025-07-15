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
        completed_counts = [w.completed_tasks for w in (
            state.available_workers + state.assigned_workers + state.completed_tasks  # type: ignore[arg-type]
        ) if hasattr(w, "completed_tasks")]

        jfi = _jains_fairness(completed_counts)
        ud = _utility_difference(completed_counts)

        # Fairness distribution (EWMA idle-time proxy)
        fairness_vals = [getattr(w, "fairness_ewma", 0.0) for w in (
            state.available_workers + state.assigned_workers  # released workers
        )]

        if fairness_vals:
            fairness_mean = float(pd.Series(fairness_vals).mean())
            fairness_p90  = float(pd.Series(fairness_vals).quantile(0.9))
            fairness_max  = float(max(fairness_vals))
        else:
            fairness_mean = fairness_p90 = fairness_max = 0.0

        # Task age metrics over active + assigned tasks
        task_pool = state.active_tasks + state.assigned_tasks
        ages = [(now - t.release_time).total_seconds() / 60.0 for t in task_pool]
        avg_age = sum(ages) / len(ages) if ages else 0.0

        record = {
            "time": now,
            "backlog": len(state.active_tasks),
            "assigned": len(state.assigned_tasks),
            "completed_total": len(state.completed_tasks),
            "jfi": jfi,
            "ud": ud,
            "avg_task_age_min": avg_age,
            "fairness_mean": fairness_mean,
            "fairness_p90": fairness_p90,
            "fairness_max": fairness_max,
        }

        self._records.append(record)
        return record

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def save_parquet(self, path: str | Path):
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    def save_csv(self, path: str | Path):
        df = self.to_dataframe()
        df.to_csv(path, index=False)


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