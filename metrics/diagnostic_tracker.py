"""Diagnostic tracker for Experiment 008: Score Normalization and Threshold Ablation.

This module provides detailed diagnostic tracking of the composite strategy's 
assignment behavior to understand the worker idle time paradox.

Key Diagnostics:
- Score component values (raw and normalized F, S, U)
- Component dominance patterns
- Soft threshold deferral statistics
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DiagnosticTracker:
    """Tracks detailed assignment diagnostics for the composite strategy.
    
    This tracker captures per-assignment details about score components,
    which component dominates, and task deferral patterns. All data can
    be exported to DataFrames for Jupyter notebook analysis.
    """
    
    def __init__(self):
        """Initialize the diagnostic tracker."""
        self._assignment_records: List[Dict[str, Any]] = []
        self._deferral_records: List[Dict[str, Any]] = []
        self._assignment_id_counter = 0
        
    def record_assignment(
        self,
        task_id: str,
        worker_id: str,
        fairness_raw: float,
        starvation_raw: float,
        utility_raw: float,
        fairness_norm: Optional[float] = None,
        starvation_norm: Optional[float] = None,
        utility_norm: Optional[float] = None,
        fairness_weight: float = 1.0,
        starvation_weight: float = 1.0,
        utility_weight: float = 1.0,
        final_score: float = 0.0,
        was_deferred_before: bool = False,
        timestamp: Optional[pd.Timestamp] = None
    ):
        """Record an assignment with score component details.
        
        Parameters
        ----------
        task_id : str
            Task identifier
        worker_id : str
            Worker identifier
        fairness_raw : float
            Raw fairness component value
        starvation_raw : float
            Raw starvation component value
        utility_raw : float
            Raw utility component value
        fairness_norm : float, optional
            Normalized fairness component (if normalization enabled)
        starvation_norm : float, optional
            Normalized starvation component (if normalization enabled)
        utility_norm : float, optional
            Normalized utility component (if normalization enabled)
        fairness_weight : float
            Weight applied to fairness component
        starvation_weight : float
            Weight applied to starvation component
        utility_weight : float
            Weight applied to utility component
        final_score : float
            Final composite score
        was_deferred_before : bool
            Whether this task was previously deferred
        timestamp : pd.Timestamp, optional
            Timestamp of assignment
        """
        # Use normalized values if available, otherwise use raw
        f_val = fairness_norm if fairness_norm is not None else fairness_raw
        s_val = starvation_norm if starvation_norm is not None else starvation_raw
        u_val = utility_norm if utility_norm is not None else utility_raw
        
        # Calculate weighted components
        f_weighted = fairness_weight * f_val
        s_weighted = starvation_weight * s_val
        u_weighted = utility_weight * u_val
        
        # Determine dominant component
        weighted_components = {
            'fairness': f_weighted,
            'starvation': s_weighted,
            'utility': u_weighted
        }
        dominant_component = max(weighted_components, key=weighted_components.get)
        dominant_value = weighted_components[dominant_component]
        
        # Calculate dominance ratio: max / sum_of_others
        other_values_sum = sum(v for k, v in weighted_components.items() if k != dominant_component)
        dominance_ratio = dominant_value / (other_values_sum + 1e-10)  # Avoid division by zero
        
        record = {
            'assignment_id': self._assignment_id_counter,
            'task_id': task_id,
            'worker_id': worker_id,
            'timestamp': timestamp,
            
            # Raw component values
            'fairness_raw': fairness_raw,
            'starvation_raw': starvation_raw,
            'utility_raw': utility_raw,
            
            # Normalized component values (None if not normalized)
            'fairness_norm': fairness_norm,
            'starvation_norm': starvation_norm,
            'utility_norm': utility_norm,
            
            # Weights
            'fairness_weight': fairness_weight,
            'starvation_weight': starvation_weight,
            'utility_weight': utility_weight,
            
            # Weighted component values (using normalized if available)
            'fairness_weighted': f_weighted,
            'starvation_weighted': s_weighted,
            'utility_weighted': u_weighted,
            
            # Dominance metrics
            'dominant_component': dominant_component,
            'dominant_value': dominant_value,
            'dominance_ratio': dominance_ratio,
            
            # Final score and metadata
            'final_score': final_score,
            'was_deferred_before': was_deferred_before,
            'normalization_used': fairness_norm is not None
        }
        
        self._assignment_records.append(record)
        self._assignment_id_counter += 1
        
    def record_task_deferred(
        self,
        task_id: str,
        best_score: float,
        threshold: float,
        reason: str = "below_threshold",
        timestamp: Optional[pd.Timestamp] = None,
        best_worker_id: Optional[str] = None
    ):
        """Record a task deferral event.
        
        Parameters
        ----------
        task_id : str
            Task identifier
        best_score : float
            Best score achieved among candidates
        threshold : float
            Threshold value that wasn't met
        reason : str
            Reason for deferral (e.g., "below_threshold", "no_candidates")
        timestamp : pd.Timestamp, optional
            Timestamp of deferral
        best_worker_id : str, optional
            ID of worker with best score (if any)
        """
        record = {
            'task_id': task_id,
            'best_score': best_score,
            'threshold': threshold,
            'score_gap': threshold - best_score,
            'reason': reason,
            'timestamp': timestamp,
            'best_worker_id': best_worker_id
        }
        
        self._deferral_records.append(record)
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get aggregated summary statistics.
        
        Returns
        -------
        dict
            Summary statistics including component dominance patterns,
            deferral rates, and score distributions.
        """
        if not self._assignment_records:
            return {
                'total_assignments': 0,
                'total_deferrals': len(self._deferral_records),
                'message': 'No assignments recorded'
            }
            
        df = pd.DataFrame(self._assignment_records)
        
        # Component dominance counts
        dominance_counts = df['dominant_component'].value_counts().to_dict()
        dominance_pct = (df['dominant_component'].value_counts(normalize=True) * 100).to_dict()
        
        # Average dominance ratios by component
        avg_dominance_by_component = df.groupby('dominant_component')['dominance_ratio'].mean().to_dict()
        
        # Score statistics
        score_stats = {
            'mean_final_score': df['final_score'].mean(),
            'median_final_score': df['final_score'].median(),
            'std_final_score': df['final_score'].std(),
        }
        
        # Component value statistics (raw)
        component_stats_raw = {
            'fairness': {
                'mean': df['fairness_raw'].mean(),
                'std': df['fairness_raw'].std(),
                'min': df['fairness_raw'].min(),
                'max': df['fairness_raw'].max(),
            },
            'starvation': {
                'mean': df['starvation_raw'].mean(),
                'std': df['starvation_raw'].std(),
                'min': df['starvation_raw'].min(),
                'max': df['starvation_raw'].max(),
            },
            'utility': {
                'mean': df['utility_raw'].mean(),
                'std': df['utility_raw'].std(),
                'min': df['utility_raw'].min(),
                'max': df['utility_raw'].max(),
            }
        }
        
        # Normalization usage
        normalization_used = df['normalization_used'].any()
        
        # Deferral statistics
        total_deferrals = len(self._deferral_records)
        total_assignments = len(self._assignment_records)
        deferral_rate = total_deferrals / (total_deferrals + total_assignments) if (total_deferrals + total_assignments) > 0 else 0
        
        if self._deferral_records:
            deferral_df = pd.DataFrame(self._deferral_records)
            deferral_stats = {
                'mean_score_gap': deferral_df['score_gap'].mean(),
                'median_score_gap': deferral_df['score_gap'].median(),
                'mean_best_score': deferral_df['best_score'].mean(),
            }
        else:
            deferral_stats = {}
        
        return {
            'total_assignments': total_assignments,
            'total_deferrals': total_deferrals,
            'deferral_rate': deferral_rate,
            
            'dominance_counts': dominance_counts,
            'dominance_percentages': dominance_pct,
            'avg_dominance_ratio_by_component': avg_dominance_by_component,
            'overall_avg_dominance_ratio': df['dominance_ratio'].mean(),
            
            'score_statistics': score_stats,
            'component_statistics_raw': component_stats_raw,
            
            'normalization_used': normalization_used,
            'deferral_statistics': deferral_stats,
        }
        
    def to_dataframe(self, record_type: str = 'assignments') -> pd.DataFrame:
        """Export records to a pandas DataFrame.
        
        Parameters
        ----------
        record_type : str
            Type of records to export: 'assignments' or 'deferrals'
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the requested records
        """
        if record_type == 'assignments':
            if not self._assignment_records:
                return pd.DataFrame()
            return pd.DataFrame(self._assignment_records)
        elif record_type == 'deferrals':
            if not self._deferral_records:
                return pd.DataFrame()
            return pd.DataFrame(self._deferral_records)
        else:
            raise ValueError(f"Unknown record_type: {record_type}. Use 'assignments' or 'deferrals'.")
            
    def reset(self):
        """Reset all tracked data."""
        self._assignment_records.clear()
        self._deferral_records.clear()
        self._assignment_id_counter = 0

