"""
Deferral Tracker for Task-Level Starvation Prevention Analysis

Tracks individual task deferral events to answer RQ3.3:
"What percentage of tasks benefit from starvation prevention vs immediate assignment?"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class DeferralTracker:
    """
    Track task deferral lifecycle for detailed starvation prevention analysis.
    
    Usage:
        tracker = DeferralTracker()
        
        # When task is deferred
        tracker.record_deferral(task_id, timestamp, score, reason)
        
        # When task is assigned
        tracker.record_assignment(task_id, timestamp, was_deferred, deferral_count)
        
        # Get summary statistics
        stats = tracker.get_summary()
    """
    
    def __init__(self):
        self.deferral_events = []  # All deferral events
        self.assignment_events = []  # All assignment events
        self.task_first_deferral = {}  # task_id -> first deferral timestamp
        
    def record_deferral(self, task_id: str, timestamp: pd.Timestamp, 
                       score: float, reason: str):
        """
        Record when a task is deferred.
        
        Parameters
        ----------
        task_id : str
            Unique task identifier
        timestamp : pd.Timestamp
            When the deferral occurred
        score : float
            Composite score at time of deferral
        reason : str
            Why deferred ('below_threshold' or 'no_candidates')
        """
        self.deferral_events.append({
            'task_id': task_id,
            'timestamp': timestamp,
            'score': score,
            'reason': reason
        })
        
        # Track first deferral for duration calculation
        if task_id not in self.task_first_deferral:
            self.task_first_deferral[task_id] = timestamp
    
    def record_assignment(self, task_id: str, timestamp: pd.Timestamp,
                         was_deferred: bool, deferral_count: int):
        """
        Record when a task is assigned to a worker.
        
        Parameters
        ----------
        task_id : str
            Unique task identifier
        timestamp : pd.Timestamp
            When the assignment occurred
        was_deferred : bool
            Whether this task was previously deferred
        deferral_count : int
            Number of times this task was deferred
        """
        # Calculate deferral duration if task was deferred
        deferral_duration_sec = None
        if was_deferred and task_id in self.task_first_deferral:
            first_deferral = self.task_first_deferral[task_id]
            deferral_duration_sec = (timestamp - first_deferral).total_seconds()
        
        self.assignment_events.append({
            'task_id': task_id,
            'timestamp': timestamp,
            'was_deferred': was_deferred,
            'deferral_count': deferral_count,
            'deferral_duration_sec': deferral_duration_sec
        })
    
    def get_summary(self) -> Dict:
        """
        Calculate comprehensive deferral statistics.
        
        Returns
        -------
        dict
            Statistics including:
            - immediate_assignments: Tasks assigned without deferral
            - deferred_assignments: Tasks assigned after deferral
            - pct_benefiting_from_starvation: % of tasks that were deferred then assigned
            - deferral_duration statistics (mean, median, P95, max)
            - deferral_count statistics
        """
        if not self.assignment_events:
            return {
                'immediate_assignments': 0,
                'deferred_assignments': 0,
                'pct_benefiting_from_starvation': 0.0,
                'mean_deferral_duration_sec': 0.0,
                'median_deferral_duration_sec': 0.0,
                'p95_deferral_duration_sec': 0.0,
                'max_deferral_duration_sec': 0.0,
                'mean_deferral_count': 0.0,
                'max_deferral_count': 0,
                'total_deferral_events': 0,
            }
        
        # Categorize assignments
        immediate = [e for e in self.assignment_events if not e['was_deferred']]
        deferred = [e for e in self.assignment_events if e['was_deferred']]
        
        immediate_count = len(immediate)
        deferred_count = len(deferred)
        total_assignments = len(self.assignment_events)
        
        # Calculate deferral durations
        deferral_durations = [e['deferral_duration_sec'] for e in deferred 
                             if e['deferral_duration_sec'] is not None]
        
        # Calculate deferral counts
        deferral_counts = [e['deferral_count'] for e in deferred]
        
        return {
            # Assignment breakdown
            'immediate_assignments': immediate_count,
            'deferred_assignments': deferred_count,
            'total_assignments': total_assignments,
            'pct_benefiting_from_starvation': 100.0 * deferred_count / total_assignments if total_assignments > 0 else 0.0,
            
            # Deferral duration statistics (in seconds)
            'mean_deferral_duration_sec': float(np.mean(deferral_durations)) if deferral_durations else 0.0,
            'median_deferral_duration_sec': float(np.median(deferral_durations)) if deferral_durations else 0.0,
            'std_deferral_duration_sec': float(np.std(deferral_durations)) if deferral_durations else 0.0,
            'p95_deferral_duration_sec': float(np.percentile(deferral_durations, 95)) if deferral_durations else 0.0,
            'max_deferral_duration_sec': float(max(deferral_durations)) if deferral_durations else 0.0,
            
            # Deferral count statistics
            'mean_deferral_count': float(np.mean(deferral_counts)) if deferral_counts else 0.0,
            'max_deferral_count': int(max(deferral_counts)) if deferral_counts else 0,
            
            # Total events
            'total_deferral_events': len(self.deferral_events),
            'unique_deferred_tasks': len(set(e['task_id'] for e in self.deferral_events)),
        }
    
    def get_deferral_reason_breakdown(self) -> Dict:
        """Get breakdown of why tasks were deferred."""
        reasons = {}
        for event in self.deferral_events:
            reason = event['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        
        total = len(self.deferral_events)
        return {
            f'pct_{reason}': 100.0 * count / total if total > 0 else 0.0
            for reason, count in reasons.items()
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export all events to a DataFrame for detailed analysis.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all deferral and assignment events
        """
        all_events = []
        
        # Add deferral events
        for event in self.deferral_events:
            all_events.append({
                'task_id': event['task_id'],
                'event_type': 'deferred',
                'timestamp': event['timestamp'],
                'score': event['score'],
                'reason': event['reason']
            })
        
        # Add assignment events
        for event in self.assignment_events:
            all_events.append({
                'task_id': event['task_id'],
                'event_type': 'assigned',
                'timestamp': event['timestamp'],
                'was_deferred': event['was_deferred'],
                'deferral_count': event['deferral_count'],
                'deferral_duration_sec': event['deferral_duration_sec']
            })
        
        return pd.DataFrame(all_events).sort_values('timestamp')



