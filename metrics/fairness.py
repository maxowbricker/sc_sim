"""
Fairness metrics for spatial crowdsourcing evaluation.

Implements the metrics described in the research methodology:
- Jain's Fairness Index (JFI)
- Utility Difference (UD)
- Fairness Loss (FL)
- EWMA-based fairness metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from models.worker import Worker


def jains_fairness_index(task_counts: List[int]) -> float:
    """
    Calculate Jain's Fairness Index.
    
    JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    where x_i is the number of tasks assigned to worker i
    
    Returns value between 0 (completely unfair) and 1 (perfectly fair)
    """
    if not task_counts or len(task_counts) == 0:
        return 1.0
    
    x = np.array(task_counts, dtype=float)
    n = len(x)
    
    if np.sum(x) == 0:
        return 1.0  # No tasks assigned to anyone is "fair"
    
    numerator = (np.sum(x)) ** 2
    denominator = n * np.sum(x ** 2)
    
    return numerator / denominator if denominator > 0 else 1.0


def utility_difference(worker_utilities: List[float]) -> float:
    """
    Calculate Utility Difference - range between max and min worker utilities.
    Lower values indicate better fairness.
    """
    if not worker_utilities or len(worker_utilities) <= 1:
        return 0.0
    
    utilities = np.array(worker_utilities)
    return np.max(utilities) - np.min(utilities)


def fairness_loss(actual_assignments: List[int], ideal_assignments: List[int]) -> float:
    """
    Calculate Fairness Loss - deviation from ideal fair assignment.
    
    FL = sum(|actual_i - ideal_i|) / sum(ideal_i)
    """
    if not actual_assignments or not ideal_assignments:
        return 0.0
    
    actual = np.array(actual_assignments, dtype=float)
    ideal = np.array(ideal_assignments, dtype=float)
    
    if len(actual) != len(ideal):
        raise ValueError("Actual and ideal assignments must have same length")
    
    if np.sum(ideal) == 0:
        return 0.0
    
    return np.sum(np.abs(actual - ideal)) / np.sum(ideal)


def calculate_ideal_fair_assignment(total_tasks: int, num_workers: int) -> List[int]:
    """Calculate ideal fair assignment (equal distribution)."""
    if num_workers == 0:
        return []
    
    base_tasks = total_tasks // num_workers
    extra_tasks = total_tasks % num_workers
    
    ideal = [base_tasks] * num_workers
    # Distribute extra tasks to first few workers
    for i in range(extra_tasks):
        ideal[i] += 1
    
    return ideal


class FairnessMetricsTracker:
    """Tracks fairness metrics throughout simulation."""
    
    def __init__(self):
        self.metrics_history = []
        self.worker_stats = {}
    
    def update_worker_stats(self, workers: List[Worker]):
        """Update worker statistics for fairness calculation."""
        for worker in workers:
            worker_id = worker.id
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    'completed_tasks': 0,
                    'total_revenue': 0.0,
                    'total_idle_time': 0.0,
                    'fairness_ewma': 0.0,
                    'last_active_time': None
                }
            
            stats = self.worker_stats[worker_id]
            stats['completed_tasks'] = worker.completed_tasks
            stats['total_revenue'] = worker.revenue
            stats['total_idle_time'] = worker.total_idle_time.total_seconds()
            stats['fairness_ewma'] = worker.fairness_ewma
            stats['last_active_time'] = worker.last_active_ts
    
    def calculate_current_fairness(self) -> Dict[str, float]:
        """Calculate current fairness metrics."""
        if not self.worker_stats:
            return {}
        
        # Extract task counts and revenues
        task_counts = [stats['completed_tasks'] for stats in self.worker_stats.values()]
        revenues = [stats['total_revenue'] for stats in self.worker_stats.values()]
        idle_times = [stats['total_idle_time'] for stats in self.worker_stats.values()]
        ewma_values = [stats['fairness_ewma'] for stats in self.worker_stats.values()]
        
        total_tasks = sum(task_counts)
        num_workers = len(task_counts)
        
        # Calculate metrics
        jfi_tasks = jains_fairness_index(task_counts)
        jfi_revenue = jains_fairness_index(revenues) if sum(revenues) > 0 else 1.0
        
        ud_tasks = utility_difference(task_counts)
        ud_revenue = utility_difference(revenues)
        ud_idle = utility_difference(idle_times)
        
        ideal_assignment = calculate_ideal_fair_assignment(total_tasks, num_workers)
        fl_tasks = fairness_loss(task_counts, ideal_assignment)
        
        # EWMA-based fairness metrics
        ewma_mean = np.mean(ewma_values) if ewma_values else 0.0
        ewma_std = np.std(ewma_values) if len(ewma_values) > 1 else 0.0
        ewma_cv = ewma_std / ewma_mean if ewma_mean > 0 else 0.0  # Coefficient of variation
        
        return {
            'jains_fairness_index_tasks': jfi_tasks,
            'jains_fairness_index_revenue': jfi_revenue,
            'utility_difference_tasks': ud_tasks,
            'utility_difference_revenue': ud_revenue,
            'utility_difference_idle_time': ud_idle,
            'fairness_loss_tasks': fl_tasks,
            'ewma_fairness_mean': ewma_mean,
            'ewma_fairness_std': ewma_std,
            'ewma_fairness_coefficient_variation': ewma_cv,
            'total_completed_tasks': total_tasks,
            'active_workers': num_workers
        }
    
    def record_snapshot(self, timestamp: pd.Timestamp):
        """Record a snapshot of current fairness metrics."""
        metrics = self.calculate_current_fairness()
        metrics['timestamp'] = timestamp
        self.metrics_history.append(metrics)
    
    def get_fairness_summary(self) -> Dict[str, float]:
        """Get summary of fairness metrics over entire simulation."""
        if not self.metrics_history:
            return {}
        
        # Get final metrics
        final_metrics = self.metrics_history[-1]
        
        # Calculate temporal statistics
        jfi_values = [m['jains_fairness_index_tasks'] for m in self.metrics_history]
        ewma_cv_values = [m['ewma_fairness_coefficient_variation'] for m in self.metrics_history]
        
        summary = {
            'final_jains_fairness_index': final_metrics.get('jains_fairness_index_tasks', 0.0),
            'final_utility_difference_tasks': final_metrics.get('utility_difference_tasks', 0.0),
            'final_fairness_loss': final_metrics.get('fairness_loss_tasks', 0.0),
            'final_ewma_cv': final_metrics.get('ewma_fairness_coefficient_variation', 0.0),
            'mean_jfi_over_time': np.mean(jfi_values),
            'min_jfi_over_time': np.min(jfi_values),
            'mean_ewma_cv_over_time': np.mean(ewma_cv_values),
            'max_ewma_cv_over_time': np.max(ewma_cv_values),
        }
        
        return summary