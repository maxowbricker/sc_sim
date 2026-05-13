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
from typing import List, Dict, Tuple
from models.worker import Worker


def jains_fairness_index(task_counts: List[int]) -> float:
    """Calculate Jain's Fairness Index. Returns 0 (unfair) to 1.0 (fair)."""
    if not task_counts:
        return 1.0
    
    x = np.array(task_counts, dtype=float)
    n = len(x)
    
    sum_x = np.sum(x)
    if sum_x == 0:
        return 1.0  
    
    denominator = n * np.sum(x ** 2)
    return (sum_x ** 2) / denominator if denominator > 0 else 1.0


def utility_difference(worker_utilities: List[float]) -> float:
    """Calculate Utility Difference (UD): Max count - Min count."""
    if not worker_utilities or len(worker_utilities) <= 1:
        return 0.0
    
    utilities = np.array(worker_utilities)
    return float(np.max(utilities) - np.min(utilities))


def fairness_loss(actual_assignments: List[int], ideal_assignments: List[int]) -> float:
    """Calculate Fairness Loss - deviation from ideal fair assignment."""
    if not actual_assignments or not ideal_assignments:
        return 0.0
    
    actual = np.array(actual_assignments, dtype=float)
    ideal = np.array(ideal_assignments, dtype=float)
    
    if len(actual) != len(ideal):
        raise ValueError("Actual and ideal assignments must have same length")
    
    sum_ideal = np.sum(ideal)
    if sum_ideal == 0:
        return 0.0
    
    return float(np.sum(np.abs(actual - ideal)) / sum_ideal)


def calculate_ideal_fair_assignment(total_tasks: int, num_workers: int) -> List[int]:
    """Calculate ideal fair assignment (equal distribution)."""
    if num_workers == 0:
        return []
    
    base_tasks = total_tasks // num_workers
    extra_tasks = total_tasks % num_workers
    
    ideal = [base_tasks] * num_workers
    for i in range(extra_tasks):
        ideal[i] += 1
    
    return ideal


# Heavy Evaluation Tracking (Disabled during DRL training)

def fairness_loss_supervisor_definition(worker_stats: Dict[str, Dict]) -> float:
    """Calculate Fairness Loss (FL) based on reachable area proportion."""
    if not worker_stats:
        return 0.0
    
    total_fl = 0.0
    total_ideal = 0.0
    
    for stats in worker_stats.values():
        actual = stats.get('actual_tasks', 0)
        ideal = stats.get('ideal_share', 0)
        
        total_fl += max(0, ideal - actual)
        total_ideal += ideal
    
    return total_fl / total_ideal if total_ideal > 0 else 0.0


class FairnessMetricsTracker:
    """Tracks heavy evaluation-only fairness metrics throughout simulation."""
    
    def __init__(self, enable_diagnostics: bool = False):
        self.enable_diagnostics = enable_diagnostics
        self.metrics_history = []
        self.worker_stats = {}
        
        # Enhanced tracking for supervisor's UD and FL definitions
        self.task_eligibility_log = {}  
        self.worker_eligibility_stats = {}  
        self.reachable_distance_km = 10.0  
    
    def update_worker_stats(self, workers: List[Worker]):
        if not self.enable_diagnostics:
            return

        for worker in workers:
            worker_id = worker.id
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {
                    'completed_tasks': 0,
                    'total_idle_time': 0.0,
                    'fairness_ewma': 0.0,
                    'last_active_time': None
                }
            
            stats = self.worker_stats[worker_id]
            stats['completed_tasks'] = worker.completed_tasks
            stats['total_idle_time'] = worker.total_idle_time
            stats['fairness_ewma'] = worker.fairness_ewma
            stats['last_active_time'] = worker.last_active_ts
    
    def record_task_release(self, task, available_workers: List, current_time: float):
        """
        Record when a task is released and determine eligible workers.
        Gated behind enable_diagnostics to avoid massive O(|W|) spatial scans during RL.
        """
        if not self.enable_diagnostics:
            return

        from simulator.spatial_index import fast_manhattan_km
        
        task_id = str(task.id)
        eligible_workers = []
        
        for worker in available_workers:
            distance = fast_manhattan_km(
                worker.start_lat, worker.start_lon,
                task.pickup_lat, task.pickup_lon
            )
            
            if distance <= self.reachable_distance_km:
                eligible_workers.append(str(worker.id))
                
                worker_id = str(worker.id)
                if worker_id not in self.worker_eligibility_stats:
                    self.worker_eligibility_stats[worker_id] = {
                        'eligible_tasks': 0,
                        'actual_tasks': 0,
                        'total_eligible_distance': 0.0
                    }
                
                self.worker_eligibility_stats[worker_id]['eligible_tasks'] += 1
                self.worker_eligibility_stats[worker_id]['total_eligible_distance'] += distance
        
        self.task_eligibility_log[task_id] = {
            'eligible_workers': eligible_workers,
            'release_time': current_time,
            'assigned_worker': None,
            'assignment_time': None
        }
    
    def record_task_assignment(self, task, assigned_worker, assignment_time: float):
        if not self.enable_diagnostics:
            return

        task_id = str(task.id)
        worker_id = str(assigned_worker.id)
        
        if worker_id not in self.worker_eligibility_stats:
            self.worker_eligibility_stats[worker_id] = {
                'eligible_tasks': 0,
                'actual_tasks': 0,
                'total_eligible_distance': 0.0
            }
        
        self.worker_eligibility_stats[worker_id]['actual_tasks'] += 1
        
        if task_id in self.task_eligibility_log:
            self.task_eligibility_log[task_id]['assigned_worker'] = worker_id
            self.task_eligibility_log[task_id]['assignment_time'] = assignment_time

    def get_fairness_summary(self) -> Dict[str, float]:
        """Get summary of fairness metrics over entire simulation."""
        if not self.enable_diagnostics or not self.worker_eligibility_stats:
            return {}
            
        supervisor_ud = self.calculate_supervisor_utility_difference()
        supervisor_fl, ior_stats = self.calculate_supervisor_fairness_loss()
        
        return {
            'supervisor_utility_difference': supervisor_ud,
            'supervisor_fairness_loss': supervisor_fl,
            'mean_input_output_ratio': float(np.mean(list(ior_stats.values()))) if ior_stats else 0.0,
            'min_input_output_ratio': float(np.min(list(ior_stats.values()))) if ior_stats else 0.0,
            'max_input_output_ratio': float(np.max(list(ior_stats.values()))) if ior_stats else 0.0,
            'workers_with_eligibility_data': len(self.worker_eligibility_stats),
            'total_task_assignments_tracked': len(self.task_eligibility_log),
        }
    
    def calculate_supervisor_utility_difference(self) -> float:
        if not self.worker_eligibility_stats:
            return 0.0
        task_counts = [stats['actual_tasks'] for stats in self.worker_eligibility_stats.values()]
        return max(task_counts) - min(task_counts) if task_counts else 0.0
    
    def calculate_supervisor_fairness_loss(self) -> Tuple[float, Dict[str, float]]:
        if not self.worker_eligibility_stats:
            return 0.0, {}
        
        total_assigned = sum(stats['actual_tasks'] for stats in self.worker_eligibility_stats.values())
        total_eligibility = sum(stats['eligible_tasks'] for stats in self.worker_eligibility_stats.values())
        
        if total_eligibility == 0:
            return 0.0, {}
        
        worker_stats_for_fl = {}
        ior_stats = {}
        
        for worker_id, stats in self.worker_eligibility_stats.items():
            eligible_tasks = stats['eligible_tasks']
            actual_tasks = stats['actual_tasks']
            
            ideal_share = (eligible_tasks / total_eligibility) * total_assigned
            
            worker_stats_for_fl[worker_id] = {
                'actual_tasks': actual_tasks,
                'ideal_share': ideal_share
            }
            ior_stats[worker_id] = actual_tasks / eligible_tasks if eligible_tasks > 0 else 0.0
        
        fl_value = fairness_loss_supervisor_definition(worker_stats_for_fl)
        return fl_value, ior_stats
 

def gini_coefficient(task_counts: List[int]) -> float:
    """
    Calculate the Gini coefficient of a frequency distribution (e.g., worker task counts).
    Returns a value between 0.0 (perfect equality) and 1.0 (maximum inequality).
    """
    x = np.array(task_counts, dtype=np.float64)
    if x.size == 0 or np.sum(x) == 0:
        return 0.0

    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)

    # Vectorized Gini calculation
    gini = ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))
    return float(gini)