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
    Calculate Utility Difference (UD) as per supervisor feedback:
    UD = (Maximum task count among workers) - (Minimum task count among workers)
    
    This measures disparity in task allocation - lower values indicate better fairness.
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


def fairness_loss_supervisor_definition(worker_stats: Dict[str, Dict]) -> float:
    """
    Calculate Fairness Loss (FL) as per supervisor feedback:
    FL = sum((Worker's ideal task share) - (Actual tasks assigned)) for all workers
    
    Where ideal task share is based on worker availability in their reachable area.
    This implements the F-Aware paper's Input/Output Ratio (IOR) concept.
    
    Args:
        worker_stats: Dict mapping worker_id -> {
            'actual_tasks': int,
            'eligible_tasks': int,  # tasks within reachable area
            'ideal_share': float    # proportional share based on eligibility
        }
    """
    if not worker_stats:
        return 0.0
    
    total_fl = 0.0
    total_ideal = 0.0
    
    for worker_id, stats in worker_stats.items():
        actual = stats.get('actual_tasks', 0)
        ideal = stats.get('ideal_share', 0)
        
        # FL contribution for this worker
        fl_contribution = max(0, ideal - actual)  # Only positive deviations
        total_fl += fl_contribution
        total_ideal += ideal
    
    return total_fl / total_ideal if total_ideal > 0 else 0.0


def calculate_input_output_ratio(worker_stats: Dict[str, Dict]) -> Dict[str, float]:
    """
    Calculate Input/Output Ratio (IOR) for each worker as per F-Aware paper:
    IOR = (tasks received) / (tasks within worker's reachable area)
    
    Args:
        worker_stats: Dict mapping worker_id -> {
            'actual_tasks': int,
            'eligible_tasks': int
        }
    
    Returns:
        Dict mapping worker_id -> IOR value
    """
    ior_results = {}
    
    for worker_id, stats in worker_stats.items():
        actual = stats.get('actual_tasks', 0)
        eligible = stats.get('eligible_tasks', 0)
        
        ior_results[worker_id] = actual / eligible if eligible > 0 else 0.0
    
    return ior_results


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
        
        # Enhanced tracking for supervisor's UD and FL definitions
        self.task_eligibility_log = {}  # task_id -> [eligible_worker_ids]
        self.worker_eligibility_stats = {}  # worker_id -> {'eligible_tasks': int, 'actual_tasks': int}
        self.reachable_distance_km = 10.0  # Default reachable area (can be configured)
    
    def update_worker_stats(self, workers: List[Worker]):
        """Update worker statistics for fairness calculation."""
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
    
    def calculate_current_fairness(self) -> Dict[str, float]:
        """Calculate current fairness metrics."""
        if not self.worker_stats:
            return {}
        
        # Extract task counts and idle times
        task_counts = [stats['completed_tasks'] for stats in self.worker_stats.values()]
        idle_times = [stats['total_idle_time'] for stats in self.worker_stats.values()]
        ewma_values = [stats['fairness_ewma'] for stats in self.worker_stats.values()]
        
        total_tasks = sum(task_counts)
        num_workers = len(task_counts)
        
        # Calculate metrics
        jfi_tasks = jains_fairness_index(task_counts)
        ud_tasks = utility_difference(task_counts)
        ud_idle = utility_difference(idle_times)
        
        ideal_assignment = calculate_ideal_fair_assignment(total_tasks, num_workers)
        fl_tasks = fairness_loss(task_counts, ideal_assignment)
        
        # EWMA-based fairness metrics
        ewma_mean = np.mean(ewma_values) if ewma_values else 0.0
        ewma_std = np.std(ewma_values) if len(ewma_values) > 1 else 0.0
        ewma_cv = ewma_std / ewma_mean if ewma_mean > 0 else 0.0  # Coefficient of variation
        
        return {
            'jains_fairness_index_tasks': jfi_tasks,
            'utility_difference_tasks': ud_tasks,
            'utility_difference_idle_time': ud_idle,
            'fairness_loss_tasks': fl_tasks,
            'ewma_fairness_mean': ewma_mean,
            'ewma_fairness_std': ewma_std,
            'ewma_fairness_coefficient_variation': ewma_cv,
            'total_completed_tasks': total_tasks,
            'active_workers': num_workers
        }
    
    def record_task_release(self, task, available_workers: List, current_time: pd.Timestamp):
        """
        Record when a task is released and determine eligible workers.
        
        Args:
            task: The released task
            available_workers: List of currently available workers
            current_time: Current simulation time
        """
        from simulator.spatial_index import manhattan_km
        
        task_id = str(task.id)
        eligible_workers = []
        
        # Determine which workers are within reachable distance
        for worker in available_workers:
            distance = manhattan_km(
                worker.start_lat, worker.start_lon,
                task.pickup_lat, task.pickup_lon
            )
            
            if distance <= self.reachable_distance_km:
                eligible_workers.append(str(worker.id))
                
                # Track eligibility for this worker
                worker_id = str(worker.id)
                if worker_id not in self.worker_eligibility_stats:
                    self.worker_eligibility_stats[worker_id] = {
                        'eligible_tasks': 0,
                        'actual_tasks': 0,
                        'total_eligible_distance': 0.0
                    }
                
                self.worker_eligibility_stats[worker_id]['eligible_tasks'] += 1
                self.worker_eligibility_stats[worker_id]['total_eligible_distance'] += distance
        
        # Log eligible workers for this task
        self.task_eligibility_log[task_id] = {
            'eligible_workers': eligible_workers,
            'release_time': current_time,
            'assigned_worker': None,
            'assignment_time': None
        }
    
    def record_task_assignment(self, task, assigned_worker, assignment_time: pd.Timestamp):
        """
        Record when a task is actually assigned to a worker.
        
        Args:
            task: The assigned task
            assigned_worker: The worker assigned to the task
            assignment_time: When the assignment occurred
        """
        task_id = str(task.id)
        worker_id = str(assigned_worker.id)
        
        # Update eligibility stats for the assigned worker
        if worker_id not in self.worker_eligibility_stats:
            self.worker_eligibility_stats[worker_id] = {
                'eligible_tasks': 0,
                'actual_tasks': 0,
                'total_eligible_distance': 0.0
            }
        
        self.worker_eligibility_stats[worker_id]['actual_tasks'] += 1
        
        # Update task log
        if task_id in self.task_eligibility_log:
            self.task_eligibility_log[task_id]['assigned_worker'] = worker_id
            self.task_eligibility_log[task_id]['assignment_time'] = assignment_time

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
        
        # Calculate supervisor's UD and FL metrics
        supervisor_ud = self.calculate_supervisor_utility_difference()
        supervisor_fl, ior_stats = self.calculate_supervisor_fairness_loss()
        
        summary = {
            'final_jains_fairness_index': final_metrics.get('jains_fairness_index_tasks', 0.0),
            'final_utility_difference_tasks': final_metrics.get('utility_difference_tasks', 0.0),
            'final_fairness_loss': final_metrics.get('fairness_loss_tasks', 0.0),
            'final_ewma_cv': final_metrics.get('ewma_fairness_coefficient_variation', 0.0),
            'mean_jfi_over_time': np.mean(jfi_values),
            'min_jfi_over_time': np.min(jfi_values),
            'mean_ewma_cv_over_time': np.mean(ewma_cv_values),
            'max_ewma_cv_over_time': np.max(ewma_cv_values),
            
            # Supervisor's enhanced fairness metrics
            'supervisor_utility_difference': supervisor_ud,
            'supervisor_fairness_loss': supervisor_fl,
            'mean_input_output_ratio': np.mean(list(ior_stats.values())) if ior_stats else 0.0,
            'min_input_output_ratio': np.min(list(ior_stats.values())) if ior_stats else 0.0,
            'max_input_output_ratio': np.max(list(ior_stats.values())) if ior_stats else 0.0,
            'workers_with_eligibility_data': len(self.worker_eligibility_stats),
            'total_task_assignments_tracked': len(self.task_eligibility_log),
        }
        
        return summary
    
    def calculate_supervisor_utility_difference(self) -> float:
        """
        Calculate Utility Difference (UD) as per supervisor feedback:
        UD = (Maximum task count among workers) - (Minimum task count among workers)
        """
        if not self.worker_eligibility_stats:
            return 0.0
        
        task_counts = [stats['actual_tasks'] for stats in self.worker_eligibility_stats.values()]
        
        if not task_counts:
            return 0.0
        
        return max(task_counts) - min(task_counts)
    
    def calculate_supervisor_fairness_loss(self) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Fairness Loss (FL) as per supervisor feedback:
        FL = sum((Worker's ideal task share) - (Actual tasks assigned))
        
        Where ideal task share is proportional to worker's eligibility.
        
        Returns:
            tuple: (fairness_loss_value, input_output_ratios)
        """
        if not self.worker_eligibility_stats:
            return 0.0, {}
        
        # Calculate total tasks assigned
        total_assigned = sum(stats['actual_tasks'] for stats in self.worker_eligibility_stats.values())
        total_eligibility = sum(stats['eligible_tasks'] for stats in self.worker_eligibility_stats.values())
        
        if total_eligibility == 0:
            return 0.0, {}
        
        # Calculate ideal share for each worker based on their eligibility proportion
        worker_stats_for_fl = {}
        ior_stats = {}
        
        for worker_id, stats in self.worker_eligibility_stats.items():
            eligible_tasks = stats['eligible_tasks']
            actual_tasks = stats['actual_tasks']
            
            # Ideal share = (worker's eligibility / total eligibility) * total assigned tasks
            ideal_share = (eligible_tasks / total_eligibility) * total_assigned if total_eligibility > 0 else 0
            
            worker_stats_for_fl[worker_id] = {
                'actual_tasks': actual_tasks,
                'eligible_tasks': eligible_tasks,
                'ideal_share': ideal_share
            }
            
            # Calculate Input/Output Ratio for this worker
            ior_stats[worker_id] = actual_tasks / eligible_tasks if eligible_tasks > 0 else 0.0
        
        # Calculate FL using supervisor's definition
        fl_value = fairness_loss_supervisor_definition(worker_stats_for_fl)
        
        return fl_value, ior_stats