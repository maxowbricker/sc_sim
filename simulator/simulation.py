"""
Event-driven Spatial Crowdsourcing simulator.
"""

import pandas as pd
import numpy as np
from heapq import heappush, heappop
import copy

from typing import Dict, Optional, List, Tuple, Any
from config import get_simulation_config
from simulator.state import StateManager
from simulator.strategies import get_strategy
from metrics.fairness import FairnessMetricsTracker
from metrics.tracker import MetricTracker
from metrics.diagnostic_tracker import DiagnosticTracker

class EventSimulator:
    """
    Event-driven simulator that supports step-based execution for RL control.
    """
    
    def __init__(self, workers, tasks, sim_config: Optional[Dict] = None):
        """
        Initialize the simulator.
        
        Args:
            workers: List of Worker objects
            tasks: List of Task objects
            sim_config: Configuration dictionary
        """
        self.initial_workers = workers
        self.initial_tasks = tasks
        self.sim_config = get_simulation_config() if sim_config is None else {**get_simulation_config(), **sim_config}
        
        # Strategy setup
        self.strategy_name = self.sim_config["assignment_strategy"]
        self.strategy_params = self.sim_config.get("strategy_params", {})
        
        # Initialize handlers
        strategy_handler_factory = get_strategy(self.strategy_name)
        strategy_handlers = strategy_handler_factory()
        self.new_task_handler = strategy_handlers["NEW_TASK"]
        self.free_worker_handler = strategy_handlers["FREE_WORKER"]
        
        # State variables
        self.state = None
        self.event_queue = []
        self.current_time = None
        self.end_time = None
        self.summary = {}
        self.fairness_tracker = None
        self.metric_tracker = None
        self.diagnostic_tracker = None
        self.deferral_tracker = None
        
        # Logging/Tracking
        self.total_tasks_count = len(tasks)
        self.event_count = 0
        self.last_progress_report = 0
        self.max_events = len(tasks) * 10
        
        # Temporal tracking
        self.ewma_temporal_history = []
        self.temporal_log_interval = 50
        self.next_log_checkpoint = self.temporal_log_interval
        
    def reset(self, start_time=None, end_time=None):
        """
        Reset the simulation to initial state.
        """
        # Deep copy workers and tasks to ensure fresh state
        current_workers = copy.deepcopy(self.initial_workers)
        current_tasks = copy.deepcopy(self.initial_tasks)
        
        self.state = StateManager(current_workers, current_tasks)
        self.event_queue = []
        self.fairness_tracker = FairnessMetricsTracker()
        self.metric_tracker = MetricTracker()
        
        # Optional trackers
        if self.strategy_name == "composite" and self.strategy_params.get('enable_diagnostics', False):
            self.diagnostic_tracker = DiagnosticTracker()
            self.strategy_params['diagnostic_tracker'] = self.diagnostic_tracker
            
        if self.strategy_name == "composite" and self.strategy_params.get('enable_deferral_tracking', False):
            from metrics.deferral_tracker import DeferralTracker
            self.deferral_tracker = DeferralTracker()
            self.strategy_params['deferral_tracker'] = self.deferral_tracker
            
        if self.strategy_name == "fatp_ann":
            from simulator.strategies.fatp_ann import FairnessCapTracker
            fairness_cap_tracker = FairnessCapTracker()
            fairness_cap_tracker.initialize(current_workers)
            self.strategy_params['fairness_cap_tracker'] = fairness_cap_tracker

        # Determine start time
        if start_time is None:
            releases = [w.release_time for w in current_workers] + [t.release_time for t in current_tasks]
            if not releases: raise ValueError("Cannot infer start_time")
            start_time = min(releases)

        self.current_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time) if end_time is not None else None
        
        # Populate event queue
        for w in current_workers:
            heappush(self.event_queue, (w.release_time, "WORKER_RELEASE", w.id))
        for t in current_tasks:
            heappush(self.event_queue, (t.release_time, "TASK_RELEASE", t.id))
            
        # Reset summary
        self.summary = {
            'total_tasks': len(current_tasks),
            'completed_tasks': 0, 'total_travel_km': 0.0, 'empty_km': 0.0,
            'passenger_km': 0.0, 'total_wait_min': 0.0, 'wait_times': [],
            'service_times': [], 'backlog_peak': 0,
            'pickup_distances': [], 'assignment_delays': [],
            'expired_tasks': [],  # Track expired task IDs
        }
        
        self.event_count = 0
        self.ewma_temporal_history = []
        self.next_log_checkpoint = self.temporal_log_interval
        
        # RL tracking: total tasks released into the system
        self.total_tasks_released = 0
        
        return self.get_state()

    def update_weights(self, lambda1, lambda2, lambda3):
        """
        Update the strategy weights dynamically.
        """
        if self.strategy_name == "composite":
            self.strategy_params['λ1'] = lambda1
            self.strategy_params['λ2'] = lambda2
            self.strategy_params['λ3'] = lambda3

    def step(self, duration_seconds: float = None) -> bool:
        """
        Run simulation for a fixed duration or until completion.
        
        Args:
            duration_seconds: Time to advance simulation. If None, run to completion.
            
        Returns:
            done: True if simulation is finished, False otherwise.
        """
        if self.state is None:
            raise RuntimeError("Simulator not reset. Call reset() first.")
            
        # Initialize windowed stats for this step
        self.step_summary = {
            'completed_tasks': 0,
            'wait_times': [],
            'total_travel_km': 0.0,
            'empty_km': 0.0
        }
            
        target_time = self.current_time + pd.Timedelta(seconds=duration_seconds) if duration_seconds else None
        
        while self.event_queue:
            # Peek at next event time
            next_event_time = self.event_queue[0][0]
            
            # If we have a target time and next event is beyond it, stop here
            if target_time and next_event_time > target_time:
                # Advance current time to target time (simulating passage of time without events)
                time_delta = (target_time - self.current_time).total_seconds()
                if time_delta > 0:
                    for w in self.state.available_workers:
                        w.update_idle_time(time_delta)
                self.current_time = target_time
                return False # Not done
            
            # Process event
            self.event_count += 1
            if self.event_count > self.max_events:
                print(f"⚠️  Simulation terminated: Exceeded {self.max_events:,} events")
                return True
                
            event_time, event_type, event_id = heappop(self.event_queue)
            
            if self.end_time and event_time > self.end_time:
                return True

            time_delta_seconds = (event_time - self.current_time).total_seconds()
            if time_delta_seconds > 0:
                for w in self.state.available_workers:
                    w.update_idle_time(time_delta_seconds)
            
            self.current_time = event_time
            
            self._process_event(event_type, event_id)
            
            # Periodic updates
            current_backlog = len(self.state.active_tasks) + len(self.state.deferred_tasks)
            self.summary['backlog_peak'] = max(self.summary['backlog_peak'], current_backlog)
            
            if len(self.event_queue) % 100 == 0:
                self.fairness_tracker.update_worker_stats(self.state.all_workers_map.values())
                self.fairness_tracker.record_snapshot(self.current_time)
                self.metric_tracker.snapshot(self.state, self.current_time)
        
        return True # Done (queue empty)

    def _process_event(self, event_type, event_id):
        """Handle a single event."""
        if event_type == "WORKER_RELEASE":
            worker = self.state.get_worker(event_id)
            self.state.release_worker(worker)
            assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
            if assignment:
                assigned_task, assigned_worker, _ = assignment
                self.fairness_tracker.record_task_assignment(assigned_task, assigned_worker, self.current_time)
                heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                if assigned_task.start_time:
                    assignment_delay = (self.current_time - assigned_task.release_time).total_seconds()
                    self.summary['assignment_delays'].append(assignment_delay)

        elif event_type == "TASK_RELEASE":
            task = self.state.get_task(event_id)
            self.state.release_task(task)
            self.total_tasks_released += 1  # Track total tasks released for RL
            self.fairness_tracker.record_task_release(task, list(self.state.available_workers), self.current_time)
            
            if self.state.available_workers:
                # Add expiry scheduler callback to strategy params for tasks deferred in handler
                strategy_params_with_scheduler = {
                    **self.strategy_params,
                    'expiry_scheduler': lambda t: heappush(self.event_queue, (t.expire_time, "TASK_EXPIRE", t.id)) if self.current_time < t.expire_time else None
                }
                assignments = self.new_task_handler(self.state, self.current_time, [task], **strategy_params_with_scheduler)
                if assignments:
                    assigned_task, assigned_worker, _ = assignments[0]
                    self.fairness_tracker.record_task_assignment(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                    if assigned_task.start_time:
                        assignment_delay = (self.current_time - assigned_task.release_time).total_seconds()
                        self.summary['assignment_delays'].append(assignment_delay)
            else:
                if self.strategy_name == "composite":
                    # Defer task and schedule expiry event if not already expired
                    if self.state.defer_task(task, self.current_time):
                        heappush(self.event_queue, (task.expire_time, "TASK_EXPIRE", task.id))

        elif event_type == "TASK_COMPLETE":
            task = self.state.get_task(event_id)
            if not task.is_completed:
                worker = task.assigned_worker
                self.state.complete_task(task, worker, self.current_time)
                
                assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
                if assignment:
                    assigned_task, assigned_worker, _ = assignment
                    self.fairness_tracker.record_task_assignment(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                    if assigned_task.start_time:
                        assignment_delay = (self.current_time - assigned_task.release_time).total_seconds()
                        self.summary['assignment_delays'].append(assignment_delay)

                self._update_completion_stats(task)

        elif event_type == "TASK_EXPIRE":
            # Remove expired task from deferred state (set + index)
            # (may have already been assigned/completed, in which case remove_deferred_task is safe no-op)
            task = self.state.get_task(event_id)
            if task:
                self.state.remove_deferred_task(task)
                
                # Check if the task is currently assigned (running).
                # If it's assigned, it shouldn't count as expired, even if the expire event fires.
                # This prevents "zombie events" where TASK_EXPIRE fires for tasks that were
                # assigned before expiry but complete after expiry.
                is_assigned = task in self.state.assigned_tasks
                
                # Track expired tasks for metrics (only if not completed AND not assigned)
                if not is_assigned and task not in self.state.completed_tasks and not task.is_completed:
                    self.summary['expired_tasks'].append(task.id)

    def _update_completion_stats(self, task):
        """Update summary stats after task completion."""
        # Update global summary
        self.summary['completed_tasks'] += 1
        self.summary['total_travel_km'] += (task.pickup_km or 0) + (task.drop_km or 0)
        self.summary['empty_km'] += task.pickup_km or 0
        if task.pickup_km is not None:
            self.summary['pickup_distances'].append(task.pickup_km)
        self.summary['passenger_km'] += task.drop_km or 0
        wait_min = (task.start_time - task.release_time).total_seconds()/60 if task.start_time else 0
        self.summary['total_wait_min'] += wait_min
        self.summary['wait_times'].append(wait_min)
        service_min = (task.drop_km / 30 * 60) if task.drop_km is not None else 0
        self.summary['service_times'].append(service_min)
        
        # Update windowed step summary
        if hasattr(self, 'step_summary'):
            self.step_summary['completed_tasks'] += 1
            self.step_summary['wait_times'].append(wait_min)
            self.step_summary['total_travel_km'] += (task.pickup_km or 0) + (task.drop_km or 0)
            self.step_summary['empty_km'] += task.pickup_km or 0
        
        # Log EWMA temporal data
        if self.summary['completed_tasks'] >= self.next_log_checkpoint:
            ewma_values = [w.fairness_ewma for w in self.state.all_workers_map.values()]
            self.ewma_temporal_history.append({
                'timestamp': self.current_time.isoformat(),
                'completed_tasks': self.summary['completed_tasks'],
                'ewma_mean': float(np.mean(ewma_values)),
                'ewma_std': float(np.std(ewma_values))
            })
            self.next_log_checkpoint += self.temporal_log_interval

    def get_state(self):
        """
        Return the current state of the simulation for RL observation.
        """
        # Calculate windowed metrics
        step_avg_wait = 0.0
        if hasattr(self, 'step_summary') and self.step_summary['wait_times']:
            step_avg_wait = np.mean(self.step_summary['wait_times'])
        
        # Worker idle time statistics
        workers = list(self.state.all_workers_map.values())
        if workers:
            idle_times_min = [w.total_idle_time.total_seconds() / 60.0 for w in workers]
            mean_idle = float(np.mean(idle_times_min))
            std_idle = float(np.std(idle_times_min)) if len(idle_times_min) > 1 else 0.0
            cv_idle = std_idle / mean_idle if mean_idle > 0 else 0.0  # Coefficient of variation
        else:
            mean_idle = 0.0
            cv_idle = 0.0
        
        # Deferral reason breakdown
        if self.deferral_tracker:
            reason_breakdown = self.deferral_tracker.get_deferral_reason_breakdown()
            pct_below_threshold = reason_breakdown.get('pct_below_threshold', 0.0) / 100.0  # Convert to 0-1
            pct_no_candidates = reason_breakdown.get('pct_no_candidates', 0.0) / 100.0
        else:
            pct_below_threshold = 0.0
            pct_no_candidates = 0.0
        
        # Task release rate calculation
        step_duration = (self.current_time - self.step_start_time).total_seconds() if hasattr(self, 'step_start_time') and self.step_start_time else 1.0
        tasks_released = getattr(self, 'step_tasks_released', 0)
        assigned_workers_count = len(self.state.assigned_workers)
        total_active_workers = len(self.state.available_workers) + assigned_workers_count
        
        if step_duration > 0 and total_active_workers > 0:
            task_release_rate_per_min = (tasks_released / step_duration) * 60
            task_worker_ratio = task_release_rate_per_min / total_active_workers
        else:
            task_worker_ratio = 0.0

        return {
            'active_tasks': len(self.state.active_tasks),
            'deferred_tasks': len(self.state.deferred_tasks),
            'available_workers': len(self.state.available_workers),
            'assigned_workers': assigned_workers_count,  # NEW: For task-worker ratio
            'total_workers': len(self.state.all_workers_map),
            'completed_tasks': self.summary['completed_tasks'],
            'current_time': self.current_time,
            'workers': list(self.state.all_workers_map.values()),
            'backlog_peak': self.summary['backlog_peak'],
            'total_tasks_released': self.total_tasks_released,  # For RL deferred ratio calculation
            # Windowed stats
            'step_avg_wait': step_avg_wait,
            'step_completed_tasks': self.step_summary['completed_tasks'] if hasattr(self, 'step_summary') else 0,
            # NEW: Enhanced metrics for RL
            'task_worker_ratio': task_worker_ratio,  # Tasks per minute per worker
            'mean_worker_idle_min': mean_idle,
            'cv_worker_idle': cv_idle,  # Worker inequality (lower = more fair)
            'pct_deferrals_below_threshold': pct_below_threshold,
            'pct_deferrals_no_candidates': pct_no_candidates,
        }

    def get_final_results(self):
        """
        Calculate and return final simulation statistics.
        """
        total_tasks_count = self.total_tasks_count
        summary = self.summary
        
        tar = summary['completed_tasks'] / total_tasks_count if total_tasks_count else 0
        avg_travel_km = summary['total_travel_km'] / summary['completed_tasks'] if summary['completed_tasks'] else 0
        avg_wait_min = summary['total_wait_min'] / summary['completed_tasks'] if summary['completed_tasks'] else 0
        
        # Helper for safe stats
        def safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
        def safe_std(arr): return float(np.std(arr)) if arr else 0.0
        def safe_percentile(arr, p): return float(np.percentile(arr, p)) if arr else 0.0
        def safe_max(arr): return float(np.max(arr)) if arr else 0.0
        
        # Task completion metrics
        summary['task_assignment_ratio'] = tar
        summary['total_tasks'] = total_tasks_count
        
        # Wait times
        summary['avg_wait_time_minutes'] = avg_wait_min
        summary['std_wait_time_minutes'] = safe_std(summary['wait_times'])
        summary['p90_wait_time_minutes'] = safe_percentile(summary['wait_times'], 90)
        summary['max_wait_time_minutes'] = safe_max(summary['wait_times'])
        
        # Worker idle times
        worker_idle_times = [w.total_idle_time.total_seconds() / 60.0 for w in self.state.all_workers_map.values()]
        summary['mean_worker_idle_time_min'] = safe_mean(worker_idle_times)
        
        # Final fairness metrics calculation
        self.fairness_tracker.update_worker_stats(self.state.all_workers_map.values())
        fairness_summary = self.fairness_tracker.get_fairness_summary()
        
        # Combine all metrics
        summary.update(fairness_summary)
        summary['metric_tracker'] = self.metric_tracker
        
        if self.diagnostic_tracker:
            summary['diagnostic_tracker'] = self.diagnostic_tracker
            summary['diagnostic_summary'] = self.diagnostic_tracker.get_summary_stats()
            
        if self.deferral_tracker:
            summary['deferral_stats'] = self.deferral_tracker.get_summary()
            
        if self.ewma_temporal_history:
            summary['ewma_temporal_history'] = self.ewma_temporal_history
            summary['ewma_final_mean'] = self.ewma_temporal_history[-1]['ewma_mean'] if self.ewma_temporal_history else 0
            
        return summary


def run_simulation(workers, tasks, start_time=None, end_time=None, sim_config: Optional[Dict] = None):
    """
    Wrapper for backward compatibility.
    """
    sim = EventSimulator(workers, tasks, sim_config)
    sim.reset(start_time, end_time)
    sim.step(duration_seconds=None) # Run to completion
    return sim.get_final_results()


class Simulation:
    """Wrapper class for the event-driven simulation to match notebook expectations."""
    
    def __init__(self, config, workers_df, tasks_df):
        """Initialize simulation with config and data."""
        self.config = config
        self.workers_df = workers_df
        self.tasks_df = tasks_df
        self.metric_tracker = None  # Will be populated after running simulation
        
    def run(self):
        """Run the simulation and return standardized results."""
        # Convert DataFrames to the expected format
        from models.worker import Worker
        from models.task import Task
        
        print("🚀 Converting DataFrames to objects...")
        
        # Convert workers DataFrame to Worker objects (FAST vectorized approach)
        workers = []
        total_workers = len(self.workers_df)
        print(f"   📊 Converting {total_workers:,} workers...")
        
        # Use to_dict('records') which is much faster than iterrows()
        worker_records = self.workers_df.to_dict('records')
        
        for i, row in enumerate(worker_records):
            # Progress indicator every 10000 workers
            if i % 10000 == 0 and i > 0:
                print(f"   📊 Progress: {i:,}/{total_workers:,} ({i/total_workers*100:.1f}%)")
            
            worker_dict = {
                'worker_id': row['worker_id'],
                'start_lat': row['start_lat'],
                'start_lon': row['start_lon'],
                'release_time': row['release_time'],
                'deadline': row['deadline']
            }
            worker = Worker(worker_dict)
            workers.append(worker)
        
        print(f"   ✅ Converted {len(workers):,} workers")
        
        # CRITICAL: Update worker gamma if specified in config (Exp 019 fix)
        strategy_params = self.config.get('strategy_params', {})
        if 'gamma' in strategy_params:
            gamma_value = strategy_params['gamma']
            print(f"   🔧 Updating worker gamma to {gamma_value}...")
            for worker in workers:
                worker.gamma = gamma_value
        
        # Convert tasks DataFrame to Task objects (FAST vectorized approach)
        tasks = []
        total_tasks = len(self.tasks_df)
        print(f"   📊 Converting {total_tasks:,} tasks...")
        
        # Use to_dict('records') which is much faster than iterrows()
        task_records = self.tasks_df.to_dict('records')
        
        for i, row in enumerate(task_records):
            # Progress indicator every 20000 tasks
            if i % 20000 == 0 and i > 0:
                print(f"   📊 Progress: {i:,}/{total_tasks:,} ({i/total_tasks*100:.1f}%)")
            
            task_dict = {
                'task_id': row['task_id'],
                'pickup_lat': row['pickup_lat'],
                'pickup_lon': row['pickup_lon'],
                'dropoff_lat': row['dropoff_lat'],
                'dropoff_lon': row['dropoff_lon'],
                'release_time': row['release_time'],
                'expire_time': row['expire_time']
            }
            task = Task(task_dict)
            tasks.append(task)
        
        print(f"   ✅ Converted {len(tasks):,} tasks")
        print("🚀 Starting simulation...")
        
        # Run the simulation
        results = run_simulation(workers, tasks, sim_config=self.config)
        
        # Store the enhanced metric tracker for later analysis
        self.metric_tracker = results.get('metric_tracker')
        
        # Standardize results for notebook analysis
        total_tasks = len(tasks)
        completed_tasks = results.get('completed_tasks', 0)
        assigned_tasks = results.get('total_task_assignments_tracked', 0)
        
        standardized_results = {
            'jfi': results.get('final_jains_fairness_index', 0.0),
            # TAR: Percentage of tasks that were assigned to a worker
            'task_assignment_ratio': assigned_tasks / total_tasks if total_tasks > 0 else 0.0,
            # Throughput: Percentage of tasks that completed full service cycle
            'task_completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0.0,
            'avg_wait_time_minutes': results.get('total_wait_min', 0) / completed_tasks if completed_tasks > 0 else 0.0,
            'avg_pickup_distance_km': results.get('empty_km', 0) / completed_tasks if completed_tasks > 0 else 0.0,
            'total_travel_distance_km': results.get('total_travel_km', 0),
            'empty_km_ratio': results.get('empty_km', 0) / results.get('total_travel_km', 1) if results.get('total_travel_km', 0) > 0 else 0.0,
            'ewma_cv': results.get('final_ewma_cv', 1.0),
            'utility_difference': results.get('final_utility_difference_tasks', 0.0),
            'fairness_loss': results.get('final_fairness_loss', 0.0),
            'total_tasks': total_tasks,
            'assigned_tasks': assigned_tasks,  # Number of tasks assigned to workers
            'completed_tasks': completed_tasks,  # Number of tasks that finished service
            'max_wait_time': max(results.get('wait_times', [0])) if results.get('wait_times') else 0.0,
            'backlog_peak': results.get('backlog_peak', 0),
            
            # Supervisor's enhanced fairness metrics
            'supervisor_utility_difference': results.get('supervisor_utility_difference'),
            'supervisor_fairness_loss': results.get('supervisor_fairness_loss'),
            'mean_input_output_ratio': results.get('mean_input_output_ratio'),
            'min_input_output_ratio': results.get('min_input_output_ratio'),
            'max_input_output_ratio': results.get('max_input_output_ratio'),
            'workers_with_eligibility_data': results.get('workers_with_eligibility_data', 0),
            'total_task_assignments_tracked': results.get('total_task_assignments_tracked', 0),
            
            # EXPERIMENT 008: Pass through diagnostic tracker and summary
            'diagnostic_tracker': results.get('diagnostic_tracker'),
            'diagnostic_summary': results.get('diagnostic_summary'),
            
            # EXPERIMENT 019: Pass through deferral tracker stats (RQ3.3)
            'deferral_stats': results.get('deferral_stats'),
        }
        
        return standardized_results
