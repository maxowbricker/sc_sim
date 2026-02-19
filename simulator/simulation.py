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
from metrics.manager import MetricsManager
from simulator.spatial_index import set_city_constants

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
        
        # Metrics manager
        metrics_config = {
            'enable_diagnostics': self.strategy_params.get('enable_diagnostics', False),
            'enable_deferral_tracking': self.strategy_params.get('enable_deferral_tracking', False),
            'strategy_params': self.strategy_params  # Pass through for tracker injection
        }
        self.metrics = MetricsManager(metrics_config)
        
        # Logging/Tracking
        self.total_tasks_count = len(tasks)
        self.event_count = 0
        self.last_progress_report = 0
        self.max_events = len(tasks) * 10
        
    def reset(self, start_time=None, end_time=None):
        """
        Reset the simulation to initial state.
        """
        # Deep copy workers and tasks to ensure fresh state
        current_workers = copy.deepcopy(self.initial_workers)
        current_tasks = copy.deepcopy(self.initial_tasks)
        
        # --- FLAT EARTH SETUP ---
        # Calculate mean latitude once per episode for optimized distance calculations
        # This pre-calculates cos(lat) to avoid expensive trig calls in hot loops
        all_lats = [w.start_lat for w in current_workers] + [t.pickup_lat for t in current_tasks]
        if all_lats:
            mean_lat = float(np.mean(all_lats))
            set_city_constants(mean_lat)
        
        # Ensure dynamic attributes are initialized correctly and timestamps are floats for fresh episode
        for worker in current_workers:
            # Reset total_idle_time to 0.0 (float) for fresh episode
            worker.total_idle_time = 0.0
            # Ensure last_state_ts is initialized to release_time (which is now float)
            worker.last_state_ts = worker.release_time
            # Reset last_active_ts (will be set when worker completes tasks)
            worker.last_active_ts = None
        
        self.state = StateManager(current_workers, current_tasks)
        self.event_queue = []
        
        # Reset metrics manager (it will reinitialize trackers)
        metrics_config = {
            'enable_diagnostics': self.strategy_params.get('enable_diagnostics', False),
            'enable_deferral_tracking': self.strategy_params.get('enable_deferral_tracking', False),
            'strategy_params': self.strategy_params
        }
        self.metrics = MetricsManager(metrics_config)
        
        # Special trackers for other strategies
        if self.strategy_name == "fatp_ann":
            from simulator.strategies.fatp_ann import FairnessCapTracker
            fairness_cap_tracker = FairnessCapTracker()
            fairness_cap_tracker.initialize(current_workers)
            self.strategy_params['fairness_cap_tracker'] = fairness_cap_tracker

        # Determine start time (timestamps are already floats from data loader)
        if start_time is None:
            releases = [w.release_time for w in current_workers] + [t.release_time for t in current_tasks]
            if not releases: raise ValueError("Cannot infer start_time")
            start_time = min(releases)

        self.current_time = start_time
        self.end_time = end_time
        
        # Populate event queue
        for w in current_workers:
            heappush(self.event_queue, (w.release_time, "WORKER_RELEASE", w.id))
        for t in current_tasks:
            heappush(self.event_queue, (t.release_time, "TASK_RELEASE", t.id))
        
        self.event_count = 0
        
        # Track step start time for RL
        self.step_start_time = None
        
        return self.get_state()

    def update_weights(self, lambda1, lambda2, lambda3):
        """
        Update the strategy weights dynamically.
        """
        if self.strategy_name == "composite":
            self.strategy_params['λ1'] = lambda1
            self.strategy_params['λ2'] = lambda2
            self.strategy_params['λ3'] = lambda3

    def switch_strategy(self, strategy_name: str, strategy_params: Optional[Dict] = None):
        """
        Hot-swap the assignment strategy.
        Used to switch from 'greedy' (warmup) to 'composite' (RL training).
        
        Args:
            strategy_name: Name of the new strategy (e.g., 'greedy', 'composite')
            strategy_params: Optional dictionary of strategy parameters to update
        """
        self.strategy_name = strategy_name
        if strategy_params:
            self.strategy_params.update(strategy_params)
        
        # Re-initialize handlers for the new strategy
        strategy_handler_factory = get_strategy(self.strategy_name)
        strategy_handlers = strategy_handler_factory()
        
        self.new_task_handler = strategy_handlers["NEW_TASK"]
        self.free_worker_handler = strategy_handlers["FREE_WORKER"]
        
        # Update metrics config to reflect new strategy
        if hasattr(self, 'metrics'):
            self.metrics.config['strategy_params'] = self.strategy_params

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
            
        # Track step start time for metrics
        if self.step_start_time is None:
            self.step_start_time = self.current_time
            
        target_time = self.current_time + duration_seconds if duration_seconds else None
        
        while self.event_queue:
            # Peek at next event time
            next_event_time = self.event_queue[0][0]
            
            # If we have a target time and next event is beyond it, stop here
            if target_time and next_event_time > target_time:
                # Advance current time to target time (simulating passage of time without events)
                self.current_time = target_time
                # Snapshot metrics at end of step
                self.metrics.snapshot_step(self.state, self.current_time, self.step_start_time)
                self.step_start_time = None  # Reset for next step
                return False # Not done
            
            # Process event
            self.event_count += 1
            if self.event_count > self.max_events:
                print(f"⚠️  Simulation terminated: Exceeded {self.max_events:,} events")
                return True
                
            event_time, event_type, event_id = heappop(self.event_queue)
            
            if self.end_time and event_time > self.end_time:
                return True
            
            self.current_time = event_time
            
            self._process_event(event_type, event_id)
            
        # Snapshot metrics at end of simulation
        if self.step_start_time:
            self.metrics.snapshot_step(self.state, self.current_time, self.step_start_time)
        
        return True # Done (queue empty)

    def _process_event(self, event_type, event_id):
        """Handle a single event."""
        if event_type == "WORKER_RELEASE":
            worker = self.state.get_worker(event_id)
            self.state.release_worker(worker)
            assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
            if assignment:
                assigned_task, assigned_worker, _ = assignment
                # Metrics manager handles assignment tracking
                self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

        elif event_type == "TASK_RELEASE":
            task = self.state.get_task(event_id)
            self.state.release_task(task)
            # Metrics manager tracks task release
            self.metrics.on_task_released(task, list(self.state.available_workers), self.current_time)
            
            if self.state.available_workers:
                # Add expiry scheduler callback to strategy params for tasks deferred in handler
                strategy_params_with_scheduler = {
                    **self.strategy_params,
                    'expiry_scheduler': lambda t: heappush(self.event_queue, (t.expire_time, "TASK_EXPIRE", t.id)) if self.current_time < t.expire_time else None
                }
                assignments = self.new_task_handler(self.state, self.current_time, [task], **strategy_params_with_scheduler)
                if assignments:
                    assigned_task, assigned_worker, _ = assignments[0]
                    # Metrics manager handles assignment tracking
                    self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
            else:
                if self.strategy_name == "composite":
                    # Defer task and schedule expiry event if not already expired
                    if self.state.defer_task(task, self.current_time):
                        heappush(self.event_queue, (task.expire_time, "TASK_EXPIRE", task.id))
                        # Track deferral (score and reason will be tracked by strategy if deferral_tracker is enabled)
                        self.metrics.on_task_deferred(task, 0.0, "no_candidates", self.current_time)

        elif event_type == "TASK_COMPLETE":
            task = self.state.get_task(event_id)
            if not task.is_completed:
                worker = task.assigned_worker
                self.state.complete_task(task, worker, self.current_time)
                
                assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
                if assignment:
                    assigned_task, assigned_worker, _ = assignment
                    # Metrics manager handles assignment tracking
                    self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

                # Metrics manager handles completion tracking
                self.metrics.on_task_completed(task, worker, self.current_time)

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
                    self.metrics.summary['expired_tasks'].append(task.id)

    def get_state(self):
        """
        Return the current state of the simulation for RL observation.
        
        Now delegates to MetricsManager for clean, unified data.
        """
        # Get observation data from metrics manager
        obs_data = self.metrics.get_observation_data(self.state, self.current_time)
        
        # Add state-specific data
        obs_data.update({
            'active_tasks': len(self.state.active_tasks),
            'assigned_workers': len(self.state.assigned_workers),
        })
        
        return obs_data

    def get_final_results(self):
        """
        Calculate and return final simulation statistics.
        
        Now delegates to MetricsManager for unified results.
        """
        # Get results from metrics manager
        results = self.metrics.get_final_results()
        
        # Add simulation-specific calculations
        total_tasks_count = self.total_tasks_count
        summary = self.metrics.summary
        
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
        worker_idle_times = [w.total_idle_time / 60.0 for w in self.state.all_workers_map.values()]
        summary['mean_worker_idle_time_min'] = safe_mean(worker_idle_times)
        
        # Update results with summary
        results.update(summary)
        
        return results


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
        
        # --- FLAT EARTH SETUP ---
        # Calculate mean latitude from the raw DataFrames FIRST
        # This ensures constants are set before Worker/Task __init__ runs
        # (Task.__init__ calls _calculate_base_utility() which uses fast_manhattan_km)
        print("   🌍 Configuring Flat-Earth Projection...")
        mean_worker_lat = self.workers_df['start_lat'].mean()
        mean_task_lat = self.tasks_df['pickup_lat'].mean()
        # Weighted average or simple mean is fine for this scale
        mean_lat = (mean_worker_lat + mean_task_lat) / 2
        set_city_constants(mean_lat)
        print(f"   ✅ Mean latitude: {mean_lat:.4f}°")
        
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
            
            # EXPERIMENT 019: Pass through deferral tracker stats (RQ3.3)
            'deferral_stats': results.get('deferral_stats'),
        }
        
        return standardized_results
