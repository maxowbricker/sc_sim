"""
Event-driven Spatial Crowdsourcing simulator.
"""
import numpy as np
from heapq import heappush, heappop
import copy

from typing import Dict, Optional
from config import get_simulation_config
from simulator.state import StateManager
from simulator.strategies import get_strategy
from metrics.manager import MetricsManager
from simulator.spatial_index import set_city_constants

"""
Event-driven Spatial Crowdsourcing simulator.
Highly optimized for Deep Reinforcement Learning (DRL) throughput.
"""

import copy
from heapq import heappush, heappop
from typing import Dict, Optional

import numpy as np

from config import get_simulation_config
from simulator.state import StateManager
from simulator.strategies import get_strategy
from metrics.manager import MetricsManager
from simulator.spatial_index import set_city_constants

class EventSimulator:
    """
    Event-driven simulator that supports step-based execution for RL control.
    """

    @staticmethod
    def _coerce_assignments(result):
        """Strategies may return one (task, worker, _) tuple or a list of tuples."""
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]
    
    def __init__(self, workers, tasks, sim_config: Optional[Dict] = None):
        self.initial_workers = workers
        self.initial_tasks = tasks
        self.sim_config = sim_config or get_simulation_config()
        
        self.strategy_name = self.sim_config["assignment_strategy"]
        self.strategy_params = self.sim_config.get("strategy_params", {})
        
        self._init_strategy_handlers()
        
        self.state = None
        self.event_queue = []
        self.current_time = None
        self.end_time = None
        
        # Lean metrics configuration
        self.metrics = MetricsManager({'strategy_params': self.strategy_params})
        
        self.total_tasks_count = len(tasks)
        self.event_count = 0
        self.max_events = len(tasks) * 10
        self.step_start_time = None

    def _init_strategy_handlers(self):
        """Binds the active strategy handlers."""
        strategy_handler_factory = get_strategy(self.strategy_name)
        strategy_handlers = strategy_handler_factory()
        self.new_task_handler = strategy_handlers["NEW_TASK"]
        self.free_worker_handler = strategy_handlers["FREE_WORKER"]
        
        # Inject standard simulator callbacks into strategy params
        self.strategy_params['expiry_scheduler'] = self._schedule_task_expiry
        
    def _schedule_task_expiry(self, task):
        """O(1) Callback provided to strategies to schedule expirations."""
        if self.current_time < task.expire_time:
            heappush(self.event_queue, (task.expire_time, "TASK_EXPIRE", task.id))

    def reset(self, start_time=None, end_time=None):
        """Resets the simulation to the initial zero-state."""
        current_workers = copy.deepcopy(self.initial_workers)
        current_tasks = copy.deepcopy(self.initial_tasks)
        
        # --- FLAT EARTH SETUP ---
        # Memory-efficient mean calculation avoiding massive array allocations
        total_items = len(current_workers) + len(current_tasks)
        if total_items > 0:
            total_lat = sum(w.start_lat for w in current_workers) + sum(t.pickup_lat for t in current_tasks)
            set_city_constants(total_lat / total_items)
        
        # Reset dynamic tracking variables
        for worker in current_workers:
            worker.total_idle_time = 0.0
            worker.last_state_ts = worker.release_time
            worker.last_active_ts = None
        
        self.state = StateManager(current_workers, current_tasks)
        self.event_queue = []
        self.metrics = MetricsManager({'strategy_params': self.strategy_params})
        
        # Specific tracker injection (e.g. FATP limits)
        if self.strategy_name == "fatp_ann":
            from simulator.strategies.fatp_ann import FairnessCapTracker
            fairness_cap_tracker = FairnessCapTracker()
            fairness_cap_tracker.initialize(current_workers)
            self.strategy_params['fairness_cap_tracker'] = fairness_cap_tracker

        if start_time is None:
            # Generator expression for O(1) memory footprint
            releases = (obj.release_time for seq in (current_workers, current_tasks) for obj in seq)
            try:
                start_time = min(releases)
            except ValueError:
                raise ValueError("Cannot infer start_time: No workers or tasks provided.")

        self.current_time = start_time
        self.end_time = end_time
        self.event_count = 0
        self.step_start_time = None
        
        # Populate initial arrival events
        for w in current_workers:
            heappush(self.event_queue, (w.release_time, "WORKER_RELEASE", w.id))
        for t in current_tasks:
            heappush(self.event_queue, (t.release_time, "TASK_RELEASE", t.id))
        
        return self.get_state()

    def update_weights(self, fairness_weight, starvation_weight, utility_weight=None):
        """Update composite strategy weights. Uses standard names for DRL compatibility."""
        if self.strategy_name == "composite":
            params = {
                'fairness_weight': float(fairness_weight),
                'starvation_weight': float(starvation_weight),
            }
            if utility_weight is not None:
                params['utility_weight'] = float(utility_weight)
            self.strategy_params.update(params)

    def switch_strategy(self, strategy_name: str, strategy_params: Optional[Dict] = None):
        self.strategy_name = strategy_name
        if strategy_params:
            self.strategy_params.update(strategy_params)
            
        self._init_strategy_handlers()
        self.metrics.config['strategy_params'] = self.strategy_params

    def step(self, duration_seconds: float = None) -> bool:
        if self.state is None:
            raise RuntimeError("Simulator not reset. Call reset() first.")
            
        if self.step_start_time is None:
            self.step_start_time = self.current_time
            
        target_time = self.current_time + duration_seconds if duration_seconds else None
        
        while self.event_queue:
            next_event_time = self.event_queue[0][0]
            
            if target_time and next_event_time > target_time:
                self.current_time = target_time
                self.metrics.snapshot_step(self.state, self.current_time, self.step_start_time)
                self.step_start_time = None  
                return False 
            
            self.event_count += 1
            if self.event_count > self.max_events:
                print(f"Simulation terminated: Exceeded {self.max_events:,} events")
                return True
                
            event_time, event_type, event_id = heappop(self.event_queue)
            
            if self.end_time and event_time > self.end_time:
                return True
            
            self.current_time = event_time
            self._process_event(event_type, event_id)
            
        if self.step_start_time:
            self.metrics.snapshot_step(self.state, self.current_time, self.step_start_time)
        
        return True

    def _process_event(self, event_type: str, event_id: int):
        if event_type == "WORKER_RELEASE":
            worker = self.state.get_worker(event_id)
            self.state.release_worker(worker)
            
            assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
            for assigned_task, assigned_worker, _ in self._coerce_assignments(assignment):
                self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

        elif event_type == "TASK_RELEASE":
            task = self.state.get_task(event_id)
            self.state.release_task(task)
            self.metrics.on_task_released(
                task, self.state.available_workers, self.current_time,
                spatial_index=self.state.spatial_index,
            )
            
            if self.state.available_workers:
                assignments = self.new_task_handler(self.state, self.current_time, [task], **self.strategy_params)
                for assigned_task, assigned_worker, _ in self._coerce_assignments(assignments):
                    self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
            else:
                # Strategy agnostic deferral mechanism
                if self.state.defer_task(task, self.current_time):
                    self._schedule_task_expiry(task)
                    self.metrics.on_task_deferred(task, 0.0, "no_candidates", self.current_time)

        elif event_type == "TASK_COMPLETE":
            task = self.state.get_task(event_id)
            if not task.is_completed:
                worker = task.assigned_worker
                self.state.complete_task(task, worker, self.current_time)
                
                assignment = self.free_worker_handler(self.state, self.current_time, worker, **self.strategy_params)
                for assigned_task, assigned_worker, _ in self._coerce_assignments(assignment):
                    self.metrics.on_task_assigned(assigned_task, assigned_worker, self.current_time)
                    heappush(self.event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

                self.metrics.on_task_completed(task, worker, self.current_time)

        elif event_type == "TASK_EXPIRE":
            task = self.state.get_task(event_id)
            if task:
                self.state.remove_deferred_task(task)
                is_assigned = task in self.state.assigned_tasks
                
                if not is_assigned and task not in self.state.completed_tasks and not task.is_completed:
                    # UPDATED: Pass the current_time so we can track the 30-minute window
                    self.metrics.on_task_expired(task.id, self.current_time)

    def get_state(self):
        obs_data = self.metrics.get_observation_data(self.state, self.current_time)
        obs_data.update({
            'active_tasks': len(self.state.active_tasks),
            'assigned_workers': len(self.state.assigned_workers),
        })
        return obs_data

    def get_final_results(self):
        results = self.metrics.get_final_results()
        completed = results.get('completed_tasks', 0)

        # Helpers for safe calculations
        def safe_mean(arr): return float(np.mean(arr)) if arr else 0.0
        def safe_std(arr): return float(np.std(arr)) if arr else 0.0
        def safe_percentile(arr, p): return float(np.percentile(arr, p)) if arr else 0.0
        def safe_max(arr): return float(np.max(arr)) if arr else 0.0

        results['task_assignment_ratio'] = completed / self.total_tasks_count if self.total_tasks_count else 0
        results['total_tasks'] = self.total_tasks_count
        results['avg_wait_time_minutes'] = results['total_wait_min'] / completed if completed else 0
        results['avg_pickup_distance_km'] = (
            results.get('empty_km', 0) / completed if completed > 0 else 0.0
        )
        results['std_wait_time_minutes'] = safe_std(results.get('wait_times', []))
        results['p90_wait_time_minutes'] = safe_percentile(results.get('wait_times', []), 90)
        results['max_wait_time_minutes'] = safe_max(results.get('wait_times', []))

        worker_idle_times = [w.total_idle_time / 60.0 for w in self.state.all_workers_map.values()]
        results['mean_worker_idle_time_min'] = safe_mean(worker_idle_times)

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
        self.config = config
        self.workers_df = workers_df
        self.tasks_df = tasks_df
        self.metric_tracker = None 
        
    def run(self):
        """Run the simulation and return standardized results."""
        from models.worker import Worker
        from models.task import Task
        from simulator.spatial_index import set_city_constants
        
        print("🚀 Initializing Simulation Environment...")
        
        # --- FLAT EARTH SETUP ---
        mean_lat = (self.workers_df['start_lat'].mean() + self.tasks_df['pickup_lat'].mean()) / 2
        set_city_constants(mean_lat)
        print(f"   🌍 Flat-Earth Projection Configured (Mean Lat: {mean_lat:.4f}°)")
        
        # --- FAST VECTORIZED INSTANTIATION ---
        print(f"   📊 Instantiating {len(self.workers_df):,} Workers...")
        workers = [Worker(row) for row in self.workers_df.to_dict('records')]
        
        print(f"   📊 Instantiating {len(self.tasks_df):,} Tasks...")
        tasks = [Task(row) for row in self.tasks_df.to_dict('records')]
        
        print("🚀 Executing Event Loop...")
        results = run_simulation(workers, tasks, sim_config=self.config)
        
        self.metric_tracker = results.get('metric_tracker')
        
        total_tasks = len(tasks)
        completed_tasks = results.get('completed_tasks', 0)
        assigned_tasks = results.get('total_task_assignments_tracked', 0)
        total_travel_km = results.get('total_travel_km', 0)
        
        return {
            'jfi': results.get('final_jains_fairness_index', 0.0),
            'task_assignment_ratio': assigned_tasks / total_tasks if total_tasks > 0 else 0.0,
            'task_completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0.0,
            'avg_wait_time_minutes': results.get('total_wait_min', 0) / completed_tasks if completed_tasks > 0 else 0.0,
            'avg_pickup_distance_km': results.get('empty_km', 0) / completed_tasks if completed_tasks > 0 else 0.0,
            'total_travel_distance_km': total_travel_km,
            'empty_km_ratio': results.get('empty_km', 0) / total_travel_km if total_travel_km > 0 else 0.0,
            'ewma_cv': results.get('final_ewma_cv', 1.0),
            'utility_difference': results.get('final_utility_difference_tasks', 0.0),
            'fairness_loss': results.get('final_fairness_loss', 0.0),
            'total_tasks': total_tasks,
            'assigned_tasks': assigned_tasks,
            'completed_tasks': completed_tasks,
            'max_wait_time': max(results.get('wait_times', [0])) if results.get('wait_times') else 0.0,
            'backlog_peak': results.get('backlog_peak', 0),
            'supervisor_utility_difference': results.get('supervisor_utility_difference'),
            'supervisor_fairness_loss': results.get('supervisor_fairness_loss'),
            'mean_input_output_ratio': results.get('mean_input_output_ratio'),
            'min_input_output_ratio': results.get('min_input_output_ratio'),
            'max_input_output_ratio': results.get('max_input_output_ratio'),
            'workers_with_eligibility_data': results.get('workers_with_eligibility_data', 0),
            'total_task_assignments_tracked': results.get('total_task_assignments_tracked', 0),
            'deferral_stats': results.get('deferral_stats'),
        }