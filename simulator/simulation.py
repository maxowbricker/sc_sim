"""
Event-driven Spatial Crowdsourcing simulator.
"""

import pandas as pd
import numpy as np
from heapq import heappush, heappop

from typing import Dict, Optional
from config import get_simulation_config
from simulator.state import StateManager
from simulator.strategies import get_strategy
from metrics.fairness import FairnessMetricsTracker
from metrics.tracker import MetricTracker
from metrics.diagnostic_tracker import DiagnosticTracker

def run_simulation(
    workers,
    tasks,
    start_time=None,
    end_time=None,
    sim_config: Optional[Dict] = None,
):
    """Run an event-driven spatial-crowdsourcing simulation."""

    cfg = get_simulation_config() if sim_config is None else {**get_simulation_config(), **sim_config}
    strategy_params = cfg.get("strategy_params", {})
    
    # Get the strategy handlers from the config
    strategy_name = cfg["assignment_strategy"]
    strategy_handler_factory = get_strategy(strategy_name)
    strategy_handlers = strategy_handler_factory()
    new_task_handler = strategy_handlers["NEW_TASK"]
    free_worker_handler = strategy_handlers["FREE_WORKER"]

    state = StateManager(workers, tasks)
    total_tasks_count = len(tasks)
    event_queue = []
    fairness_tracker = FairnessMetricsTracker()
    metric_tracker = MetricTracker()
    
    # Diagnostic tracker (opt-in for performance, only when explicitly enabled)
    diagnostic_tracker = None
    if strategy_name == "composite" and strategy_params.get('enable_diagnostics', False):
        diagnostic_tracker = DiagnosticTracker()
        strategy_params['diagnostic_tracker'] = diagnostic_tracker
    
    # Fairness cap tracker for FATP-ANN strategy
    if strategy_name == "fatp_ann":
        from simulator.strategies.fatp_ann import FairnessCapTracker
        fairness_cap_tracker = FairnessCapTracker()
        fairness_cap_tracker.initialize(workers)
        strategy_params['fairness_cap_tracker'] = fairness_cap_tracker

    if start_time is None:
        releases = [w.release_time for w in workers] + [t.release_time for t in tasks]
        if not releases: raise ValueError("Cannot infer start_time")
        start_time = min(releases)

    for w in workers:
        heappush(event_queue, (w.release_time, "WORKER_RELEASE", w.id))
    for t in tasks:
        heappush(event_queue, (t.release_time, "TASK_RELEASE", t.id))

    current_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time) if end_time is not None else None

    print(f"Starting event-driven simulation with '{strategy_name}' strategy...")

    summary = {
        'total_tasks': len(tasks),  # Total tasks in simulation
        'completed_tasks': 0, 'total_travel_km': 0.0, 'empty_km': 0.0,
        'passenger_km': 0.0, 'total_wait_min': 0.0, 'wait_times': [],
        'service_times': [], 'backlog_peak': 0,
        'pickup_distances': [], 'assignment_delays': [],
    }

    # Temporal EWMA tracking (for Exp 015)
    ewma_temporal_history = []
    temporal_log_interval = 50  # Log every 50 completed tasks
    next_log_checkpoint = temporal_log_interval

    # Safety counters to prevent infinite loops
    max_events = len(tasks) * 10  # Reasonable upper bound
    event_count = 0
    last_progress_report = 0
    
    while event_queue:
        event_count += 1
        
        # Progress reporting every 10,000 events
        if event_count - last_progress_report >= 10000:
            remaining_events = len(event_queue)
            completed = summary['completed_tasks']
            print(f"   📊 Simulation progress: {completed:,}/{total_tasks_count:,} tasks completed, {remaining_events:,} events remaining")
            last_progress_report = event_count
        
        # Safety check: Prevent infinite loops
        if event_count > max_events:
            print(f"⚠️  Simulation terminated: Exceeded {max_events:,} events (likely pathological parameter combination)")
            print(f"   Completed {summary['completed_tasks']:,}/{total_tasks_count:,} tasks before termination")
            break
            
        event_time, event_type, event_id = heappop(event_queue)
        
        if end_time and event_time > end_time:
            break

        time_delta_seconds = (event_time - current_time).total_seconds()
        if time_delta_seconds > 0:
            for w in state.available_workers:
                w.update_idle_time(time_delta_seconds)
        
        current_time = event_time

        if event_type == "WORKER_RELEASE":
            worker = state.get_worker(event_id)
            state.release_worker(worker)
            assignment = free_worker_handler(state, current_time, worker, **strategy_params)
            if assignment:
                assigned_task, assigned_worker, _ = assignment
                # Track task assignment for supervisor's fairness metrics
                fairness_tracker.record_task_assignment(assigned_task, assigned_worker, current_time)
                heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                # Track assignment delay
                if assigned_task.start_time:
                    assignment_delay = (current_time - assigned_task.release_time).total_seconds()
                    summary['assignment_delays'].append(assignment_delay)

        elif event_type == "TASK_RELEASE":
            task = state.get_task(event_id)
            state.release_task(task)
            
            # Track task eligibility for supervisor's fairness metrics
            fairness_tracker.record_task_release(task, list(state.available_workers), current_time)
            
            if state.available_workers:
                assignments = new_task_handler(state, current_time, [task], **strategy_params)
                if assignments:
                    assigned_task, assigned_worker, _ = assignments[0]
                    # Track task assignment for supervisor's fairness metrics
                    fairness_tracker.record_task_assignment(assigned_task, assigned_worker, current_time)
                    heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                    # Track assignment delay
                    if assigned_task.start_time:
                        assignment_delay = (current_time - assigned_task.release_time).total_seconds()
                        summary['assignment_delays'].append(assignment_delay)
            else:
                # No workers available, so task must wait.
                # The greedy strategy will find it when a worker is freed.
                # The composite strategy will explicitly add it to the deferred pool.
                if strategy_name == "composite":
                    state.defer_task(task)


        elif event_type == "TASK_COMPLETE":
            task = state.get_task(event_id)
            if task.is_completed: continue
            
            worker = task.assigned_worker
            state.complete_task(task, worker, current_time)
            
            assignment = free_worker_handler(state, current_time, worker, **strategy_params)
            if assignment:
                assigned_task, assigned_worker, _ = assignment
                # Track task assignment for supervisor's fairness metrics  
                fairness_tracker.record_task_assignment(assigned_task, assigned_worker, current_time)
                heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
                # Track assignment delay
                if assigned_task.start_time:
                    assignment_delay = (current_time - assigned_task.release_time).total_seconds()
                    summary['assignment_delays'].append(assignment_delay)

            summary['completed_tasks'] += 1
            summary['total_travel_km'] += (task.pickup_km or 0) + (task.drop_km or 0)
            summary['empty_km'] += task.pickup_km or 0
            if task.pickup_km is not None:
                summary['pickup_distances'].append(task.pickup_km)
            summary['passenger_km'] += task.drop_km or 0
            wait_min = (task.start_time - task.release_time).total_seconds()/60 if task.start_time else 0
            summary['total_wait_min'] += wait_min
            summary['wait_times'].append(wait_min)
            service_min = (task.drop_km / 30 * 60) if task.drop_km is not None else 0
            summary['service_times'].append(service_min)
            
            # Log EWMA temporal data at checkpoints (Exp 015)
            if summary['completed_tasks'] >= next_log_checkpoint:
                ewma_values = [w.fairness_ewma for w in state.all_workers_map.values()]
                ewma_temporal_history.append({
                    'timestamp': current_time.isoformat(),
                    'completed_tasks': summary['completed_tasks'],
                    'ewma_mean': float(np.mean(ewma_values)),
                    'ewma_std': float(np.std(ewma_values)),
                    'ewma_p10': float(np.percentile(ewma_values, 10)),
                    'ewma_p50': float(np.percentile(ewma_values, 50)),
                    'ewma_p90': float(np.percentile(ewma_values, 90))
                })
                next_log_checkpoint += temporal_log_interval
        
        current_backlog = len(state.active_tasks) + len(state.deferred_tasks)
        summary['backlog_peak'] = max(summary['backlog_peak'], current_backlog)
        
        # Record fairness metrics every 100 events for performance
        if len(event_queue) % 100 == 0:
            fairness_tracker.update_worker_stats(state.all_workers_map.values())
            fairness_tracker.record_snapshot(current_time)
            # Record enhanced metrics for temporal analysis
            metric_tracker.snapshot(state, current_time)

    print("\nSimulation complete.")
    
    tar = summary['completed_tasks'] / total_tasks_count if total_tasks_count else 0
    avg_travel_km = summary['total_travel_km'] / summary['completed_tasks'] if summary['completed_tasks'] else 0
    avg_wait_min = summary['total_wait_min'] / summary['completed_tasks'] if summary['completed_tasks'] else 0
    wait_p90 = np.percentile(summary['wait_times'], 90) if summary['wait_times'] else 0
    wait_max = max(summary['wait_times']) if summary['wait_times'] else 0
    svc_avg = np.mean(summary['service_times']) if summary['service_times'] else 0
    svc_max = max(summary['service_times']) if summary['service_times'] else 0
    empty_share = summary['empty_km'] / summary['total_travel_km'] if summary['total_travel_km'] else 0
    expired_tasks = total_tasks_count - summary['completed_tasks']

    # Final fairness metrics calculation
    fairness_tracker.update_worker_stats(state.all_workers_map.values())
    fairness_summary = fairness_tracker.get_fairness_summary()
    
    # Add task wait time statistics to summary
    if summary['wait_times']:
        wait_times_array = np.array(summary['wait_times'])
        summary['avg_wait_time_minutes'] = avg_wait_min  # Already computed
        summary['std_wait_time_minutes'] = float(np.std(wait_times_array))
        summary['p90_wait_time_minutes'] = float(wait_p90)  # Already computed
        summary['p95_wait_time_minutes'] = float(np.percentile(wait_times_array, 95))
        summary['max_wait_time_minutes'] = float(wait_max)  # Already computed
        summary['cv_wait_time'] = float(summary['std_wait_time_minutes'] / avg_wait_min) if avg_wait_min > 0 else 0
    else:
        summary['avg_wait_time_minutes'] = 0
        summary['std_wait_time_minutes'] = 0
        summary['p90_wait_time_minutes'] = 0
        summary['p95_wait_time_minutes'] = 0
        summary['max_wait_time_minutes'] = 0
        summary['cv_wait_time'] = 0
    
    # Compute worker idle time statistics
    worker_idle_times = [w.total_idle_time.total_seconds() / 60.0 for w in state.all_workers_map.values()]
    if worker_idle_times:
        summary['mean_worker_idle_time_min'] = float(np.mean(worker_idle_times))
        summary['std_worker_idle_time_min'] = float(np.std(worker_idle_times))
        summary['p90_worker_idle_time_min'] = float(np.percentile(worker_idle_times, 90))
        summary['max_worker_idle_time_min'] = float(np.max(worker_idle_times))
        summary['cv_worker_idle_time'] = float(np.std(worker_idle_times) / np.mean(worker_idle_times)) if np.mean(worker_idle_times) > 0 else 0
    else:
        summary['mean_worker_idle_time_min'] = 0
        summary['std_worker_idle_time_min'] = 0
        summary['p90_worker_idle_time_min'] = 0
        summary['max_worker_idle_time_min'] = 0
        summary['cv_worker_idle_time'] = 0
    
    # Pickup distance distribution statistics
    if summary['pickup_distances']:
        pickup_array = np.array(summary['pickup_distances'])
        summary['std_pickup_distance_km'] = float(np.std(pickup_array))
        summary['p90_pickup_distance_km'] = float(np.percentile(pickup_array, 90))
        summary['max_pickup_distance_km'] = float(np.max(pickup_array))
    else:
        summary['std_pickup_distance_km'] = 0
        summary['p90_pickup_distance_km'] = 0
        summary['max_pickup_distance_km'] = 0
    
    # Assignment delay statistics
    if summary['assignment_delays']:
        delay_array = np.array(summary['assignment_delays'])
        summary['mean_assignment_delay_sec'] = float(np.mean(delay_array))
        summary['std_assignment_delay_sec'] = float(np.std(delay_array))
        summary['p90_assignment_delay_sec'] = float(np.percentile(delay_array, 90))
    else:
        summary['mean_assignment_delay_sec'] = 0
        summary['std_assignment_delay_sec'] = 0
        summary['p90_assignment_delay_sec'] = 0
    
    # Worker task distribution statistics
    tasks_per_worker = [w.completed_tasks for w in state.all_workers_map.values()]
    if tasks_per_worker:
        tasks_array = np.array(tasks_per_worker)
        summary['tasks_per_worker_mean'] = float(np.mean(tasks_array))
        summary['tasks_per_worker_std'] = float(np.std(tasks_array))
        summary['tasks_per_worker_cv'] = float(np.std(tasks_array) / np.mean(tasks_array)) if np.mean(tasks_array) > 0 else 0
        summary['pct_workers_zero_tasks'] = float(sum(1 for t in tasks_per_worker if t == 0) / len(tasks_per_worker))
        summary['pct_workers_single_task'] = float(sum(1 for t in tasks_per_worker if t == 1) / len(tasks_per_worker))
        
        # Gini coefficient
        sorted_tasks = sorted(tasks_per_worker)
        n = len(sorted_tasks)
        if np.sum(sorted_tasks) > 0:
            index_sum = sum((i+1) * t for i, t in enumerate(sorted_tasks))
            summary['tasks_per_worker_gini'] = float((2 * index_sum) / (n * np.sum(sorted_tasks)) - (n + 1) / n)
        else:
            summary['tasks_per_worker_gini'] = 0.0
        
        # Percentiles
        summary['tasks_per_worker_p10'] = float(np.percentile(tasks_array, 10))
        summary['tasks_per_worker_p50'] = float(np.percentile(tasks_array, 50))
        summary['tasks_per_worker_p90'] = float(np.percentile(tasks_array, 90))
    else:
        summary['tasks_per_worker_mean'] = 0
        summary['tasks_per_worker_std'] = 0
        summary['tasks_per_worker_cv'] = 0
        summary['pct_workers_zero_tasks'] = 0
        summary['pct_workers_single_task'] = 0
        summary['tasks_per_worker_gini'] = 0
        summary['tasks_per_worker_p10'] = 0
        summary['tasks_per_worker_p50'] = 0
        summary['tasks_per_worker_p90'] = 0
    
    # Worker utilization statistics
    utilization_rates = []
    for w in state.all_workers_map.values():
        available_time = (w.deadline - w.release_time).total_seconds()
        if available_time > 0:
            busy_time = available_time - w.total_idle_time.total_seconds()
            utilization = max(0.0, min(1.0, busy_time / available_time))  # Clamp to [0,1]
            utilization_rates.append(utilization)

    if utilization_rates:
        util_array = np.array(utilization_rates)
        summary['mean_worker_utilization'] = float(np.mean(util_array))
        summary['std_worker_utilization'] = float(np.std(util_array))
        summary['p10_worker_utilization'] = float(np.percentile(util_array, 10))
        summary['p90_worker_utilization'] = float(np.percentile(util_array, 90))
    else:
        summary['mean_worker_utilization'] = 0
        summary['std_worker_utilization'] = 0
        summary['p10_worker_utilization'] = 0
        summary['p90_worker_utilization'] = 0
    
    # Deferred task statistics
    all_tasks = list(state.all_tasks_map.values())
    deferral_counts = [getattr(t, 'deferral_count', 0) for t in all_tasks]
    tasks_with_deferrals = sum(1 for c in deferral_counts if c > 0)

    summary['total_deferrals'] = int(sum(deferral_counts))
    summary['pct_tasks_deferred'] = float(tasks_with_deferrals / len(all_tasks)) if all_tasks else 0
    summary['mean_deferrals_per_task'] = float(np.mean(deferral_counts)) if deferral_counts else 0
    summary['max_deferrals_per_task'] = int(max(deferral_counts)) if deferral_counts else 0
    
    print("\n---- Simulation Summary ----")
    print(f"Total tasks:           {total_tasks_count}")
    print(f"Completed tasks:       {summary['completed_tasks']}")
    print(f"Task Assignment Ratio: {tar:.2%}")
    print(f"Avg wait time (min):   {avg_wait_min:.1f}")
    print(f"Avg travel distance km:{avg_travel_km:.2f}")
    print(f"P90 wait (min):        {wait_p90:.1f}   max {wait_max:.1f}")
    print(f"Avg service min:       {svc_avg:.1f}   max {svc_max:.1f}")
    print(f"Empty-km share:        {empty_share:.2%}")
    print(f"Peak backlog:          {summary['backlog_peak']}")
    print(f"Expired/unserved:      {expired_tasks}")
    
    if fairness_summary:
        print("\n---- Fairness Metrics ----")
        print(f"Jain's Fairness Index: {fairness_summary.get('final_jains_fairness_index', 0):.3f}")
        print(f"Utility Difference:    {fairness_summary.get('final_utility_difference_tasks', 0):.1f}")
        print(f"Fairness Loss:         {fairness_summary.get('final_fairness_loss', 0):.3f}")
        print(f"EWMA CV:              {fairness_summary.get('final_ewma_cv', 0):.3f}")
        print(f"Mean JFI over time:    {fairness_summary.get('mean_jfi_over_time', 0):.3f}")

    # Combine all metrics for return
    summary.update(fairness_summary)
    summary['metric_tracker'] = metric_tracker  # Include enhanced metrics tracker
    
    # EXPERIMENT 008: Include diagnostic tracker if available
    if diagnostic_tracker:
        summary['diagnostic_tracker'] = diagnostic_tracker
        summary['diagnostic_summary'] = diagnostic_tracker.get_summary_stats()
    
    # EXPERIMENT 015: Add temporal EWMA history
    if ewma_temporal_history:
        summary['ewma_temporal_history'] = ewma_temporal_history
        summary['ewma_final_mean'] = ewma_temporal_history[-1]['ewma_mean']
    
    return summary


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
        }
        
        return standardized_results


if __name__ == "__main__":
    print("Standalone execution not yet implemented for event-driven simulator.")
