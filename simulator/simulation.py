"""
Event-driven Spatial Crowdsourcing simulator.
"""

import pandas as pd
import numpy as np
from heapq import heappush, heappop

from config import SIM_CONFIG
from simulator.state import StateManager
from simulator.strategies import get_strategy
from metrics.fairness import FairnessMetricsTracker

def run_simulation(
    workers,
    tasks,
    start_time=None,
    end_time=None,
    sim_config: dict | None = None,
):
    """Run an event-driven spatial-crowdsourcing simulation."""

    cfg = SIM_CONFIG if sim_config is None else {**SIM_CONFIG, **sim_config}
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
        'completed_tasks': 0, 'total_travel_km': 0.0, 'empty_km': 0.0,
        'passenger_km': 0.0, 'total_wait_min': 0.0, 'wait_times': [],
        'service_times': [], 'backlog_peak': 0,
    }

    while event_queue:
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
                assigned_task, _, _ = assignment
                heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

        elif event_type == "TASK_RELEASE":
            task = state.get_task(event_id)
            state.release_task(task)
            if state.available_workers:
                assignments = new_task_handler(state, current_time, [task], **strategy_params)
                if assignments:
                    assigned_task, _, _ = assignments[0]
                    heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))
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
                assigned_task, _, _ = assignment
                heappush(event_queue, (assigned_task.finish_time, "TASK_COMPLETE", assigned_task.id))

            summary['completed_tasks'] += 1
            summary['total_travel_km'] += (task.pickup_km or 0) + (task.drop_km or 0)
            summary['empty_km'] += task.pickup_km or 0
            summary['passenger_km'] += task.drop_km or 0
            wait_min = (task.start_time - task.release_time).total_seconds()/60 if task.start_time else 0
            summary['total_wait_min'] += wait_min
            summary['wait_times'].append(wait_min)
            service_min = (task.drop_km / 30 * 60) if task.drop_km is not None else 0
            summary['service_times'].append(service_min)
        
        current_backlog = len(state.active_tasks) + len(state.deferred_tasks)
        summary['backlog_peak'] = max(summary['backlog_peak'], current_backlog)
        
        # Record fairness metrics every 100 events for performance
        if len(event_queue) % 100 == 0:
            fairness_tracker.update_worker_stats(state.all_workers_map.values())
            fairness_tracker.record_snapshot(current_time)

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
    return summary

if __name__ == "__main__":
    print("Standalone execution not yet implemented for event-driven simulator.")
