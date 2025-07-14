"""
Timestep‑based Spatial Crowdsourcing simulator.

This version iterates from START_TIME to END_TIME in fixed TIME_STEP
increments.  At each tick it:
    1. releases workers and tasks whose release_time has arrived
    2. runs greedy assignment (or any pluggable strategy)
    3. instantly completes tasks (service_time_mode == "instant")
    4. (placeholder) updates metrics / logging
"""

import pandas as pd
# Global SIM_CONFIG still holds generic flags (service_time_mode, teleport, etc.)
# but data ingestion & time-range parameters are now supplied by the caller.
from config import SIM_CONFIG
# NOTE: Data reading is no longer done inside this file.  Provide pre-constructed
# lists of `Worker` and `Task` objects when calling `run_simulation()`.
from simulator.state import StateManager
from metrics.tracker import MetricTracker
from simulator.strategies import get_strategy


def run_simulation(
    workers,
    tasks,
    start_time=None,
    end_time=None,
    time_step="5min",
    sim_config: dict | None = None,
):
    """Run a discrete-time spatial-crowdsourcing simulation.

    Parameters
    ----------
    workers : list[Worker]
        Pre-initialised Worker objects.
    tasks   : list[Task]
        Pre-initialised Task objects.
    start_time, end_time : str | pd.Timestamp | None
        If *None*, the simulator will infer the earliest release time across
        all workers/tasks as the start, and will run until the system clears
        (no more tasks pending or to be released).
    time_step : str | pd.Timedelta
        Simulation tick size, e.g. "5min" or "3s".
    sim_config : dict, optional
        Extra flags ("service_time_mode", "teleport_on_complete", …).  If
        omitted the global ``SIM_CONFIG`` is used.
    """

    cfg = SIM_CONFIG if sim_config is None else {**SIM_CONFIG, **sim_config}

    # 1.  Initialise state
    state = StateManager(workers, tasks)
    total_tasks_count = len(tasks)

    # Metrics collector
    metrics_tracker = MetricTracker()

    # ------------------------------------------------------------------
    # Determine clock parameters
    # ------------------------------------------------------------------

    # Handle None → infer from data
    if start_time is None:
        releases = [w.release_time for w in workers] + [t.release_time for t in tasks]
        if not releases:
            raise ValueError("Cannot infer start_time: empty workers and tasks list")
        start_time = min(releases)

    current_time = pd.to_datetime(start_time)

    # Optional user-provided end_time; otherwise we'll stop when system empty
    end_time = pd.to_datetime(end_time) if end_time is not None else None

    step = pd.to_timedelta(time_step)

    print("Starting timestep simulation …")
    tick = 0
    while True:
        print(f"\n--- Tick {tick} | {current_time} ---")

        # 2. Release / update state, capture completions
        prev_completed = len(state.completed_tasks)
        state.step(current_time)
        completed_now = state.completed_tasks[prev_completed:]

        # accumulate for summary
        if 'summary' not in locals():
            summary = {
                'completed_tasks': 0,
                'total_travel_km': 0.0,
                'empty_km': 0.0,
                'passenger_km': 0.0,
                'total_wait_min': 0.0,
                'wait_times': [],
                'service_times': [],
                'backlog_peak': 0,
                'backlog_total': 0,
            }

        # 3. Assignment
        strategy = get_strategy(cfg["assignment_strategy"])
        assignments = strategy(state, current_time, **cfg.get("strategy_params", {}))
        for task_id, worker_id, metric in assignments:
            print(f"ASSIGN   | Task {task_id} -> Worker {worker_id} (metric={metric:.2f})")

        # 4. Complete instantly (service_time_mode == "instant")
        if cfg.get("service_time_mode", "instant") == "instant":
            for _tid, _wid, _ in assignments:
                # Lookup objects from IDs
                task_obj = next(t for t in state.assigned_tasks if t.id == _tid)
                worker_obj = next(w for w in state.assigned_workers if w.id == _wid)
                state.complete_task(task_obj, worker_obj, current_time)

        # Track backlog stats
        current_backlog = len(state.active_tasks)
        summary['backlog_peak'] = max(summary['backlog_peak'], current_backlog)
        summary['backlog_total'] += current_backlog

        # Print completions
        for task in completed_now:
            worker = task.assigned_worker
            trip_time = (task.finish_time - task.start_time).total_seconds() / 60 if task.start_time else 0
            pickup_min = (task.pickup_km / 30) * 60 if task.pickup_km is not None else 0
            service_min = (task.drop_km / 30) * 60 if task.drop_km is not None else 0
            print(
                f"COMPLETE | Task {task.id} by Worker {worker.id} "
                f"pickup {pickup_min:.1f} min + service {service_min:.1f} min → "
                f"drop ({task.dropoff_lat:.5f}, {task.dropoff_lon:.5f})"
            )

        # update summary stats
        for task in completed_now:
            summary['completed_tasks'] += 1
            summary['total_travel_km'] += (task.pickup_km or 0) + (task.drop_km or 0)
            summary['empty_km'] += task.pickup_km or 0
            summary['passenger_km'] += task.drop_km or 0
            wait_min = (task.start_time - task.release_time).total_seconds()/60 if task.start_time else 0
            summary['total_wait_min'] += wait_min
            summary['wait_times'].append(wait_min)
            service_min = (task.drop_km or 0)/30*60
            summary['service_times'].append(service_min)

        # Metrics snapshot (after assignments + completions of this tick)
        metrics_tracker.snapshot(state, current_time)

        current_time += step
        tick += 1

        # Termination condition if no explicit end_time: stop when nothing left
        if end_time is None:
            no_tasks_left = not (
                state.all_tasks or state.active_tasks or state.assigned_tasks
            )
            if no_tasks_left:
                break
        else:
            if current_time > end_time:
                break

    print("\nSimulation complete.")

    # Export metrics for later analysis (optional)
    try:
        metrics_tracker.save_csv("metrics_snapshot.csv")
        print("Metrics written to metrics_snapshot.csv")
    except Exception as e:
        print(f"Warning: failed to save metrics CSV – {e}")

    # Print summary
    sim_minutes = tick * (pd.to_timedelta(time_step).total_seconds()/60)
    tar = summary['completed_tasks']/total_tasks_count if total_tasks_count else 0
    avg_travel_km = summary['total_travel_km']/summary['completed_tasks'] if summary['completed_tasks'] else 0
    avg_wait_min = summary['total_wait_min']/summary['completed_tasks'] if summary['completed_tasks'] else 0
    import numpy as _np
    wait_p90 = _np.percentile(summary['wait_times'],90) if summary['wait_times'] else 0
    wait_max = max(summary['wait_times']) if summary['wait_times'] else 0
    svc_avg = _np.mean(summary['service_times']) if summary['service_times'] else 0
    svc_max = max(summary['service_times']) if summary['service_times'] else 0
    empty_share = summary['empty_km']/summary['total_travel_km'] if summary['total_travel_km'] else 0
    avg_backlog = summary['backlog_total']/tick if tick else 0
    expired_tasks = total_tasks_count - summary['completed_tasks']

    print("\n---- Simulation Summary ----")
    print(f"Total tasks:           {total_tasks_count}")
    print(f"Completed tasks:       {summary['completed_tasks']}")
    print(f"Task Assignment Ratio: {tar:.2%}")
    print(f"Simulated minutes:     {sim_minutes:.1f}")
    print(f"Avg wait time (min):   {avg_wait_min:.1f}")
    print(f"Avg travel distance km:{avg_travel_km:.2f}")
    print(f"P90 wait (min):        {wait_p90:.1f}   max {wait_max:.1f}")
    print(f"Avg service min:       {svc_avg:.1f}   max {svc_max:.1f}")
    print(f"Empty-km share:        {empty_share:.2%}")
    print(f"Peak backlog:          {summary['backlog_peak']}")
    print(f"Avg backlog:           {avg_backlog:.1f}")
    print(f"Expired/unserved:      {expired_tasks}")


if __name__ == "__main__":
    # Minimal standalone demo: falls back to legacy CSV paths defined in
    # SIM_CONFIG.  For production runs prefer calling ``run_simulation`` from
    # an external script after loading data via ``data.loader.load_workers_tasks``.

    required_keys = {"worker_file", "task_file", "start_time", "end_time", "time_step"}
    if required_keys.issubset(SIM_CONFIG):
        from data.loader import load_workers, load_tasks

        workers = load_workers(SIM_CONFIG["worker_file"])
        tasks   = load_tasks(SIM_CONFIG["task_file"])

        run_simulation(
            workers,
            tasks,
            SIM_CONFIG["start_time"],
            SIM_CONFIG["end_time"],
            SIM_CONFIG["time_step"],
        )
    else:
        raise RuntimeError(
            "SIM_CONFIG missing required keys for standalone execution. "
            "Call run_simulation() from your own script with pre-loaded data instead."
        )