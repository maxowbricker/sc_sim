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

        # Print completions
        for task in completed_now:
            worker = task.assigned_worker
            trip_time = (task.finish_time - task.start_time).total_seconds() / 60 if task.start_time else 0
            print(
                f"COMPLETE | Task {task.id} by Worker {worker.id} "
                f"{trip_time:.1f} min → drop ({task.dropoff_lat:.5f}, {task.dropoff_lon:.5f})"
            )

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