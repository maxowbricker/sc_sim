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
from simulator.task_assignments import assign_tasks
from metrics.tracker import MetricTracker


def run_simulation(
    workers,
    tasks,
    start_time,
    end_time,
    time_step,
    sim_config: dict | None = None,
):
    """Run a discrete-time spatial-crowdsourcing simulation.

    Parameters
    ----------
    workers : list[Worker]
        Pre-initialised Worker objects.
    tasks   : list[Task]
        Pre-initialised Task objects.
    start_time, end_time : str | pd.Timestamp
        Inclusive start and end times of the simulation.
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

    current_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    step = pd.to_timedelta(time_step)

    print("Starting timestep simulation …")
    tick = 0
    while current_time <= end_time:
        print(f"\n--- Tick {tick} | {current_time} ---")

        # 2. Release workers / tasks scheduled for this tick
        state.step(current_time)

        # 3. Assignment
        assignments = assign_tasks(state, current_time)
        for task_id, worker_id, dist in assignments:
            print(f"ASSIGN   | Task {task_id} -> Worker {worker_id} ({dist:.2f} km)")

        # 4. Complete instantly (service_time_mode == "instant")
        if cfg.get("service_time_mode", "instant") == "instant":
            for _tid, _wid, _ in assignments:
                # Lookup objects from IDs
                task_obj = next(t for t in state.assigned_tasks if t.id == _tid)
                worker_obj = next(w for w in state.assigned_workers if w.id == _wid)
                state.complete_task(task_obj, worker_obj, current_time)

        # Metrics snapshot (after assignments + completions of this tick)
        metrics_tracker.snapshot(state, current_time)

        current_time += step
        tick += 1

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