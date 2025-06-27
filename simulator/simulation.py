from data.loader import load_workers, load_tasks
from simulator.state import StateManager

def run_simulation(config):
    """
    Runs the timestep-based spatial crowdsourcing simulation.
    """
    # Load initial data
    workers = load_workers(config["worker_file"])
    tasks = load_tasks(config["task_file"])
    state = StateManager(workers, tasks)

    current_time = config["start_time"]
    end_time = config["end_time"]
    step = config["time_step"]

    print("Starting simulation...")
    
    while current_time <= end_time:
        print(f"\n--- Timestep: {current_time} ---")

        # 1. Update state for this timestep (release tasks and workers)
        state.step(current_time)

        # 2. Perform task assignment
        # assign_tasks(current_time)

        # 3. Update state and track metrics
        # update_system_state(current_time)
        # log_metrics(current_time)

        current_time += step

    print("Simulation complete.")