from simulator.state import StateManager
from data.adapters.checkin import Event


def run_simulation(adapter, config=None):
    """
    Run the timestep-based SC simulation using a streaming event adapter.

    Parameters
    ----------
    adapter : Adapter
        Must implement .stream(timestep) and yield List[Event].
    config : dict (optional)
        Reserved for future config extensions.
    """
    print("Starting simulation...\n")
    state = StateManager()

    for i, bucket in enumerate(adapter.stream()):
        print(f"\n--- Timestep {i} ---")
        for event in bucket:
            print(f"{event.ts} | {event.type:12} | {event.payload}")

            if event.type == "worker_join":
                state.register_worker(event.payload)

            elif event.type == "task_release":
                state.register_task(event.payload)

            elif event.type == "gps":
                state.update_worker_position(event.payload)

        # Placeholder for task assignment and metrics
        # assign_tasks(state)
        # update_metrics(state)

    print("\nSimulation complete.")