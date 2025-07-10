from geopy.distance import geodesic

def assign_tasks(state, current_time):
    """
    Assigns available tasks to available workers using a greedy nearest-neighbor strategy.
    """
    assignments = []

    for task in list(state.active_tasks):  # use list to avoid mutation issues
        best_worker = None
        best_distance = float('inf')

        for worker in state.available_workers:
            distance = geodesic(
                (worker.start_lat, worker.start_lon),
                (task.pickup_lat, task.pickup_lon)
            ).km

            if distance < best_distance:
                best_distance = distance
                best_worker = worker

        if best_worker:
            task.assign_to_worker(best_worker)
            best_worker.assign_task(task)
            state.assign_task(task, best_worker)
            assignments.append((task.id, best_worker.id, best_distance))

    return assignments