"""
LAF (Least Allocated Worker First) baseline strategy.

Assigns each arriving task to the globally least-loaded feasible worker —
the one with the fewest completed tasks — using an O(|W|) scan of the full
fleet.  Distance is used only as a tie-breaker.

LAF is the structurally simpler predecessor of k-NLF: it applies the same
task-count fairness criterion but without the spatial constraint.  The absence
of a k-NN filter allows it to send workers long distances, increasing pickup
times and wait times relative to k-NLF.

NEW_TASK   → O(|W|) scan → least-loaded feasible worker (distance tie-break)
FREE_WORKER → nearest feasible pending task (greedy, proximity-optimal)
"""

from simulator.strategies import register
from simulator.spatial_index import fast_manhattan_km

AVG_SPEED_KMH = 30


def _commit_assignment(task, worker, now):
    pickup_distance = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                        task.pickup_lat, task.pickup_lon)
    drop_distance = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                      task.dropoff_lat, task.dropoff_lon)
    task.pickup_km = pickup_distance
    task.drop_km = drop_distance
    pickup_travel_hours = pickup_distance / AVG_SPEED_KMH
    service_travel_hours = drop_distance / AVG_SPEED_KMH
    task.start_time = now + (pickup_travel_hours * 3600)
    task.finish_time = task.start_time + (service_travel_hours * 3600)
    task.assign_to_worker(worker)
    worker.assign_task(task)
    return task


def assign_new_tasks_laf(state, now, tasks_to_assign, **_):
    """
    NEW_TASK handler: O(|W|) scan to find the feasible worker with the fewest
    completed tasks.  Pickup distance is used as a tie-breaker.  Tasks with no
    feasible match are left in active_tasks for subsequent FREE_WORKER pickup.
    """
    assignments = []

    for task in tasks_to_assign:
        best_worker = None
        best_task_count = float("inf")
        best_dist = float("inf")

        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                      task.dropoff_lat, task.dropoff_lon)

        for worker in state.available_workers:
            pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                            task.pickup_lat, task.pickup_lon)

            pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
            finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)

            if pickup_eta > task.expire_time or finish_eta > worker.deadline:
                continue

            if worker.completed_tasks < best_task_count:
                best_task_count = worker.completed_tasks
                best_worker = worker
                best_dist = pickup_dist
            elif worker.completed_tasks == best_task_count and pickup_dist < best_dist:
                best_worker = worker
                best_dist = pickup_dist

        if best_worker:
            assigned_task = _commit_assignment(task, best_worker, now)
            state.assign_task(assigned_task, best_worker)
            assignments.append((assigned_task, best_worker, best_dist))

    return assignments


def match_worker_laf(state, now, worker, **_):
    """
    LAF matching when a worker becomes available.

    Design Decision: Worker-side matching uses GREEDY (nearest task) approach.
    Rationale: LAF enforces fairness when tasks choose workers (NEW_TASK event),
    not when workers choose tasks. On worker release, spatial efficiency is acceptable.

    Scans both deferred_tasks (tasks deferred by the simulator when no workers were
    available at arrival) and active_tasks (tasks that arrived while workers existed
    but found no feasible match) to ensure no pending task is missed.
    """
    pending = list(state.deferred_tasks) + list(state.active_tasks)
    if not pending:
        return None

    best_task = None
    best_dist = float("inf")

    for task in pending:
        pickup_dist = fast_manhattan_km(worker.start_lat, worker.start_lon,
                                        task.pickup_lat, task.pickup_lon)

        drop_dist = fast_manhattan_km(task.pickup_lat, task.pickup_lon,
                                      task.dropoff_lat, task.dropoff_lon)
        pickup_eta = now + ((pickup_dist / AVG_SPEED_KMH) * 3600)
        finish_eta = now + (((pickup_dist + drop_dist) / AVG_SPEED_KMH) * 3600)

        if pickup_eta > task.expire_time or finish_eta > worker.deadline:
            continue

        if pickup_dist < best_dist:
            best_dist = pickup_dist
            best_task = task

    if best_task:
        assigned_task = _commit_assignment(best_task, worker, now)
        state.assign_task(assigned_task, worker)
        return (assigned_task, worker, best_dist)

    return None


@register("laf")
def get_laf_handlers():
    return {
        "NEW_TASK": assign_new_tasks_laf,
        "FREE_WORKER": match_worker_laf
    }

