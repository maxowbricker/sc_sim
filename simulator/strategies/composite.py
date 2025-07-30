from simulator.strategies import register
from math import log, fabs, cos, radians
import random
from models.worker import Worker
import pandas as pd

AVG_SPEED_KMH = 30


def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

def score(task, worker, λ1, λ2, λ3, now):
    """Composite score = λ1·fairness + λ2·starvation + λ3·utility.

    fairness   – EWMA idle-time seconds (higher → more deserving)
    starvation – log age seconds (higher → more urgent)
    utility    – inverse distance 1/(1+d) km (higher when closer)
    """

    distance = manhattan_km(worker.start_lat, worker.start_lon,
                            task.pickup_lat, task.pickup_lon)

    fairness   = worker.fairness_ewma
    starvation = log(1 + (now - task.release_time).total_seconds())
    utility    = 1.0 / (1.0 + distance)

    return λ1 * fairness + λ2 * starvation + λ3 * utility

@register("composite")
def assign(state, now, λ1=1.0, λ2=1.0, λ3=1.0, k: int = 15, score_filter: float = 0.8, soft_threshold: float = 4.0, **_):
    assignments = []
    # Only consider tasks that are newly active
    unassigned_tasks = [t for t in state.active_tasks if t.worker_id is None and t.start_time is None]

    for task in unassigned_tasks:
        # ------------------------------------------------------------------ #
        # Phase 1 – candidate filtering by proximity + feasibility
        # ------------------------------------------------------------------ #

        drop_distance_const = manhattan_km(task.pickup_lat, task.pickup_lon,
                                           task.dropoff_lat, task.dropoff_lon)

        candidates: list[tuple[float, Worker, float]] = []  # (distance, worker, score placeholder)

        for w in state.available_workers:
            d_pick = manhattan_km(w.start_lat, w.start_lon,
                                  task.pickup_lat, task.pickup_lon)

            # Feasibility: worker must finish before own deadline AND before task expiry
            total_km_tmp = d_pick + drop_distance_const
            finish_eta = pd.Timestamp(now) + pd.to_timedelta(total_km_tmp / AVG_SPEED_KMH, unit="h")

            if finish_eta > w.deadline or finish_eta > task.expire_time:
                continue

            candidates.append((d_pick, w, 0.0))

        # Keep k-nearest candidates
        candidates.sort(key=lambda tup: tup[0])
        if k > 0:
            candidates = candidates[:k]

        if not candidates:
            continue  # no feasible worker

        # ------------------------------------------------------------------ #
        # Phase 2 – composite scoring & fairness filtering
        # ------------------------------------------------------------------ #

        scored: list[tuple[float, Worker]] = []
        best_score = float("-inf")
        for dist, w, _ in candidates:
            s = score(task, w, λ1, λ2, λ3, now)
            scored.append((s, w))
            if s > best_score:
                best_score = s

        # Soft-threshold delay – skip assignment if even best candidate < threshold
        if best_score < soft_threshold:
            continue  # task will wait until score grows (e.g., starvation)

        # Discard candidates with score < score_filter * best_score
        threshold = best_score * score_filter
        scored = [(s, w) for s, w in scored if s >= threshold]

        if not scored:
            continue

        # Choose worker with minimum completed_tasks for fairness; tie-break randomly
        min_comp = min(w.completed_tasks for _, w in scored)
        finalists = [w for _, w in scored if w.completed_tasks == min_comp]
        best_w = random.choice(finalists)

        # For logging/debug, recompute score for chosen worker
        fairness_val = best_w.fairness_ewma
        starvation_val = log(1 + (now - task.release_time).total_seconds())
        distance_val = manhattan_km(best_w.start_lat, best_w.start_lon,
                                   task.pickup_lat, task.pickup_lon)
        utility_val = 1.0 / (1.0 + distance_val)
        best_s = λ1 * fairness_val + λ2 * starvation_val + λ3 * utility_val

        # ------------------------------------------------------------------ #
        # Commit assignment (same as before)
        # ------------------------------------------------------------------ #

        # Service duration
        pickup_distance = manhattan_km(best_w.start_lat, best_w.start_lon,
                                       task.pickup_lat, task.pickup_lon)
        drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon,
                                     task.dropoff_lat, task.dropoff_lon)
        total_km = pickup_distance + drop_distance
        task.pickup_km = pickup_distance
        task.drop_km = drop_distance
        hours = total_km / AVG_SPEED_KMH
        task.finish_time = pd.Timestamp(now) + pd.to_timedelta(hours, unit="h")
        task.start_time = pd.Timestamp(now)

        task.assign_to_worker(best_w)
        best_w.assign_task(task)
        state.assign_task(task, best_w)
        assignments.append((task.id, best_w.id, best_s, utility_val, fairness_val, starvation_val))
    return assignments
