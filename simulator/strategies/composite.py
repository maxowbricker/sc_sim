from simulator.strategies import register
from math import log, fabs, cos, radians
import pandas as pd

AVG_SPEED_KMH = 30


def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

def score(task, worker, λ1, λ2, λ3, now):
    distance = manhattan_km(worker.start_lat, worker.start_lon,
                            task.pickup_lat, task.pickup_lon)
    fairness   = worker.total_idle_time.total_seconds()
    starvation = log(1 + (now - task.release_time).total_seconds())
    return λ1 * fairness + λ2 * starvation - λ3 * distance

@register("composite")
def assign(state, now, λ1=1.0, λ2=1.0, λ3=1.0, **_):
    assignments = []
    for task in list(state.active_tasks):
        best_w = None; best_s = float("-inf")
        for w in state.available_workers:
            s = score(task, w, λ1, λ2, λ3, now)
            if s > best_s:
                best_s, best_w = s, w
        if best_w:
            # Service duration
            pickup_distance = manhattan_km(best_w.start_lat, best_w.start_lon,
                                           task.pickup_lat, task.pickup_lon)
            drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon,
                                         task.dropoff_lat, task.dropoff_lon)
            total_km = pickup_distance + drop_distance
            hours = total_km / AVG_SPEED_KMH
            task.finish_time = pd.Timestamp(now) + pd.to_timedelta(hours, unit="h")
            task.start_time = pd.Timestamp(now)

            task.assign_to_worker(best_w)
            best_w.assign_task(task)
            state.assign_task(task, best_w)
            assignments.append((task.id, best_w.id, best_s))
    return assignments
