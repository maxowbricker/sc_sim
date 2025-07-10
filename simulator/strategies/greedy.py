from simulator.strategies import register
import pandas as pd
from math import fabs, cos, radians

AVG_SPEED_KMH = 30  # simple constant speed for service time


def manhattan_km(lat1, lon1, lat2, lon2):
    km_per_deg = 111  # approx km in one degree lat
    d_lat = fabs(lat1 - lat2) * km_per_deg
    avg_lat = (lat1 + lat2) / 2
    d_lon = fabs(lon1 - lon2) * km_per_deg * cos(radians(avg_lat))
    return d_lat + d_lon

@register("greedy")
def assign(state, now, **_):
    assignments = []
    for task in list(state.active_tasks):
        best_w = None; best_d = float("inf")
        for w in state.available_workers:
            d = manhattan_km(w.start_lat, w.start_lon,
                             task.pickup_lat, task.pickup_lon)
            if d < best_d:
                best_d, best_w = d, w
        if best_w:
            # Compute service duration: worker→pickup plus pickup→dropoff
            pickup_distance = best_d  # already computed (km)
            drop_distance = manhattan_km(task.pickup_lat, task.pickup_lon,
                                         task.dropoff_lat, task.dropoff_lon)
            total_km = pickup_distance + drop_distance
            hours = total_km / AVG_SPEED_KMH
            task.finish_time = pd.Timestamp(now) + pd.to_timedelta(hours, unit="h")
            task.start_time = pd.Timestamp(now)

            task.assign_to_worker(best_w)
            best_w.assign_task(task)
            state.assign_task(task, best_w)
            assignments.append((task.id, best_w.id, best_d))
    return assignments
