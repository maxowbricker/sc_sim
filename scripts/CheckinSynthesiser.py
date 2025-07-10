import pandas as pd
from geopy.distance import geodesic

def load_weeplaces_checkins(file_path):
    """
    Load the weeplaces check-in data from a .txt file.

    Args:
        file_path (str): Path to the weeplaces check-in file.

    Returns:
        pd.DataFrame: DataFrame with columns [user_id, datetime, latitude, longitude, point_id].
    """
    df = pd.read_csv(file_path, header=None, names=["user_id", "datetime", "latitude", "longitude", "point_id"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df

def generate_workers_from_checkins(checkins):
    """
    Generate worker dataset from check-in data.

    Args:
        checkins (pd.DataFrame): Check-in data with columns [user_id, datetime, latitude, longitude, point_id].

    Returns:
        pd.DataFrame: Worker dataset with columns:
            [worker_id, release_time, start_lat, start_lon, deadline]
    """
    workers = []
    worker_id = 0
    for user_id, group in checkins.groupby("user_id"):
        group = group.sort_values("datetime").reset_index(drop=True)
        for i in range(len(group) - 1):
            a = group.loc[i]
            b = group.loc[i + 1]
            duration = (b["datetime"] - a["datetime"]).total_seconds() / 60
            distance_km = geodesic((a["latitude"], a["longitude"]), (b["latitude"], b["longitude"])).km

            if duration < 5 or duration > 720:
                continue  # Skip unrealistically short or long availability windows
            if distance_km == 0:
                continue  # Skip check-in pairs with no movement

            workers.append({
                "worker_id": f"w{worker_id}",
                "release_time": a["datetime"],
                "start_lat": a["latitude"],
                "start_lon": a["longitude"],
                "deadline": b["datetime"]
            })
            worker_id += 1
    return pd.DataFrame(workers)

def generate_tasks_from_checkins(df):
    """
    Generate task dataset from check-ins at the same location.

    Args:
        df (pd.DataFrame): Check-in data with columns [user_id, datetime, latitude, longitude, point_id].

    Returns:
        pd.DataFrame: Task dataset with columns:
            [task_id, release_time, expire_time, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    tasks = []
    task_id = 0

    for location_id, group in df.groupby("point_id"):
        group = group.sort_values("datetime").reset_index(drop=True)
        for i in range(len(group) - 1):
            a = group.loc[i]
            b = group.loc[i + 1]
            if a["user_id"] != b["user_id"] and a["datetime"] < b["datetime"]:
                # NOTE:
                # For simplicity, we assume tasks are completed at the pickup location (pickup == dropoff).
                # This models SC tasks like verification or photography where no relocation is needed.
                # If simulating worker relocation or delivery-style tasks in the future, use:
                # "dropoff_lat": b["latitude"], "dropoff_lon": b["longitude"]
                tasks.append({
                    "task_id": f"t{task_id}",
                    "release_time": a["datetime"],
                    "expire_time": b["datetime"],
                    "pickup_lat": a["latitude"],
                    "pickup_lon": a["longitude"],
                    "dropoff_lat": a["latitude"],
                    "dropoff_lon": a["longitude"]
                })
                task_id += 1

    return pd.DataFrame(tasks)

# Example usage
if __name__ == "__main__":
    file_path = "weeplacesCheckins.txt"
    weeplaces_df = load_weeplaces_checkins(file_path)
    workers_df = generate_workers_from_checkins(weeplaces_df)
    print(workers_df.head())
    workers_df.to_csv("workers.txt", index=False)

    # Load and process Gowalla data
    gowalla_df = pd.read_csv("gowallaCheckins.txt", header=None, names=["user_id", "datetime", "latitude", "longitude", "point_id"])
    gowalla_df["datetime"] = pd.to_datetime(gowalla_df["datetime"], utc=True)

    tasks_df = generate_tasks_from_checkins(gowalla_df)
    print(tasks_df.head())
    tasks_df.to_csv("tasks.txt", index=False)