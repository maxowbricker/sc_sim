"""
Adapter for Gowalla / Weeplaces check‑in logs.

Each CSV row            → one **task_release** event
First sighting per user → one **worker_join** event
Coordinates are already WGS‑84, times are converted to UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict
import pandas as pd


# --------------------------------------------------------------------------- #
# Canonical event dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    ts: pd.Timestamp
    type: str            # "worker_join" | "task_release"
    payload: Dict        # normalised fields (see README / adapter spec)


# --------------------------------------------------------------------------- #
# Adapter implementation
# --------------------------------------------------------------------------- #
class Adapter:
    """
    Parameters
    ----------
    root_path : str
        Directory containing one or more check‑in CSV files with columns
        `user_id, datetime, lat, lon, point_id`.  Header row is optional.
    """

    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser()
        self.df   = self._load_checkins()

    # ------------------------------------------------------------------ #
    # Canonical DataFrame export (for timestep simulator)
    # ------------------------------------------------------------------ #

    def to_dataframes(self):
        """Return (workers_df, tasks_df) in canonical column format.

        workers_df columns:
            worker_id, start_lat, start_lon, release_time, deadline

        tasks_df columns:
            task_id, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            release_time, expire_time
        """

        # Workers – first appearance of each user
        first_seen = (
            self.df.groupby("user_id")
            .first()
            .reset_index()[["user_id", "datetime", "lon", "lat"]]
            .rename(columns={
                "user_id": "worker_id",
                "lon": "start_lon",
                "lat": "start_lat",
                "datetime": "release_time",
            })
        )

        # Simple heuristic: worker stays active for 24h after first sighting
        first_seen["deadline"] = first_seen["release_time"] + pd.Timedelta("1D")

        workers_df = first_seen[[
            "worker_id",
            "start_lat",
            "start_lon",
            "release_time",
            "deadline",
        ]]

        # Tasks – every check-in row becomes a task at that exact location
        tasks_df = (
            self.df.reset_index()
            .rename(columns={
                "index": "idx",
                "lat": "pickup_lat",
                "lon": "pickup_lon",
                "datetime": "release_time",
            })
        )

        tasks_df["task_id"] = "ci_" + tasks_df["idx"].astype(str)
        tasks_df["dropoff_lat"] = tasks_df["pickup_lat"]
        tasks_df["dropoff_lon"] = tasks_df["pickup_lon"]
        tasks_df["expire_time"] = tasks_df["release_time"] + pd.Timedelta("1D")

        tasks_df = tasks_df[[
            "task_id",
            "pickup_lat",
            "pickup_lon",
            "dropoff_lat",
            "dropoff_lon",
            "release_time",
            "expire_time",
        ]]

        return workers_df, tasks_df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def stream(self, timestep: str = "60s") -> Iterator[List[Event]]:
        """
        Yield chronologically ordered *buckets* of Event objects where every
        event timestamp falls inside the half‑open interval
        [bucket_start, bucket_start + timestep).

        Parameters
        ----------
        timestep : str
            A pandas offset alias (“10s”, “1min”, “5min”, …).
        """
        events = self._build_events()
        events.sort(key=lambda e: e.ts)

        step         = pd.Timedelta(timestep)
        bucket_start = events[0].ts.floor(timestep)
        bucket: List[Event] = []

        for ev in events:
            while ev.ts >= bucket_start + step:
                yield bucket
                bucket        = []
                bucket_start += step
            bucket.append(ev)

        if bucket:        # flush final bucket
            yield bucket

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _load_checkins(self) -> pd.DataFrame:
        """
        Load every *.csv file in `self.root` and return a single dataframe
        sorted by timestamp.
        """
        files = list(self.root.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No .csv files found in {self.root}")

        dfs = []
        for path in files:
            df = pd.read_csv(path)
            # If file is header‑less (5 columns), add headers.
            if df.shape[1] == 5:
                df.columns = ["user_id", "datetime", "lat", "lon", "point_id"]

            # Normalise dtypes
            df["datetime"] = pd.to_datetime(
                df["datetime"], utc=True, errors="coerce"
            )
            df = df.dropna(subset=["datetime", "lat", "lon"])
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        return merged.sort_values("datetime")

    def _build_events(self) -> List[Event]:
        """
        Build the list of canonical Event objects:
        - One `worker_join` at first sighting of each user.
        - One `task_release` for every check‑in row.
        """
        events: List[Event] = []

        # 1. worker_join events
        first_seen = (
            self.df.groupby("user_id")
            .first()
            .reset_index()[["user_id", "datetime", "lon", "lat"]]
        )
        events.extend(
            Event(
                ts=row.datetime,
                type="worker_join",
                payload={
                    "worker_id": row.user_id,
                    "lon": float(row.lon),
                    "lat": float(row.lat),
                },
            )
            for row in first_seen.itertuples()
        )

        # 2. task_release events
        for idx, row in self.df.iterrows():
            events.append(
                Event(
                    ts=row["datetime"],
                    type="task_release",
                    payload={
                        "task_id": f"ci_{idx}",
                        "pickup": (float(row["lon"]), float(row["lat"])),
                        "dropoff": (float(row["lon"]), float(row["lat"])),
                        "deadline": None,  # no explicit expiry
                    },
                )
            )

        return events