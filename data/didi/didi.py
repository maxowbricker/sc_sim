"""
Adapter for Didi Gaia GPS and Order data.

Files:
- gps.txt:    driver_id, order_id, timestamp, lon, lat
- order.txt:  order_id, start_billing, end_billing, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict
import pandas as pd


@dataclass
class Event:
    ts: pd.Timestamp
    type: str
    payload: Dict


class Adapter:
    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser()
        self.gps_df = self._load_gps()
        self.orders_df = self._load_orders()

    # ------------------------------------------------------------------ #
    # Canonical DataFrame export (for timestep simulator)
    # ------------------------------------------------------------------ #

    def to_dataframes(self):
        """Return (workers_df, tasks_df) in canonical format.

        workers_df columns:
            worker_id, start_lat, start_lon, release_time, deadline

        tasks_df columns:
            task_id, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
            release_time, expire_time
        """

        # Workers – first GPS ping per driver
        first_gps = (
            self.gps_df.groupby("driver_id")
            .first()
            .reset_index()[["driver_id", "timestamp", "lon", "lat"]]
            .rename(columns={
                "driver_id": "worker_id",
                "lon": "start_lon",
                "lat": "start_lat",
                "timestamp": "release_time",
            })
        )

        # Heuristic: driver deadline = last known GPS ping + 2h
        last_gps = (
            self.gps_df.groupby("driver_id")["timestamp"].last().reset_index()
            .rename(columns={"timestamp": "last_seen"})
        )
        first_gps = first_gps.merge(last_gps, on="worker_id")
        first_gps["deadline"] = first_gps["last_seen"] + pd.Timedelta("2h")

        workers_df = first_gps[[
            "worker_id",
            "start_lat",
            "start_lon",
            "release_time",
            "deadline",
        ]]

        # Tasks – directly from orders table
        tasks_df = self.orders_df.rename(columns={
            "order_id": "task_id",
            "pickup_lat": "pickup_lat",
            "pickup_lon": "pickup_lon",
            "dropoff_lat": "dropoff_lat",
            "dropoff_lon": "dropoff_lon",
            "start_billing": "release_time",
            "end_billing": "expire_time",
        })[
            [
                "task_id",
                "pickup_lat",
                "pickup_lon",
                "dropoff_lat",
                "dropoff_lon",
                "release_time",
                "expire_time",
            ]
        ]

        return workers_df, tasks_df

    def stream(self, timestep: str = "3s") -> Iterator[List[Event]]:
        events = self._build_events()
        events.sort(key=lambda e: e.ts)

        step = pd.Timedelta(timestep)
        bucket_start = events[0].ts.floor(timestep)
        bucket: List[Event] = []

        for ev in events:
            while ev.ts >= bucket_start + step:
                yield bucket
                bucket = []
                bucket_start += step
            bucket.append(ev)

        if bucket:
            yield bucket

    def _load_gps(self) -> pd.DataFrame:
        path = self.root / "gps.txt"
        df = pd.read_csv(path, header=None)
        df.columns = ["driver_id", "order_id", "timestamp", "lon", "lat"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values("timestamp")

    def _load_orders(self) -> pd.DataFrame:
        path = self.root / "order.txt"
        df = pd.read_csv(path, header=None)
        df.columns = [
            "order_id", "start_billing", "end_billing",
            "pickup_lon", "pickup_lat",
            "dropoff_lon", "dropoff_lat"
        ]
        df["start_billing"] = pd.to_datetime(df["start_billing"], unit="s", utc=True)
        df["end_billing"] = pd.to_datetime(df["end_billing"], unit="s", utc=True)
        return df.sort_values("start_billing")

    def _build_events(self) -> List[Event]:
        events: List[Event] = []

        # Worker join: first GPS ping
        first_gps = (
            self.gps_df.groupby("driver_id")
            .first()
            .reset_index()[["driver_id", "timestamp", "lon", "lat"]]
        )
        events.extend([
            Event(
                ts=row.timestamp,
                type="worker_join",
                payload={
                    "worker_id": row.driver_id,
                    "lon": float(row.lon),
                    "lat": float(row.lat)
                }
            )
            for row in first_gps.itertuples()
        ])

        # GPS pings
        for row in self.gps_df.itertuples():
            events.append(Event(
                ts=row.timestamp,
                type="gps",
                payload={
                    "worker_id": row.driver_id,
                    "lon": float(row.lon),
                    "lat": float(row.lat)
                }
            ))

        # Task releases from orders
        for row in self.orders_df.itertuples():
            events.append(Event(
                ts=row.start_billing,
                type="task_release",
                payload={
                    "task_id": row.order_id,
                    "pickup": (float(row.pickup_lon), float(row.pickup_lat)),
                    "dropoff": (float(row.dropoff_lon), float(row.dropoff_lat)),
                    "deadline": row.end_billing
                }
            ))

        return events
