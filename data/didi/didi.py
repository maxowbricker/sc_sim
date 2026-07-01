"""
Adapter for Didi Gaia GPS and Order data.
Provides fast, vectorized conversion from raw logs to Simulation DataFrames.

Files:
- gps.txt:    driver_id, order_id, timestamp, lon, lat
- order.txt:  order_id, start_billing, end_billing, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


class Adapter:
    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser()
        self.gps_df = self._load_gps()
        self.orders_df = self._load_orders()

    def _load_gps(self) -> pd.DataFrame:
        """Load raw GPS data"""
        gps_path = self.root / "gps.txt"
        if not gps_path.exists() and (self.root / "gps").exists():
            gps_path = self.root / "gps"  # Fallback for no extension
            
        return pd.read_csv(
            gps_path,
            names=["driver_id", "order_id", "timestamp", "lon", "lat"]
        )

    def _load_orders(self) -> pd.DataFrame:
        """Load raw Orders data"""
        order_path = self.root / "order.txt"
        if not order_path.exists() and (self.root / "order").exists():
            order_path = self.root / "order" # Fallback for no extension
            
        return pd.read_csv(
            order_path,
            names=[
                "order_id", "start_billing", "end_billing", 
                "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"
            ]
        )

    # ------------------------------------------------------------------ #
    # Canonical DataFrame export (for timestep simulator)
    # ------------------------------------------------------------------ #

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (workers_df, tasks_df) in canonical format.
        Uses highly optimized C-level Pandas operations.
        """
        
        # 1. FAST VECTORIZED WORKER EXTRACTION
        # drop_duplicates is exponentially faster than groupby().first() on large DataFrames
        first_gps = (
            self.gps_df.drop_duplicates(subset=["driver_id"], keep="first")
            [["driver_id", "timestamp", "lon", "lat"]]
        )
        
        workers_df = first_gps.rename(columns={
            "driver_id": "worker_id",
            "lat": "start_lat",
            "lon": "start_lon",
            "timestamp": "release_time"
        }).copy()
        
        # Assuming an 8-hour shift (28,800 seconds) if deadline isn't natively in the GPS data
        workers_df['deadline'] = workers_df['release_time'] + 28800.0

        # 2. FAST VECTORIZED TASK EXTRACTION
        tasks_df = self.orders_df.rename(columns={
            "order_id": "task_id",
            "pickup_lon": "pickup_lon",
            "pickup_lat": "pickup_lat",
            "dropoff_lon": "dropoff_lon",
            "dropoff_lat": "dropoff_lat",
            "start_billing": "release_time",
            "end_billing": "expire_time" 
        }).copy()

        # 3. STRICT FLOAT BOUNDARY
        # Ensure all timestamps are strict floats for the event-driven simulator
        for col in ['release_time', 'deadline']:
            if col in workers_df.columns:
                workers_df[col] = workers_df[col].astype(float)
                
        for col in ['release_time', 'expire_time']:
            if col in tasks_df.columns:
                tasks_df[col] = tasks_df[col].astype(float)

        return workers_df, tasks_df