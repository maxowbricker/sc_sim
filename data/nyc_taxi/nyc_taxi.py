"""
Adapter for NYC TLC Yellow Taxi Trip Records (modern parquet format).

The modern TLC re-release uses location zone IDs (PULocationID / DOLocationID)
instead of raw GPS coordinates. This adapter resolves zone IDs to geographic
centroids using a pre-computed zone_centroids.csv file.

One-time setup (required before first use):
    python data/nyc_taxi/generate_zone_centroids.py

Expected files in root_path (default: data/nyc_taxi/):
    yellow_tripdata_<YYYY-MM>.parquet   — TLC trip records
    zone_centroids.csv                  — LocationID, lat, lon

Workers are synthesised via the bootstrap method: N trips are sampled from
the task pool and their pickup location + time become worker spawn points.
This matches the spatial density of the real demand without requiring driver
GPS tracks (which were removed from the modern TLC data release).

Usage:
    from data.loader import load_workers_tasks

    # Load a single day (recommended for per-day experiments)
    workers, tasks = load_workers_tasks(
        "nyc_taxi",
        root_path="data/nyc_taxi",
        date="2012-05-01",
    )

    # Load with a fixed fleet size instead of proportional
    workers, tasks = load_workers_tasks(
        "nyc_taxi",
        root_path="data/nyc_taxi",
        date="2012-05-01",
        use_proportional_workers=False,
        num_workers=5000,
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class Adapter:
    def __init__(
        self,
        root_path: str,
        date: str | None = None,
        use_proportional_workers: bool = True,
        workers_per_task_ratio: float = 0.2,
        num_workers: int = 5000,
        random_state: int = 42,
    ):
        """
        Args:
            root_path: Directory containing the parquet file and zone_centroids.csv.
            date: ISO date string (e.g. "2012-05-01") to filter to a single day.
                  None loads the entire file (full month).
            use_proportional_workers: If True, fleet size = tasks * workers_per_task_ratio.
                                      If False, a fixed num_workers fleet is bootstrapped.
            workers_per_task_ratio: Ratio applied when use_proportional_workers=True.
            num_workers: Fixed fleet size when use_proportional_workers=False.
            random_state: RNG seed for reproducible bootstrap sampling.
        """
        self.root = Path(root_path).expanduser()
        self.date = date
        self.use_proportional_workers = use_proportional_workers
        self.workers_per_task_ratio = workers_per_task_ratio
        self.num_workers = num_workers
        self.random_state = random_state

        self._zone_centroids = self._load_zone_centroids()
        self._trips_df = self._load_trips()

    # ------------------------------------------------------------------ #
    # Private loaders
    # ------------------------------------------------------------------ #

    def _load_zone_centroids(self) -> dict[int, tuple[float, float]]:
        """Load LocationID → (lat, lon) from zone_centroids.csv."""
        csv_path = self.root / "zone_centroids.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ Zone centroids file not found at '{csv_path}'.\n"
                f"   Run the one-time setup script to generate it:\n\n"
                f"       python data/nyc_taxi/generate_zone_centroids.py\n"
            )
        df = pd.read_csv(csv_path)
        return {
            int(row["LocationID"]): (float(row["lat"]), float(row["lon"]))
            for _, row in df.iterrows()
        }

    def _load_trips(self) -> pd.DataFrame:
        """Load the parquet file and apply date + validity filters."""
        parquet_files = sorted(self.root.glob("yellow_tripdata_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No 'yellow_tripdata_*.parquet' files found in '{self.root}'."
            )

        df = pd.read_parquet(parquet_files[0])

        # Ensure datetime columns are parsed
        for col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Filter to a single calendar day if requested
        if self.date is not None:
            target_date = pd.Timestamp(self.date).date()
            mask = df["tpep_pickup_datetime"].dt.date == target_date
            df = df[mask]

        # Drop rows with null timestamps
        df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

        # Drop rows whose zone IDs don't exist in the centroid lookup
        valid_zones = set(self._zone_centroids.keys())
        df = df[
            df["PULocationID"].isin(valid_zones)
            & df["DOLocationID"].isin(valid_zones)
        ]

        # Sanity: dropoff must be after pickup
        df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]]

        if df.empty:
            raise ValueError(
                f"No valid trips found for date='{self.date}' in '{self.root}'.\n"
                f"Check the date string (expected format: YYYY-MM-DD) and that the "
                f"parquet file covers that date."
            )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Canonical DataFrame export
    # ------------------------------------------------------------------ #

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (workers_df, tasks_df) in canonical simulation format.

        tasks_df columns:  task_id, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                           release_time, expire_time
        workers_df columns: worker_id, start_lat, start_lon, release_time, deadline
        """
        df = self._trips_df.copy()

        # --- ZONE ID → COORDINATES (vectorised) ---
        df["pickup_lat"]  = df["PULocationID"].map(lambda z: self._zone_centroids[z][0])
        df["pickup_lon"]  = df["PULocationID"].map(lambda z: self._zone_centroids[z][1])
        df["dropoff_lat"] = df["DOLocationID"].map(lambda z: self._zone_centroids[z][0])
        df["dropoff_lon"] = df["DOLocationID"].map(lambda z: self._zone_centroids[z][1])

        # --- DATETIME → UNIX TIMESTAMP (seconds) ---
        df["release_time"] = df["tpep_pickup_datetime"].astype("int64") / 1e9
        df["expire_time"]  = df["tpep_dropoff_datetime"].astype("int64") / 1e9

        # --- TASKS ---
        tasks_df = df[[
            "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
            "release_time", "expire_time",
        ]].copy().reset_index(drop=True)
        tasks_df["task_id"] = tasks_df.index
        tasks_df = tasks_df.dropna()

        # --- WORKERS (bootstrap from task pickup locations) ---
        n_workers = (
            max(1, int(len(tasks_df) * self.workers_per_task_ratio))
            if self.use_proportional_workers
            else self.num_workers
        )
        n_workers = min(n_workers, len(tasks_df))

        spawn = tasks_df.sample(n=n_workers, random_state=self.random_state)
        workers_df = spawn[["pickup_lat", "pickup_lon", "release_time"]].copy()
        workers_df = workers_df.rename(
            columns={"pickup_lat": "start_lat", "pickup_lon": "start_lon"}
        )
        workers_df["deadline"] = workers_df["release_time"] + 28800.0
        workers_df = workers_df.reset_index(drop=True)
        workers_df["worker_id"] = workers_df.index

        # --- STRICT FLOAT BOUNDARIES ---
        for col in ("release_time", "deadline"):
            workers_df[col] = workers_df[col].astype(float)
        for col in ("release_time", "expire_time"):
            tasks_df[col] = tasks_df[col].astype(float)

        return workers_df, tasks_df
