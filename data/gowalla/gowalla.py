"""
Adapter for the Stanford SNAP Gowalla Location-Based Social Network dataset.

Generates both Task and Worker objects from a single LBSN check-in file,
following the methodology of Zhang et al. (FATP) and Lan et al. (FATP-ANN).

Dataset source:
    https://snap.stanford.edu/data/loc-gowalla.html
    File: loc-gowalla_totalCheckins.txt.gz
    Format: user_id <TAB> ISO_timestamp <TAB> lat <TAB> lon <TAB> location_id

Task modes (task_mode parameter):
    "checkin" (default, recommended):
        Every individual check-in becomes one task.
          - pickup   = check-in location
          - release_time = check-in timestamp
          - expire_time  = release_time + task_window_hours * 3600
          - dropoff  = pickup + small Gaussian noise (~dropoff_noise_km)
        Produces many tasks and a realistic worker:task ratio.

    "location_pair" (paper-faithful):
        A task is created for each (location_id, day) that has >= 2 check-ins.
          - release_time = earliest check-in at that location that day
          - expire_time  = latest   check-in at that location that day
        Produces far fewer tasks than workers — inverted ratio — only useful
        when comparing directly to the FATP paper's exact numbers.

Worker generation (same for both modes):
    For each (user_id, calendar_day), the user's first check-in of that day
    defines their spawn: start_lat, start_lon, release_time.
    deadline = release_time + shift_hours * 3600.
    Workers are then downsampled to workers_per_task_ratio * n_tasks so the
    supply/demand ratio stays in the expected 1:3 to 1:10 range.

Temporal compression (compress_to_day=True):
    LBSN check-in rates are ~150x lower than ride-hailing.  Over a 29-day
    month only ~31 tasks are active simultaneously — too sparse for strategies
    to differentiate.  compress_to_day strips the calendar date from every
    check-in (keeping only HH:MM:SS) and maps all events onto a single
    reference day, stacking the month's check-ins into 24 hours.

    Before: 43k tasks spread over 29 days  ->  ~31 concurrent tasks
    After:  43k tasks within 24 hours      ->  ~910 concurrent tasks

    This preserves the real intra-day rhythm (morning/evening patterns) while
    creating the worker-task density needed for meaningful strategy comparison.
    Strongly recommended when using multi-day date windows.

Built-in city presets (region parameter):
    "austin"        — densest Gowalla cluster (lat 29.9-30.7, lon -98.1 to -97.5)
    "san_francisco" — second densest cluster  (lat 37.6-37.9, lon -122.6 to -122.3)

Usage:
    from data.loader import load_workers_tasks

    # Default: Austin, checkin mode, 1:5 worker:task ratio
    workers, tasks = load_workers_tasks("gowalla", root_path="data/gowalla")

    # Paper-faithful location-pair mode (inverted ratio — expect many more workers)
    workers, tasks = load_workers_tasks(
        "gowalla", root_path="data/gowalla",
        task_mode="location_pair",
    )

    # San Francisco, narrow date window
    workers, tasks = load_workers_tasks(
        "gowalla",
        root_path="data/gowalla",
        region="san_francisco",
        date_start="2010-01-01",
        date_end="2010-06-30",
    )
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Built-in city bounding boxes  (lat_min, lat_max, lon_min, lon_max)
# ---------------------------------------------------------------------------
CITY_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "austin":        (29.9,  30.7, -98.1, -97.5),
    "san_francisco": (37.6,  37.9, -122.6, -122.3),
}


class Adapter:
    def __init__(
        self,
        root_path: str,
        region: str | None = "austin",
        bbox: tuple[float, float, float, float] | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        task_mode: str = "checkin",
        task_window_hours: float = 0.5,
        shift_hours: float = 8.0,
        dropoff_noise_km: float = 2.0,
        workers_per_task_ratio: float = 0.2,
        compress_to_day: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            root_path: Directory containing loc-gowalla_totalCheckins.txt.gz.
            region: Named city preset ("austin" | "san_francisco"). Ignored
                    when bbox is provided.
            bbox: Custom bounding box (lat_min, lat_max, lon_min, lon_max).
                  Overrides region. Pass None with region=None for global data.
            date_start: ISO date (e.g. "2010-01-01"). Keep check-ins on/after.
            date_end:   ISO date (e.g. "2010-12-31"). Keep check-ins on/before.
            task_mode: "checkin" — every check-in = 1 task (recommended).
                       "location_pair" — same-location pairing as in the paper.
            task_window_hours: Expiry window for "checkin" mode tasks.
                               Ignored in "location_pair" mode.
            shift_hours: Worker shift length (deadline = release + shift * 3600).
            dropoff_noise_km: Std-dev radius for synthetic dropoff displacement.
            workers_per_task_ratio: Workers = max(1, round(n_tasks * ratio)).
                                    Set to None to keep all (user, day) workers.
            compress_to_day: If True (default), strip the calendar date from every
                             check-in and map all events onto a single 24-hour
                             reference window.  This stacks a multi-day dataset
                             into one simulated day, creating the worker-task
                             density needed for meaningful strategy comparison.
                             Strongly recommended when date_start/date_end span
                             more than one day.
            random_state: RNG seed for reproducible sampling and dropoff noise.
        """
        self.root = Path(root_path).expanduser()
        self.task_mode = task_mode
        self.task_window_hours = task_window_hours
        self.shift_hours = shift_hours
        self.dropoff_noise_km = dropoff_noise_km
        self.workers_per_task_ratio = workers_per_task_ratio
        self.compress_to_day = compress_to_day
        self.random_state = random_state

        if task_mode not in ("checkin", "location_pair"):
            raise ValueError(
                f"Unknown task_mode '{task_mode}'. "
                "Choose 'checkin' (recommended) or 'location_pair'."
            )

        # Resolve bounding box
        if bbox is not None:
            self.bbox: tuple[float, float, float, float] | None = bbox
        elif region is not None:
            if region not in CITY_BBOXES:
                raise ValueError(
                    f"Unknown region '{region}'. "
                    f"Available: {list(CITY_BBOXES.keys())}. "
                    f"Or pass a custom bbox tuple."
                )
            self.bbox = CITY_BBOXES[region]
        else:
            self.bbox = None

        self.date_start = pd.Timestamp(date_start, tz="UTC") if date_start else None
        self.date_end   = pd.Timestamp(date_end,   tz="UTC") if date_end   else None

        self._checkins = self._load_checkins()

    # ------------------------------------------------------------------ #
    # Private loader
    # ------------------------------------------------------------------ #

    def _load_checkins(self) -> pd.DataFrame:
        checkin_path = self.root / "loc-gowalla_totalCheckins.txt.gz"
        if not checkin_path.exists():
            raise FileNotFoundError(
                f"Check-in file not found at '{checkin_path}'.\n"
                f"Download from: https://snap.stanford.edu/data/loc-gowalla.html"
            )

        df = pd.read_csv(
            checkin_path,
            sep="\t",
            names=["user_id", "checkin_time", "lat", "lon", "location_id"],
            compression="gzip",
        )

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon", "checkin_time"])

        df["dt"] = pd.to_datetime(df["checkin_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["dt"])
        df["timestamp"] = df["dt"].astype("int64") / 1e9
        df["date"] = df["dt"].dt.date

        # Bounding-box filter
        if self.bbox is not None:
            lat_min, lat_max, lon_min, lon_max = self.bbox
            df = df[
                df["lat"].between(lat_min, lat_max)
                & df["lon"].between(lon_min, lon_max)
            ]

        # Date range filter
        if self.date_start is not None:
            df = df[df["dt"] >= self.date_start]
        if self.date_end is not None:
            df = df[df["dt"] <= self.date_end]

        if df.empty:
            raise ValueError(
                "No check-ins remain after filtering. "
                "Try a wider bounding box, broader date range, or region=None."
            )

        if self.compress_to_day:
            df = self._compress_timestamps_to_day(df)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Temporal compression
    # ------------------------------------------------------------------ #

    def _compress_timestamps_to_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip the calendar date from every check-in, keeping only HH:MM:SS,
        and map all events onto a single reference day (midnight of the
        earliest check-in date).

        Effect: stacks a multi-day dataset into one simulated 24-hour window,
        dramatically increasing temporal density without altering spatial
        distribution or intra-day rhythm.

        The 'date' column is rebuilt from the compressed timestamps so that
        worker (user, day) deduplication still produces one worker per user
        (all days collapse to the same reference date).
        """
        ref_midnight_unix = float(
            df["dt"].min().normalize().timestamp()
        )
        # seconds since midnight of each check-in's own day
        day_start_unix = (
            df["dt"].dt.normalize().astype("int64") / 1e9
        ).values
        time_of_day_sec = df["timestamp"].values - day_start_unix

        df = df.copy()
        df["timestamp"] = ref_midnight_unix + time_of_day_sec
        # Rebuild dt for any downstream time arithmetic, but intentionally
        # keep the original calendar df["date"] so that (user, day) worker
        # deduplication still produces one worker per user-per-original-day.
        # Without this, all days collapse to one reference date and the worker
        # pool shrinks to just unique users (~2.6k) instead of unique user-days
        # (~8.7k), breaking the intended workers_per_task_ratio.
        df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        # df["date"] is intentionally NOT updated — original dates are preserved.
        return df

    # ------------------------------------------------------------------ #
    # Task builders
    # ------------------------------------------------------------------ #

    def _build_tasks_checkin(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Every check-in becomes one task with a fixed expiry window."""
        tasks = df[["lat", "lon", "timestamp"]].copy()
        tasks = tasks.rename(columns={"lat": "pickup_lat", "lon": "pickup_lon",
                                      "timestamp": "release_time"})
        tasks["expire_time"] = tasks["release_time"] + self.task_window_hours * 3600.0
        tasks = tasks.reset_index(drop=True)
        tasks["task_id"] = tasks.index
        return tasks

    def _build_tasks_location_pair(self, df: pd.DataFrame) -> pd.DataFrame:
        """(location_id, day) pairs with >= 2 check-ins define one task."""
        agg = (
            df.groupby(["location_id", "date"])
              .agg(
                  release_time=("timestamp", "min"),
                  expire_time=("timestamp",  "max"),
                  pickup_lat=("lat",         "first"),
                  pickup_lon=("lon",         "first"),
                  n=("timestamp",            "count"),
              )
              .reset_index()
        )
        tasks = agg[agg["n"] >= 2].copy()
        tasks = tasks[tasks["expire_time"] > tasks["release_time"]]
        tasks = tasks.reset_index(drop=True)
        tasks["task_id"] = tasks.index
        return tasks[["task_id", "pickup_lat", "pickup_lon", "release_time", "expire_time"]]

    # ------------------------------------------------------------------ #
    # Canonical DataFrame export
    # ------------------------------------------------------------------ #

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (workers_df, tasks_df) in canonical simulation format.

        tasks_df columns:   task_id, pickup_lat, pickup_lon,
                            dropoff_lat, dropoff_lon, release_time, expire_time
        workers_df columns: worker_id, start_lat, start_lon,
                            release_time, deadline
        """
        df = self._checkins
        rng = np.random.default_rng(self.random_state)

        # ---- TASKS ---------------------------------------------------------
        if self.task_mode == "checkin":
            tasks_df = self._build_tasks_checkin(df, rng)
        else:
            tasks_df = self._build_tasks_location_pair(df)

        # Synthetic dropoff: pickup + Gaussian noise converted from km to degrees
        mean_lat   = tasks_df["pickup_lat"].mean()
        lat_std    = self.dropoff_noise_km / 111.0
        lon_std    = self.dropoff_noise_km / (111.0 * math.cos(math.radians(mean_lat)))

        tasks_df["dropoff_lat"] = tasks_df["pickup_lat"] + rng.normal(0, lat_std, len(tasks_df))
        tasks_df["dropoff_lon"] = tasks_df["pickup_lon"] + rng.normal(0, lon_std, len(tasks_df))

        tasks_df = tasks_df[[
            "task_id", "pickup_lat", "pickup_lon",
            "dropoff_lat", "dropoff_lon",
            "release_time", "expire_time",
        ]].copy()

        # ---- WORKERS: first check-in per (user, day) -----------------------
        workers_raw = (
            df.sort_values("timestamp")
              .drop_duplicates(subset=["user_id", "date"], keep="first")
              [["user_id", "lat", "lon", "timestamp"]]
              .copy()
        )
        workers_raw = workers_raw.rename(columns={
            "user_id":   "worker_id",
            "lat":       "start_lat",
            "lon":       "start_lon",
            "timestamp": "release_time",
        })
        workers_raw["deadline"] = workers_raw["release_time"] + self.shift_hours * 3600.0
        workers_raw = workers_raw.reset_index(drop=True)

        # Downsample workers to a sensible supply:demand ratio
        if self.workers_per_task_ratio is not None:
            n_workers = max(1, round(len(tasks_df) * self.workers_per_task_ratio))
            n_workers = min(n_workers, len(workers_raw))
            workers_raw = workers_raw.sample(n=n_workers, random_state=self.random_state)
            workers_raw = workers_raw.reset_index(drop=True)

        workers_df = workers_raw.copy()
        workers_df["worker_id"] = workers_df.index

        # ---- Strict float boundaries ---------------------------------------
        for col in ("release_time", "deadline"):
            workers_df[col] = workers_df[col].astype(float)
        for col in ("release_time", "expire_time"):
            tasks_df[col] = tasks_df[col].astype(float)

        return workers_df, tasks_df
