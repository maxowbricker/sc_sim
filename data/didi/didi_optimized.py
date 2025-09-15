"""
Memory-optimized Didi data loader for large datasets in Jupyter.
Reduces memory usage by ~70% through efficient dtypes and chunked processing.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np

class OptimizedDidiAdapter:
    """Memory-efficient Didi data loader."""
    
    def __init__(self, root_path: str):
        self.root = Path(root_path).expanduser()
    
    def load_for_simulation(self, max_workers: int = None, max_tasks: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data optimized for simulation with memory constraints.
        
        Returns:
            workers_df: Optimized worker DataFrame
            tasks_df: Optimized task DataFrame
        """
        print("🚀 Loading data with memory optimization...")
        
        # Load orders first (smaller file)
        tasks_df = self._load_orders_optimized(max_tasks)
        
        # Get time range from tasks to limit GPS data
        if not tasks_df.empty:
            task_start = tasks_df['release_time'].min()
            task_end = tasks_df['release_time'].max() + pd.Timedelta('4h')
            print(f"📅 Task time range: {task_start} to {task_end}")
        else:
            task_start = None
            task_end = None
        
        # Load GPS data with time filtering
        workers_df = self._load_gps_optimized(max_workers, task_start, task_end)
        
        return workers_df, tasks_df
    
    def _load_orders_optimized(self, max_tasks: int = None) -> pd.DataFrame:
        """Load orders with memory optimization."""
        
        # Choose file path (prefer FULL dataset for complete experiments)
        full_path = self.root / "order.txt"
        fixed_path = self.root / "order_quarter_fixed.txt" 
        quarter_path = self.root / "order_quarter.txt"
        
        # PRIORITY: Always use full dataset if available
        if full_path.exists():
            path = full_path
            print("📊 Using FULL dataset: order.txt (220,139 tasks)")
        elif fixed_path.exists():
            path = fixed_path
            print("📊 Using quarter dataset: order_quarter_fixed.txt (~140K tasks)")
        elif quarter_path.exists():
            path = quarter_path
            print("📊 Using quarter dataset: order_quarter.txt (~140K tasks)")
        else:
            available_files = list(self.root.glob("order*.txt"))
            raise FileNotFoundError(f"No order files found in {self.root}. Available files: {available_files}")
            
        print(f"📊 Loading Orders from: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Optimized dtypes to reduce memory
        dtype_map = {
            0: 'category',  # order_id (string -> category saves memory)
            1: 'int32',     # start_billing (timestamp as int32)
            2: 'int32',     # end_billing  
            3: 'float32',   # pickup_lon (float64 -> float32 saves 50% memory)
            4: 'float32',   # pickup_lat
            5: 'float32',   # dropoff_lon  
            6: 'float32',   # dropoff_lat
        }
        
        # Load with optimized types
        if max_tasks:
            df = pd.read_csv(path, header=None, dtype=dtype_map, nrows=max_tasks)
        else:
            df = pd.read_csv(path, header=None, dtype=dtype_map)
        
        # Handle 8-column format
        if df.shape[1] == 8:
            df = df.iloc[:, :7]
            
        df.columns = [
            "order_id", "start_billing", "end_billing", 
            "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"
        ]
        
        # Convert timestamps efficiently
        df["start_billing"] = pd.to_datetime(df["start_billing"], unit="s", utc=True)
        df["end_billing"] = pd.to_datetime(df["end_billing"], unit="s", utc=True)
        
        # Create simulation format
        df["expire_time"] = df["start_billing"] + pd.Timedelta("2h")
        df = df.rename(columns={
            "order_id": "task_id",
            "start_billing": "release_time"
        })[[
            "task_id", "pickup_lat", "pickup_lon", 
            "dropoff_lat", "dropoff_lon", "release_time", "expire_time"
        ]]
        
        print(f"✅ Loaded {len(df):,} tasks")
        return df.sort_values("release_time")
    
    def _load_gps_optimized(self, max_workers: int = None, 
                           time_start: pd.Timestamp = None, 
                           time_end: pd.Timestamp = None) -> pd.DataFrame:
        """Load GPS data with memory optimization and time filtering."""
        
        # Choose file path (prefer FULL dataset for complete experiments)
        full_path = self.root / "gps.txt"
        fixed_path = self.root / "gps_quarter_fixed.txt"
        quarter_path = self.root / "gps_quarter.txt" 
        
        if full_path.exists():
            path = full_path
        elif fixed_path.exists():
            path = fixed_path
        else:
            path = quarter_path
            
        print(f"📊 Loading GPS from: {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # For very large files, use chunked processing
        if path.stat().st_size > 500 * 1024 * 1024:  # > 500MB
            return self._load_gps_chunked(path, max_workers, time_start, time_end)
        else:
            return self._load_gps_direct(path, max_workers, time_start, time_end)
    
    def _load_gps_direct(self, path: Path, max_workers: int, 
                        time_start: pd.Timestamp, time_end: pd.Timestamp) -> pd.DataFrame:
        """Direct GPS loading for smaller files."""
        
        dtype_map = {
            0: 'category',  # driver_id  
            1: 'category',  # order_id
            2: 'int32',     # timestamp
            3: 'float32',   # lon
            4: 'float32',   # lat
        }
        
        df = pd.read_csv(path, header=None, dtype=dtype_map)
        df.columns = ["driver_id", "order_id", "timestamp", "lon", "lat"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        
        # Time filtering
        if time_start and time_end:
            mask = (df["timestamp"] >= time_start) & (df["timestamp"] <= time_end)
            df = df[mask]
            print(f"🕒 Filtered to time range: {len(df):,} GPS points")
        
        # Convert to workers format
        return self._gps_to_workers(df, max_workers)
    
    def _load_gps_chunked(self, path: Path, max_workers: int,
                         time_start: pd.Timestamp, time_end: pd.Timestamp) -> pd.DataFrame:
        """Chunked GPS loading for very large files."""
        
        print(f"🔄 Using chunked loading for large file...")
        
        dtype_map = {
            0: 'category',
            1: 'category', 
            2: 'int32',
            3: 'float32',
            4: 'float32',
        }
        
        chunk_size = 100_000  # Process 100K rows at a time
        first_gps_chunks = []
        last_gps_chunks = []
        
        # Convert time filter to unix timestamps for efficiency
        time_start_unix = int(time_start.timestamp()) if time_start else None
        time_end_unix = int(time_end.timestamp()) if time_end else None
        
        for chunk in pd.read_csv(path, header=None, dtype=dtype_map, chunksize=chunk_size):
            chunk.columns = ["driver_id", "order_id", "timestamp", "lon", "lat"]
            
            # Time filtering before datetime conversion (more efficient)
            if time_start_unix and time_end_unix:
                mask = (chunk["timestamp"] >= time_start_unix) & (chunk["timestamp"] <= time_end_unix)
                chunk = chunk[mask]
                
            if chunk.empty:
                continue
                
            # Convert timestamp only for filtered data
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True)
            
            # Extract first/last GPS per driver for this chunk
            chunk_first = chunk.groupby("driver_id", observed=True).first().reset_index()
            chunk_last = chunk.groupby("driver_id", observed=True).last().reset_index()
            
            first_gps_chunks.append(chunk_first[["driver_id", "timestamp", "lon", "lat"]])
            last_gps_chunks.append(chunk_last[["driver_id", "timestamp"]])
        
        if not first_gps_chunks:
            print("❌ No GPS data found in time range!")
            return pd.DataFrame()
        
        # Combine chunks and get final first/last per driver
        print(f"🔗 Combining {len(first_gps_chunks)} chunks...")
        
        all_first = pd.concat(first_gps_chunks, ignore_index=True)
        all_last = pd.concat(last_gps_chunks, ignore_index=True)
        
        first_gps = all_first.groupby("driver_id", observed=True).first().reset_index()
        last_gps = all_last.groupby("driver_id", observed=True).last().reset_index()
        
        # Create workers DataFrame  
        first_gps = first_gps.rename(columns={
            "driver_id": "worker_id",
            "lon": "start_lon", 
            "lat": "start_lat",
            "timestamp": "release_time"
        })
        
        last_gps = last_gps.rename(columns={
            "driver_id": "worker_id",
            "timestamp": "last_seen"
        })
        
        workers_df = first_gps.merge(last_gps, on="worker_id")
        workers_df["deadline"] = workers_df["last_seen"] + pd.Timedelta("4h")
        
        workers_df = workers_df[[
            "worker_id", "start_lat", "start_lon", "release_time", "deadline"
        ]]
        
        # Limit workers if requested
        if max_workers and len(workers_df) > max_workers:
            workers_df = workers_df.head(max_workers)
        
        print(f"✅ Created {len(workers_df):,} workers")
        return workers_df.sort_values("release_time")
    
    def _gps_to_workers(self, gps_df: pd.DataFrame, max_workers: int) -> pd.DataFrame:
        """Convert GPS DataFrame to workers format."""
        
        # First GPS per driver
        first_gps = (
            gps_df.groupby("driver_id", observed=True)
            .first()
            .reset_index()[["driver_id", "timestamp", "lon", "lat"]]
            .rename(columns={
                "driver_id": "worker_id",
                "lon": "start_lon",
                "lat": "start_lat", 
                "timestamp": "release_time",
            })
        )
        
        # Last GPS per driver
        last_gps = (
            gps_df.groupby("driver_id", observed=True)["timestamp"].last().reset_index()
            .rename(columns={
                "driver_id": "worker_id",
                "timestamp": "last_seen",
            })
        )
        
        workers_df = first_gps.merge(last_gps, on="worker_id")
        workers_df["deadline"] = workers_df["last_seen"] + pd.Timedelta("4h")
        
        workers_df = workers_df[[
            "worker_id", "start_lat", "start_lon", "release_time", "deadline"
        ]]
        
        # Limit workers
        if max_workers and len(workers_df) > max_workers:
            workers_df = workers_df.head(max_workers)
            
        print(f"✅ Created {len(workers_df):,} workers")
        return workers_df.sort_values("release_time")


# Convenience function for notebooks
def load_optimized_didi_data(max_workers: int = None, max_tasks: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Didi data with memory optimization for Jupyter notebooks.
    
    Args:
        max_workers: Limit number of workers (None for all)
        max_tasks: Limit number of tasks (None for all) 
        
    Returns:
        workers_df, tasks_df: Optimized DataFrames
    """
    adapter = OptimizedDidiAdapter("/Users/maxapple/Documents/GitHub/sc_sim/data/didi")
    return adapter.load_for_simulation(max_workers, max_tasks)
