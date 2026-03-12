from __future__ import annotations

import os
import pandas as pd

# Domain models
from models.worker import Worker
from models.task import Task
from simulator.spatial_index import set_city_constants

# Dataset-specific adapters
from data.didi import didi  

def load_workers(file_path):
    """Loads workers from a CSV/txt file and returns a list of Worker objects."""
    df = pd.read_csv(file_path)
    # FAST VECTORIZED INSTANTIATION
    return [Worker(row) for row in df.to_dict('records')]

def load_tasks(file_path):
    """Loads tasks from a CSV/txt file and returns a list of Task objects."""
    df = pd.read_csv(file_path)
    
    # Configure Flat Earth constants before Tasks are created so base_utility calculates correctly
    if not df.empty and 'pickup_lat' in df.columns:
        set_city_constants(float(df['pickup_lat'].mean()))
    
    # FAST VECTORIZED INSTANTIATION
    return [Task(row) for row in df.to_dict('records')]

# --------------------------------------------------------------------------- #
# Unified Loader - The Bridge between Pandas DataFrames and the Simulation Engine
# --------------------------------------------------------------------------- #

def load_workers_tasks(dataset: str, root_path: str | None = None, **adapter_kwargs):
    """
    Return (workers, tasks) lists prepared for the simulator.
    Acts as the boundary layer: ingests raw files via Pandas, outputs pure Python objects.

    Parameters
    ----------
    dataset : str
        Identifier – e.g. "didi", "synthetic".
    root_path : str | os.PathLike
        Directory containing the raw files for that dataset.
    adapter_kwargs : Any
        Extra parameters forwarded to the adapter constructor (if needed).
    """
    if root_path is None:
        root_path = f"./data/{dataset}"

    adapter = get_adapter(dataset, root_path, **adapter_kwargs)

    if hasattr(adapter, "to_dataframes"):
        # Preferred: adapter provides tidy DataFrames
        workers_df, tasks_df = adapter.to_dataframes()
    else:
        # Fallback: assume <root>/workers.txt & <root>/tasks.txt exist
        workers_path = os.path.join(root_path, "workers.txt")
        tasks_path = os.path.join(root_path, "tasks.txt")

        if not (os.path.isfile(workers_path) and os.path.isfile(tasks_path)):
            raise FileNotFoundError(
                f"Adapter does not implement to_dataframes() and default "
                f"workers.txt / tasks.txt not found in {root_path}."
            )

        workers_df = pd.read_csv(workers_path)
        tasks_df = pd.read_csv(tasks_path)

    # --- FLAT EARTH SETUP ---
    # Calculate global constants BEFORE object instantiation
    mean_lats = []
    if not workers_df.empty and 'start_lat' in workers_df.columns:
        mean_lats.append(float(workers_df['start_lat'].mean()))
    if not tasks_df.empty and 'pickup_lat' in tasks_df.columns:
        mean_lats.append(float(tasks_df['pickup_lat'].mean()))
    
    if mean_lats:
        set_city_constants(sum(mean_lats) / len(mean_lats))

    # --- FAST VECTORIZED INSTANTIATION ---
    print(f"   📊 Instantiating {len(workers_df):,} Workers...")
    workers = [Worker(row) for row in workers_df.to_dict('records')]
    
    print(f"   📊 Instantiating {len(tasks_df):,} Tasks...")
    tasks = [Task(row) for row in tasks_df.to_dict('records')]

    return workers, tasks

def get_adapter(dataset: str, root_path: str, **kwargs):
    """Returns the appropriate adapter instance for the given dataset name."""
    if dataset == "didi":
        return didi.Adapter(root_path)
    elif dataset == "synthetic":
        raise NotImplementedError("Synthetic adapter not yet implemented.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")