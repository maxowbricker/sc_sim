from __future__ import annotations

import os
import pandas as pd

# Domain models
from models.worker import Worker
from models.task import Task
from simulator.spatial_index import set_city_constants

# Dataset-specific adapters
from data.didi import didi
from data.nyc_taxi import nyc_taxi
from data.gowalla import gowalla

# Config and Sampler
from config import get_data_sampling_config
from data.stratified_sampler import stratified_temporal_sample

def load_workers(file_path):
    """Loads workers from a CSV/txt file and returns a list of Worker objects."""
    df = pd.read_csv(file_path)
    return [Worker(row) for row in df.to_dict('records')]

def load_tasks(file_path):
    """Loads tasks from a CSV/txt file and returns a list of Task objects."""
    df = pd.read_csv(file_path)
    if not df.empty and 'pickup_lat' in df.columns:
        set_city_constants(float(df['pickup_lat'].mean()))
    return [Task(row) for row in df.to_dict('records')]

# --------------------------------------------------------------------------- #
# Unified Loader - The Bridge between Pandas DataFrames and the Simulation Engine
# --------------------------------------------------------------------------- #

def load_workers_tasks(dataset: str, root_path: str | None = None, **adapter_kwargs):
    """
    Return (workers, tasks) lists prepared for the simulator.
    Acts as the boundary layer: ingests raw files via Pandas, outputs pure Python objects.
    """
    if root_path is None:
        root_path = f"./data/{dataset}"

    adapter = get_adapter(dataset, root_path, **adapter_kwargs)

    if hasattr(adapter, "to_dataframes"):
        workers_df, tasks_df = adapter.to_dataframes()
    else:
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

    # --- STRATIFIED SAMPLING INTERCEPT ---
    sampling_cfg = get_data_sampling_config()
    if sampling_cfg.get("use_stratified_sampling", False):
        target_t = sampling_cfg.get("target_tasks", 20000)
        target_w = sampling_cfg.get("target_workers", 5000)
        print(f"   ✂️ Applying Stratified Sampling: Shrinking to {target_t} tasks and {target_w} workers...")
        
        tasks, w_dict = stratified_temporal_sample(
            all_workers=workers,
            all_tasks=tasks,
            target_tasks=target_t,
            worker_counts=[target_w],
            num_bins=sampling_cfg.get("stratified_sampling_bins", 12),
            seed=sampling_cfg.get("random_state", 42)
        )
        workers = w_dict[target_w]

    return workers, tasks

def get_adapter(dataset: str, root_path: str, **kwargs):
    if dataset == "didi":
        return didi.Adapter(root_path)
    elif dataset == "nyc_taxi":
        return nyc_taxi.Adapter(root_path, **kwargs)
    elif dataset == "gowalla":
        return gowalla.Adapter(root_path, **kwargs)
    elif dataset == "synthetic":
        raise NotImplementedError("Synthetic adapter not yet implemented.")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
