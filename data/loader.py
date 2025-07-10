# Standard libs
from __future__ import annotations

import os
import pandas as pd

# Domain models
from models.worker import Worker
from models.task import Task

# Dataset-specific adapters
from data.checkins import checkin  # Gowalla / Weeplaces
from data.synthetic import synthetic  # Synthetic generator (placeholder)
from data.didi import didi  # Didi Gaia GPS + orders

def load_workers(file_path):
    """
    Loads workers from a .txt (CSV‑formatted) file and returns a list of Worker objects.
    """
    df = pd.read_csv(file_path)
    workers = [Worker(row.to_dict()) for _, row in df.iterrows()]
    return workers

def load_tasks(file_path):
    """
    Loads tasks from a .txt (CSV‑formatted) file and returns a list of Task objects.
    """
    df = pd.read_csv(file_path)
    tasks = [Task(row.to_dict()) for _, row in df.iterrows()]
    return tasks

# --------------------------------------------------------------------------- #
# New unified loader – returns canonical Worker / Task lists independent of
# dataset-specific quirks
# --------------------------------------------------------------------------- #


# root_path becomes optional; if None, default to ./data/<dataset>

def load_workers_tasks(dataset: str, root_path: str | None = None, **adapter_kwargs):
    """Return ``(workers, tasks)`` lists prepared for the simulator.

    Parameters
    ----------
    dataset : str
        Identifier – e.g. "didi", "checkin", "synthetic".
    root_path : str | os.PathLike
        Directory containing the raw files for that dataset.
    adapter_kwargs : Any
        Extra parameters forwarded to the adapter constructor (if needed).
    """

    # Derive default path if none supplied
    if root_path is None:
        root_path = f"./data/{dataset}"

    adapter = get_adapter(dataset, root_path, **adapter_kwargs)

    if hasattr(adapter, "to_dataframes"):
        # Preferred: adapter provides tidy DataFrames
        workers_df, tasks_df = adapter.to_dataframes()
    else:
        # Fallback: assume <root>/workers.txt & <root>/tasks.txt exist in CSV
        workers_path = os.path.join(root_path, "workers.txt")
        tasks_path = os.path.join(root_path, "tasks.txt")

        if not (os.path.isfile(workers_path) and os.path.isfile(tasks_path)):
            raise FileNotFoundError(
                "Adapter does not implement to_dataframes() and default "
                "workers.txt / tasks.txt files not found in provided root_path."
            )

        workers_df = pd.read_csv(workers_path)
        tasks_df = pd.read_csv(tasks_path)

    # Convert rows → domain objects
    workers = [Worker(row._asdict()) for row in workers_df.itertuples(index=False)]
    tasks = [Task(row._asdict()) for row in tasks_df.itertuples(index=False)]

    return workers, tasks

def get_adapter(dataset: str, root_path: str, **kwargs):
    """
    Returns the appropriate adapter instance for the given dataset name.
    """
    if dataset == "checkin":
        return checkin.Adapter(root_path)
    elif dataset == "synthetic":
        return synthetic.Adapter(root_path)
    elif dataset == "didi":
        return didi.Adapter(root_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")