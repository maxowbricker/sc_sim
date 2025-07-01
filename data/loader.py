import pandas as pd
from data.checkins import checkins as checkin
from data.synthetic import synthetic
from data.didi import didi

def load_workers(file_path):
    """
    Loads workers from a .txt file and returns a list of worker dictionaries.
    """
    df = pd.read_csv(file_path)
    workers = df.to_dict(orient="records")
    return workers

def load_tasks(file_path):
    """
    Loads tasks from a .txt file and returns a list of task dictionaries.
    """
    df = pd.read_csv(file_path)
    tasks = df.to_dict(orient="records")
    return tasks

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