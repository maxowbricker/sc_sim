import pandas as pd

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