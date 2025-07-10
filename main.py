"""Convenience entry-point.

Run the simulator using the settings defined in ``config.py``.

Usage (from project root):
    $ python main.py

Optional CLI flags:
    --dataset didi|checkin|synthetic  (overrides SIM_CONFIG["dataset"])
    --root    PATH                    (overrides SIM_CONFIG["root_path"])
"""

import argparse
from pathlib import Path

from config import SIM_CONFIG
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial-crowdsourcing simulator runner")
    p.add_argument("--dataset", help="Override dataset key in config.py")
    p.add_argument("--root", help="Override root_path in config.py")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = dict(SIM_CONFIG)  # shallow copy

    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.root:
        cfg["root_path"] = args.root

    workers, tasks = load_workers_tasks(cfg["dataset"], cfg.get("root_path"))

    run_simulation(
        workers,
        tasks,
        time_step=cfg.get("time_step", "5min"),
        sim_config=cfg,
    )


if __name__ == "__main__":
    main()