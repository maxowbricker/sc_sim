"""Convenience entry-point.

Run the simulator using the settings defined in ``config.py``.

Usage (from project root):
    $ python main.py

Optional CLI flags:
    --dataset didi|checkin|synthetic  (overrides config dataset)
    --root    PATH                    (overrides config root_path)
"""

import argparse
from pathlib import Path

from config import get_simulation_config, create_composite_config
from data.loader import load_workers_tasks
from simulator.simulation import run_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spatial-crowdsourcing simulator runner")
    p.add_argument("--dataset", help="Override dataset key in config.py")
    p.add_argument("--root", help="Override root_path in config.py")
    p.add_argument("--strategy", help="Override assignment_strategy in config.py")
    return p.parse_args()


def main():
    args = parse_args()

    # Build configuration with command line overrides
    overrides = {}
    if args.dataset:
        overrides["dataset"] = args.dataset
    if args.root:
        overrides["root_path"] = args.root
    if args.strategy:
        overrides["assignment_strategy"] = args.strategy
    
    # Create config using new system
    if args.strategy == "composite" or not args.strategy:
        cfg = create_composite_config(**overrides)
    else:
        cfg = get_simulation_config()
        cfg.update(overrides)

    workers, tasks = load_workers_tasks(cfg["dataset"], cfg.get("root_path"))

    run_simulation(
        workers,
        tasks,
        sim_config=cfg,
    )


if __name__ == "__main__":
    main()