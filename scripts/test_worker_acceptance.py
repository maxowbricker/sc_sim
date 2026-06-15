#!/usr/bin/env python3
"""
Sanity tests for Basık-style stochastic worker acceptance (cascade dispatch).

Usage:
  python scripts/test_worker_acceptance.py
  python scripts/test_worker_acceptance.py --tasks 2000 --workers 500
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import get_strategy_params, get_worker_acceptance_config
from data.loader import load_workers_tasks
from simulator.behavior import acceptance_probability, seed_acceptance_rng
from simulator.simulation import run_simulation


def test_probability_math() -> None:
    print("=== Unit: acceptance_probability ===")
    cfg = get_worker_acceptance_config()
    cfg["enabled"] = True

    p0 = acceptance_probability(0.0, cfg)
    p3 = acceptance_probability(3.0, cfg)
    p1 = acceptance_probability(1.0, cfg)
    p2 = acceptance_probability(2.0, cfg)

    assert abs(p0 - 0.6) < 1e-9, f"expected 0.6 at 0km, got {p0}"
    assert abs(p3 - 0.6 * (2.718281828 ** -3)) < 0.001, f"unexpected p at 3km: {p3}"
    assert p1 > p2 > p3, "probability should decrease with deadhead distance"
    print(f"  P(0 km)  = {p0:.3f}")
    print(f"  P(1 km)  = {p1:.3f}")
    print(f"  P(2 km)  = {p2:.3f}")
    print(f"  P(3 km)  = {p3:.3f}")
    print("  OK\n")


def _run_sim(workers, tasks, acceptance_enabled: bool) -> dict:
    params = get_strategy_params("composite")
    acceptance = dict(get_worker_acceptance_config())
    acceptance["enabled"] = acceptance_enabled
    acceptance["seed"] = 42
    params["worker_acceptance"] = acceptance

    cfg = {
        "assignment_strategy": "composite",
        "strategy_params": params,
    }
    seed_acceptance_rng(42)
    return run_simulation(workers, tasks, sim_config=cfg)


def _print_row(label: str, r: dict) -> None:
    print(
        f"  {label:12s}  TAR={r['task_assignment_ratio']:.3f}  "
        f"JFI={r['final_jains_fairness_index']:.3f}  "
        f"Wait={r['avg_wait_time_minutes']:.2f}m  "
        f"Backlog={r['backlog_peak']}  "
        f"Offers={r.get('total_offers', 0)}  "
        f"Reject%={100 * (1 - r.get('offer_acceptance_rate', 1.0)):.1f}"
    )


def test_simulation_ablation(tasks_n: int, workers_n: int) -> None:
    print(f"=== Simulation ablation ({tasks_n} tasks, {workers_n} workers) ===")
    data_path = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
    days = sorted(d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)))
    workers, tasks = load_workers_tasks("didi", root_path=os.path.join(data_path, days[0]))
    workers, tasks = workers[:workers_n], tasks[:tasks_n]

    off = _run_sim(workers, tasks, acceptance_enabled=False)
    on = _run_sim(workers, tasks, acceptance_enabled=True)

    _print_row("OFF", off)
    _print_row("ON", on)

    assert off.get("total_offers", 0) == 0, "acceptance off should produce zero offers"
    assert on.get("total_offers", 0) > 0, "acceptance on should record offers"
    assert on.get("total_rejections", 0) > 0, "acceptance on should have rejections"

    # Disabled path must match prior behavior (deterministic assignment)
    off2 = _run_sim(workers, tasks, acceptance_enabled=False)
    assert off["completed_tasks"] == off2["completed_tasks"], "off-run should be reproducible"
    print("  OK\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test worker acceptance behavior")
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=300)
    args = parser.parse_args()

    test_probability_math()
    test_simulation_ablation(args.tasks, args.workers)
    print("All worker acceptance tests passed.")


if __name__ == "__main__":
    main()
