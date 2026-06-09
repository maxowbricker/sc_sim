#!/usr/bin/env python3
"""
Isolated test for platform revenue (task utility) from trip distance.

Methodology (Basık et al., "Fair Task Allocation in Crowdsourced Delivery"):
  α = d(pickup, dropoff)   — core movement cost (source → destination)
  t_j.m = β_base + β_km × α — monetary reward positively correlated with α

This is NOT the composite-strategy U term (1/(1+d_pick)), which is worker-side
proximity efficiency. Platform revenue here is task-intrinsic value.

Usage:
  python scripts/test_dynamic_utility.py
  python scripts/test_dynamic_utility.py --real --day 496528674@qq.com_20161128 --sample 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.task import core_movement_cost_km, platform_revenue_from_alpha
from config import get_platform_revenue_config
from simulator.spatial_index import set_city_constants

_cfg = get_platform_revenue_config()
BASE_FARE = _cfg["base_fare"]
PER_KM_RATE = _cfg["per_km_rate"]


def task_revenue(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon) -> tuple[float, float]:
    """Returns (α_km, revenue) for one task."""
    alpha = core_movement_cost_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    return alpha, platform_revenue_from_alpha(alpha)


def _print_table(rows: list[dict]) -> None:
    cols = ["task_id", "desc", "alpha_km", "revenue"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def run_mock_demo() -> None:
    print("=== Mock Chengdu tasks (sanity check) ===\n")
    set_city_constants(30.65)

    mock_tasks = [
        {"task_id": "T-001", "desc": "Short local trip", "p_lat": 30.650, "p_lon": 104.060, "d_lat": 30.660, "d_lon": 104.065},
        {"task_id": "T-002", "desc": "Medium city trip", "p_lat": 30.655, "p_lon": 104.050, "d_lat": 30.680, "d_lon": 104.090},
        {"task_id": "T-003", "desc": "Long highway trip", "p_lat": 30.620, "p_lon": 104.010, "d_lat": 30.720, "d_lon": 104.150},
        {"task_id": "T-004", "desc": "Tiny hop (<1 km)", "p_lat": 30.650, "p_lon": 104.060, "d_lat": 30.652, "d_lon": 104.062},
    ]

    rows = []
    for t in mock_tasks:
        alpha, rev = task_revenue(t["p_lat"], t["p_lon"], t["d_lat"], t["d_lon"])
        rows.append({
            "task_id": t["task_id"],
            "desc": t["desc"],
            "alpha_km": f"{alpha:.2f}",
            "revenue": f"${rev:.2f}",
        })
    _print_table(rows)
    print("\nLong trips should have materially higher revenue than tiny hops.\n")


def run_real_sample(day: str, sample: int, data_root: str) -> None:
    from data.loader import load_workers_tasks

    root = data_root if os.path.isabs(data_root) else os.path.join(PROJECT_ROOT, data_root)
    day_path = os.path.join(root, day)
    if not os.path.isdir(day_path):
        print(f"Day folder not found: {day_path}")
        return

    print(f"=== Real DiDi sample: {day} (n={sample}) ===\n")
    workers, tasks = load_workers_tasks("didi", root_path=day_path)
    if not tasks:
        print("No tasks loaded.")
        return

    mean_lat = float(np.mean([t.pickup_lat for t in tasks]))
    set_city_constants(mean_lat)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(tasks), size=min(sample, len(tasks)), replace=False)

    rows = []
    alphas = []
    revenues = []
    for i in idx:
        t = tasks[i]
        rows.append({
            "task_id": str(t.id)[:12],
            "desc": f"release={t.release_time:.0f}",
            "alpha_km": f"{t.core_movement_cost_km:.2f}",
            "revenue": f"${t.revenue:.2f}",
        })
        alphas.append(t.core_movement_cost_km)
        revenues.append(t.revenue)

    _print_table(rows)
    print()
    print(f"α (pickup→dropoff km): mean={np.mean(alphas):.2f}, p50={np.percentile(alphas, 50):.2f}, "
          f"p95={np.percentile(alphas, 95):.2f}, max={np.max(alphas):.2f}")
    print(f"Revenue (${BASE_FARE:.2f} + ${PER_KM_RATE:.2f}/km): "
          f"mean=${np.mean(revenues):.2f}, p50=${np.percentile(revenues, 50):.2f}, "
          f"p95=${np.percentile(revenues, 95):.2f}, max=${np.max(revenues):.2f}")
    print(f"\nLoaded {len(tasks):,} tasks total; showing {len(rows)} random samples.")
    print("Compare spread to flat utility=1 — revenue now scales with trip length.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test platform revenue from trip distance (Basık α model)")
    parser.add_argument("--real", action="store_true", help="Sample tasks from a real DiDi day folder")
    parser.add_argument("--day", default="496528674@qq.com_20161128", help="Day folder under data root")
    parser.add_argument("--sample", type=int, default=15, help="Number of random tasks to display")
    parser.add_argument("--data-root", default="data/didi/full_didi_gaia", help="DiDi day folders root")
    args = parser.parse_args()

    print("Dynamic platform utility test")
    print(f"Model: revenue = {BASE_FARE:.2f} + {PER_KM_RATE:.2f} × α  (α = Manhattan pickup→dropoff km)")
    print(f"Distance fn: simulator.spatial_index.fast_manhattan_km (same as simulation)\n")

    run_mock_demo()
    if args.real:
        run_real_sample(args.day, args.sample, args.data_root)


if __name__ == "__main__":
    main()
