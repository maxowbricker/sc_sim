#!/usr/bin/env python3
"""
Compare two static composite runs on the same day:

  A) "Utility-only": fairness_weight=0, starvation_weight=0, utility_weight=1,
     soft_threshold=0 (pure distance/utility term; other params match config).

  B) "Config baseline": STRATEGY_PARAMS['composite'] from config.py (your current baseline).

Both use normalize_scores=True (same as compare_model_to_baseline static run).

Usage:
    python scripts/compare_composite_variants.py
    python scripts/compare_composite_variants.py --day 496528674@qq.com_20161128 --stratified false
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulator.simulation import EventSimulator
from config import get_simulation_config, get_strategy_params
from data.loader import load_workers_tasks


def run_composite(label: str, strategy_params: dict, day: str, data_root: str) -> dict:
    config = get_simulation_config()
    config["assignment_strategy"] = "composite"
    params = dict(strategy_params)
    params["normalize_scores"] = True
    config["strategy_params"] = params

    day_path = os.path.join(data_root, day)
    workers, tasks = load_workers_tasks(dataset=config["dataset"], root_path=day_path)
    sim = EventSimulator(workers, tasks, config)
    sim.reset()
    sim.step()
    return sim.get_final_results()


def main():
    parser = argparse.ArgumentParser(description="Utility-only composite vs config.py composite baseline")
    parser.add_argument("--day", default="496528674@qq.com_20161128", help="Day folder under data root")
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
        help="DiDi GAIA root",
    )
    parser.add_argument(
        "--stratified",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Override DATA_SAMPLING.use_stratified_sampling (true/false); default = config.py",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(PROJECT_ROOT, args.data_root.lstrip("./"))

    if args.stratified is not None:
        import config as config_module

        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified
        print(f"Stratified sampling: {'ON' if args.stratified else 'OFF'} (override)\n")

    baseline_params = get_strategy_params("composite")

    utility_only_params = {
        **baseline_params,
        "fairness_weight": 0.0,
        "starvation_weight": 0.0,
        "utility_weight": 1.0,
        "soft_threshold": 0.0,
    }

    print("=" * 72)
    print("COMPOSITE VARIANT COMPARISON (same day, same data load)")
    print("=" * 72)
    print(f"Day: {args.day}")
    print()
    print("B) Config baseline (from config.py STRATEGY_PARAMS['composite']):")
    for k in sorted(baseline_params.keys()):
        print(f"    {k}: {baseline_params[k]}")
    print()
    print("A) Utility-only (λ1=0, λ2=0, λ3=1, soft_threshold=0); γ, k from config:")
    print(f"    fairness_weight: {utility_only_params['fairness_weight']}")
    print(f"    starvation_weight: {utility_only_params['starvation_weight']}")
    print(f"    utility_weight: {utility_only_params['utility_weight']}")
    print(f"    soft_threshold: {utility_only_params['soft_threshold']}")
    print(f"    gamma: {utility_only_params['gamma']}")
    print(f"    k: {utility_only_params['k']}")
    print("=" * 72)

    print("\nRunning A) utility-only ...")
    stats_a = run_composite("utility_only", utility_only_params, args.day, args.data_root)
    print("Running B) config baseline ...")
    stats_b = run_composite("config_baseline", baseline_params, args.day, args.data_root)

    def pick(d, key, default="—"):
        return d.get(key, default)

    jfi_a = pick(stats_a, "final_jains_fairness_index", 0.0)
    jfi_b = pick(stats_b, "final_jains_fairness_index", 0.0)
    bl_a = pick(stats_a, "backlog_peak", 0.0)
    bl_b = pick(stats_b, "backlog_peak", 0.0)
    wait_a = pick(stats_a, "avg_wait_time_minutes", 0.0)
    wait_b = pick(stats_b, "avg_wait_time_minutes", 0.0)

    print("\n" + "=" * 72)
    print(f"{'Metric':<28} | {'A) Utility-only':<18} | {'B) Config baseline':<18}")
    print("-" * 72)
    print(f"{'final_jains_fairness_index':<28} | {float(jfi_a):<18.4f} | {float(jfi_b):<18.4f}")
    print(f"{'backlog_peak':<28} | {float(bl_a):<18.0f} | {float(bl_b):<18.0f}")
    print(f"{'avg_wait_time_minutes':<28} | {float(wait_a):<18.2f} | {float(wait_b):<18.2f}")
    print("=" * 72)
    print("\nΔ (A − B):")
    print(f"  Δ JFI:    {float(jfi_a) - float(jfi_b):+.4f}")
    print(f"  Δ Backlog: {float(bl_a) - float(bl_b):+.0f}")
    print(f"  Δ Wait:   {float(wait_a) - float(wait_b):+.2f} min")
    print()


if __name__ == "__main__":
    main()
