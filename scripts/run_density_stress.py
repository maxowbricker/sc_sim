#!/usr/bin/env python3
"""
Sec 5.5 — Density Stress Test: Governor Robustness across City Scales

Evaluates the champion RL model and key baselines across four worker-density
settings on Nov 28 (standard eval day).
Tasks and workers scale together at a FIXED 4:1 ratio (matching training),
so the task-to-worker ratio stays constant across all settings.
This tests city-scale generalization, not out-of-distribution scarcity.

    Workers | Tasks  | Ratio
    --------|--------|------
      2,000 |  8,000 |  4:1
      5,000 | 20,000 |  4:1
     10,000 | 40,000 |  4:1   ← standard training setting
     15,000 | 60,000 |  4:1

Strategies compared:
    Greedy           — pure efficiency lower bound
    FATP-ANN         — state-of-the-art fairness-aware heuristic (CR-08)
    Static-Composite — tuned static weights, main comparison
    RL Governor      — proposed adaptive method

Usage (from project root):
    python scripts/run_density_stress.py

Outputs:
    figures/density_stress.pdf
    figures/density_stress.png
    density_stress_results.csv
"""

import csv
import os
import sys
import time
import numpy as np

matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_ok = False
    print("⚠️  matplotlib not found — will save CSV but skip plots.")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as config_module
from stable_baselines3 import PPO
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv
from simulator.strategies.fatp_ann import FairnessCapTracker

# Import all strategies to ensure they're registered
import simulator.strategies.greedy
import simulator.strategies.composite
import simulator.strategies.fatp_ann
import simulator.strategies.ewma_only
import simulator.strategies.random_assign

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_ROOT  = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
EVAL_DAY   = "496528674@qq.com_20161128"
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "rl_logs_sb3", "run_20260513_071355",
    "best_model", "best_model",   # no .zip suffix
)

WORKER_COUNTS = [2_000, 5_000, 10_000, 20_000]
TASK_WORKER_RATIO = 4   # Must match training: 40k tasks / 10k workers = 4:1

STATIC_COMPOSITE_PARAMS = {
    "fairness_weight":        1.0,
    "starvation_weight":      0.2,
    "utility_weight":         1.0,
    "gamma":                  0.1,
    "k":                      15,
    "soft_threshold":         0.05,
    "enable_diagnostics":     False,
    "enable_deferral_tracking": False,
}

FATP_ANN_PARAMS = {
    "mu":           0.5,
    "alpha_scale":  0.5,
    "use_k_nearest": False,
    "k":            15,
}

# ── strategy registry ────────────────────────────────────────────────────────
# Each entry: (label, strategy_name, params_dict | None for RL)
STRATEGIES = [
    ("Greedy",           "greedy",    {}),
    ("FATP-ANN",         "fatp_ann",  FATP_ANN_PARAMS),
    ("Static-Composite", "composite", STATIC_COMPOSITE_PARAMS),
    ("RL Governor",      "rl",        None),   # RL handled separately
]

METRICS = ["JFI", "Wait (m)", "TAR", "Peak Backlog", "Avg Pickup (km)"]


def _extract(stats: dict) -> dict:
    return {
        "JFI":             stats.get("final_jains_fairness_index", 0.0),
        "Wait (m)":        stats.get("avg_wait_time_minutes", 0.0),
        "TAR":             stats.get("task_assignment_ratio", 0.0),
        "Peak Backlog":    stats.get("backlog_peak", 0),
        "Avg Pickup (km)": stats.get("avg_pickup_distance_km", 0.0),
    }


def _set_density(worker_count: int):
    """Apply the worker count and scale tasks proportionally at 4:1."""
    config_module.DATA_SAMPLING["target_workers"] = worker_count
    config_module.DATA_SAMPLING["target_tasks"]   = worker_count * TASK_WORKER_RATIO


def _run_heuristic(worker_count: int, strategy_name: str, params: dict) -> dict:
    _set_density(worker_count)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[EVAL_DAY])
    env.reset(seed=42)

    # FATP-ANN needs its fairness-cap tracker injected
    if strategy_name == "fatp_ann":
        tracker = FairnessCapTracker()
        tracker.initialize(env.simulator.state.all_workers_map.values())
        params = {**params, "fairness_cap_tracker": tracker}

    env.simulator.switch_strategy(strategy_name, params)

    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        done = terminated or truncated

    return _extract(env.simulator.get_final_results())


def _run_rl(worker_count: int, model: PPO) -> dict:
    _set_density(worker_count)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[EVAL_DAY])
    obs, _ = env.reset(seed=42)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(np.ravel(action))
        done = terminated or truncated

    return _extract(env.simulator.get_final_results())


def main():
    print("🔄 Loading champion RL model …")
    model = PPO.load(MODEL_PATH)

    # results[strategy_label][metric] = list of values (one per worker count)
    results = {label: {m: [] for m in METRICS} for label, _, _ in STRATEGIES}

    for wc in WORKER_COUNTS:
        tasks = wc * TASK_WORKER_RATIO
        print(f"\n👥  Workers = {wc:,}  |  Tasks = {tasks:,}  (ratio {TASK_WORKER_RATIO}:1)")

        for label, strat, params in STRATEGIES:
            t0 = time.time()
            try:
                if strat == "rl":
                    m = _run_rl(wc, model)
                else:
                    m = _run_heuristic(wc, strat, params)

                for metric in METRICS:
                    results[label][metric].append(m[metric])

                print(f"   {label:<20} | JFI={m['JFI']:.4f}  "
                      f"Wait={m['Wait (m)']:.2f}m  TAR={m['TAR']:.3f}  "
                      f"[{time.time()-t0:.1f}s]")

            except Exception as exc:
                print(f"   {label:<20} ❌ FAILED: {exc}")
                for metric in METRICS:
                    results[label][metric].append(float("nan"))

    # Reset to training defaults
    config_module.DATA_SAMPLING["target_workers"] = 10_000
    config_module.DATA_SAMPLING["target_tasks"]   = 40_000

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "density_stress_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Workers"] + [f"{lbl}_{m}" for lbl, _, _ in STRATEGIES for m in METRICS]
        writer.writerow(header)
        for i, wc in enumerate(WORKER_COUNTS):
            row = [wc] + [results[lbl][m][i] for lbl, _, _ in STRATEGIES for m in METRICS]
            writer.writerow(row)
    print(f"\n✅  Saved {csv_path}")

    if not matplotlib_ok:
        return

    # ── Plot: 3 panels — JFI, Wait Time, TAR ─────────────────────────────────
    STYLE = {
        "Greedy":           dict(color="#95a5a6", linestyle=":",  marker="^", lw=1.8),
        "FATP-ANN":         dict(color="#8e44ad", linestyle="-.", marker="D", lw=2.0),
        "Static-Composite": dict(color="#2980b9", linestyle="--", marker="s", lw=2.0),
        "RL Governor":      dict(color="#27ae60", linestyle="-",  marker="o", lw=2.5),
    }

    workers_k = [w / 1000 for w in WORKER_COUNTS]
    plot_specs = [
        ("JFI",      "Jain's Fairness Index (JFI)",  "lower left"),
        ("Wait (m)", "Avg Wait Time (minutes)",        "upper right"),
        ("TAR",      "Task Assignment Ratio (TAR)",   "lower right"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel, leg_loc) in zip(axes, plot_specs):
        for label, _, _ in STRATEGIES:
            vals = results[label][metric]
            ax.plot(workers_k, vals, label=label, markersize=7, **STYLE[label])

        ax.set_xlabel("Number of Workers (thousands)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.set_xticks(workers_k)
        ax.set_xticklabels([f"{int(w)}k" for w in workers_k])
        ax.legend(loc=leg_loc, fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.45)

        # Shade the worker-scarce region (below standard 10k)
        ax.axvspan(workers_k[0] - 0.3, 10.0, alpha=0.07, color="red")
        ax.axvline(10.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(10.1, ax.get_ylim()[0], "standard\n(10k)", fontsize=7.5,
                color="grey", va="bottom")

    fig.suptitle(
        "Density Stress Test: Robustness across City Scales\n"
        rf"(Nov 28 eval day, fixed {TASK_WORKER_RATIO}:1 task-to-worker ratio, seed=42)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES_DIR, f"density_stress.{ext}")
        plt.savefig(out, format=ext, dpi=300, bbox_inches="tight")
        print(f"✅  Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
