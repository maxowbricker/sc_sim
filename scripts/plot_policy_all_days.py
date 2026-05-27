#!/usr/bin/env python3
"""
Sec 5.3 — Policy Visualization: Weight Trajectory across All 6 Eval Days

Runs the champion RL model on each of the 6 held-out test days and generates
two trajectory plots per day:

  _clean  — plain white background, no shading
  _delta  — background shading shows whether reward improved vs previous step
            (green = reward went UP, red = reward went DOWN)

Usage (from project root):
    python scripts/plot_policy_all_days.py

Outputs per day (12 files total):
    figures/policy_visualization_YYYYMMDD_clean.pdf / .png
    figures/policy_visualization_YYYYMMDD_delta.pdf / .png
"""

import os
import sys
import numpy as np
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv
import random

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "rl_logs_sb3", "run_20260513_071355",
    "best_model", "best_model",
)

EVAL_DAYS = [
    "496528674@qq.com_20161109",
    "496528674@qq.com_20161110",
    "496528674@qq.com_20161116",
    "496528674@qq.com_20161118",
    "496528674@qq.com_20161124",
    "496528674@qq.com_20161128",
]


def _run_and_collect(day: str, model: PPO) -> tuple:
    """Run RL agent on a day and collect per-step data."""
    random.seed(42)
    np.random.seed(42)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=DATA_ROOT, day_folders=[day])
    obs, _ = env.reset(seed=42)

    steps, lambda1, lambda2, rewards, sim_times = [], [], [], [], []
    done = False
    step_count = 0

    while not done:
        step_count += 1
        action, _ = model.predict(obs, deterministic=True)
        a = np.ravel(action)
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        lam = info.get("lambdas", [float(a[0]), float(a[1]), 1.0])
        steps.append(step_count * 5 / 60)
        lambda1.append(lam[0])
        lambda2.append(lam[1])
        rewards.append(float(reward))
        sim_times.append(info.get("current_time", 0.0))

    return (np.array(steps), np.array(lambda1), np.array(lambda2),
            np.array(rewards), sim_times)


def _draw_axes(ax1, steps, lambda1, lambda2):
    """Draw λ₁ (left) and λ₂ (right) lines. Returns ax2."""
    C1 = "#c0392b"
    ax1.plot(steps, lambda1, color=C1, linewidth=2.5, label=r"$\lambda_1$ (Fairness weight)")
    ax1.set_xlabel("Hours into Shift", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Fairness Weight $\lambda_1$", color=C1, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=C1)
    ax1.set_xlim(steps[0], steps[-1])
    ax1.set_ylim(lambda1.min() - 0.05, lambda1.max() + 0.05)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    C2 = "#e67e22"
    ax2.plot(steps, lambda2, color=C2, linewidth=2.0, linestyle="--",
             label=r"$\lambda_2$ (Starvation weight)")
    ax2.set_ylabel(r"Starvation Weight $\lambda_2$", color=C2, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=C2)
    ax2.set_ylim(lambda2.min() - 0.01, lambda2.max() + 0.01)

    return ax2


def _save(fig, path_stem: str):
    for ext in ("pdf", "png"):
        out = f"{path_stem}.{ext}"
        fig.savefig(out, format=ext, dpi=300, bbox_inches="tight")
        print(f"  ✅  {out}")
    plt.close(fig)


def _plot_clean(day, date_formatted, time_str, steps, lambda1, lambda2):
    """Plain white background — no shading."""
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = _draw_axes(ax1, steps, lambda1, lambda2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, framealpha=0.9)

    plt.title(
        f"DRL Governor: Weight Adaptation ({date_formatted}, {time_str})\n"
        r"Champion model, $\Delta$JFI reward",
        fontsize=13, pad=12,
    )
    fig.tight_layout()

    date_code = day.split("_")[-1]
    _save(fig, os.path.join(FIGURES_DIR, f"policy_visualization_{date_code}_clean"))


def _plot_delta(day, date_formatted, time_str, steps, lambda1, lambda2, rewards):
    """Background shading: green = reward improved vs previous step, red = declined."""
    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Δ reward shading: compare each step to the one before
    for i in range(len(steps) - 1):
        if i == 0:
            color = "white"   # no previous step to compare
        else:
            delta = rewards[i] - rewards[i - 1]
            color = "#d4edda" if delta >= 0 else "#f8d7da"
        ax1.axvspan(steps[i], steps[i + 1], alpha=0.35, color=color, linewidth=0)

    ax2 = _draw_axes(ax1, steps, lambda1, lambda2)

    patch_green = mpatches.Patch(color="#d4edda", alpha=0.7,
                                 label=r"Reward $\uparrow$ vs prev step")
    patch_red   = mpatches.Patch(color="#f8d7da", alpha=0.7,
                                 label=r"Reward $\downarrow$ vs prev step")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + [patch_green, patch_red],
               labels1 + labels2 + [r"Reward $\uparrow$ vs prev step",
                                     r"Reward $\downarrow$ vs prev step"],
               loc="upper right", fontsize=9, framealpha=0.9)

    plt.title(
        f"DRL Governor: Weight Adaptation ({date_formatted}, {time_str})\n"
        r"Champion model, $\Delta$JFI reward — shading = Δ reward vs previous step",
        fontsize=13, pad=12,
    )
    fig.tight_layout()

    date_code = day.split("_")[-1]
    _save(fig, os.path.join(FIGURES_DIR, f"policy_visualization_{date_code}_delta"))


def main():
    print("🔄 Loading champion model …")
    model = PPO.load(MODEL_PATH)

    for day in EVAL_DAYS:
        print(f"\n🎬 Evaluating {day} …")
        try:
            steps, lambda1, lambda2, rewards, sim_times = _run_and_collect(day, model)

            # Build time-of-day label from sim_times
            try:
                valid_times = [t for t in sim_times if t and t > 0]
                if valid_times:
                    start_dt = datetime.datetime.fromtimestamp(valid_times[0])
                    end_dt   = datetime.datetime.fromtimestamp(valid_times[-1])
                    time_str = f"{start_dt.strftime('%I:%M %p')} – {end_dt.strftime('%I:%M %p')}"
                else:
                    time_str = "8-hour shift"
            except Exception:
                time_str = "8-hour shift"

            date_str = day.split("_")[-1]
            date_formatted = f"Nov {int(date_str[-2:])}, 2016"

            _plot_clean(day, date_formatted, time_str, steps, lambda1, lambda2)
            _plot_delta(day, date_formatted, time_str, steps, lambda1, lambda2, rewards)

        except Exception as exc:
            print(f"  ❌ FAILED: {exc}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
