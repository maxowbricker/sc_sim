"""
Compare RL Agent vs. Static Baseline and Greedy

Use --log-weights PATH.txt for a single human-readable step trace (Step N: λ1=..., λ2=..., λ3=...).
Use --log-weights PATH.csv for CSV plus PATH_steps.txt (legacy).

Use --metrics-out PATH.txt to save the static vs RL vs greedy metric table without parsing console output.

Example: --model rl_logs_sb3/recent_run_21/rl_final_results/ppo_sc_final.zip
(do not use the literal placeholder path/to/model.zip)
"""
import argparse
import csv
import random
import sys
import os
import time
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from stable_baselines3 import PPO
from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv
from config import get_strategy_params


def resolve_model_load_path(model_path: str) -> str:
    """Resolve to absolute path; SB3 expects path *without* .zip suffix."""
    p = model_path.strip()
    if not os.path.isabs(p):
        p = os.path.join(PROJECT_ROOT, p.lstrip("./"))
    load_path = p[:-4] if p.endswith(".zip") else p
    zip_path = load_path + ".zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Model checkpoint not found.\n"
            f"  You passed: {model_path!r}\n"
            f"  Expected file: {zip_path}\n"
            f"  Use a real .zip from training, e.g. rl_logs_sb3/recent_run_21/rl_final_results/ppo_sc_final.zip"
        )
    return load_path


def _write_readable_steps_txt(log_rows, txt_path: str) -> str:
    """Write Step N: λ1=..., λ2=... lines to txt_path."""
    _dir = os.path.dirname(txt_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            "# RL policy output per environment step (λ1 = fairness, λ2 = starvation, λ3 = utility anchor)\n"
            "# One line per env step.\n\n"
        )
        for i, row in enumerate(log_rows, start=1):
            f.write(
                f"Step {i}: λ1={row['lambda1']:.6f}, λ2={row['lambda2']:.6f}, λ3={row['lambda3']:.6f} "
                f"(reward={row['reward']:.4f}, sim_time={row['sim_time']})\n"
            )
    return txt_path


TIER1_FAIRNESS_ROWS = [
    ("JFI (count)", "final_jains_fairness_index", True, 4),
    ("JFI_rate", "final_jfi_rate", True, 4),
    ("JFI_opportunity", "final_jfi_opportunity", True, 4),
    ("Gini (count)", "final_gini_coefficient", False, 4),
    ("Gini_rate", "final_gini_rate", False, 4),
    ("Gini_opportunity", "final_gini_opportunity", False, 4),
]

# Episode length for full-day eval. The 8 h window is a TRAINING device; for
# evaluation we drain the whole day so TAR is a real completion rate, not a
# fraction of the day that happened to fall inside the training window.
FULL_DAY_HOURS = 48.0
DEFAULT_EVAL_HOURS = 8.0

EFFICIENCY_ROWS = [
    ("Tasks Completed", "completed_tasks", True, 0, False),
    ("Tasks Released", "tasks_released", True, 0, False),
    ("TAR (released)", "completion_rate_released", True, 4, True),
    ("TAR (all-day)", "task_assignment_ratio", True, 4, True),
    ("Never released", "tasks_never_released", False, 0, False),
    ("In transit @ end", "tasks_in_transit_end", False, 0, False),
    ("Peak Backlog", "backlog_peak", False, 0, False),
    ("Avg Wait Time (m)", "avg_wait_time_minutes", False, 2, True),
    ("Avg pickup dist (km)", "avg_pickup_distance_km", False, 2, True),
]


def _stat_value(stats: dict, key: str):
    value = stats.get(key)
    if value is None:
        return None
    return float(value)


def _fmt_stat(value, decimals: int) -> str:
    if value is None:
        return "n/a"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def _trend(delta: float, higher_is_better: bool) -> str:
    if delta == 0:
        return "⚪"
    improved = delta > 0 if higher_is_better else delta < 0
    return "🟢" if improved else "🔴"


def _enable_tier1_fairness_diagnostics(simulator) -> None:
    simulator.metrics.enable_tier1_fairness_diagnostics()


def _comparison_table_lines(static_stats: dict, rl_stats: dict, greedy_stats: dict = None) -> list:
    width = 80 if greedy_stats is not None else 60
    col = 12 if greedy_stats is not None else 15
    header = (
        f"{'Metric':<20} | {'Static':<{col}} | {'RL Agent':<{col}} | "
        + (f"{'Greedy':<{col}} | " if greedy_stats is not None else "")
        + "RL vs Static\n"
    )
    lines = [
        "Static baseline vs RL agent"
        + (" vs Greedy" if greedy_stats is not None else "")
        + " (same day, eval_seed, protocol as console run)\n",
        "=" * width + "\n",
        header,
        "-" * width + "\n",
    ]

    for label, key, higher_is_better, decimals in TIER1_FAIRNESS_ROWS:
        static_val = _stat_value(static_stats, key)
        rl_val = _stat_value(rl_stats, key)
        greedy_val = _stat_value(greedy_stats, key) if greedy_stats is not None else None
        delta = (rl_val - static_val) if (rl_val is not None and static_val is not None) else 0.0
        trend = _trend(delta, higher_is_better)
        delta_txt = f"{trend} {delta:+.4f}" if rl_val is not None and static_val is not None else "n/a"
        if greedy_stats is not None:
            lines.append(
                f"{label:<20} | {_fmt_stat(static_val, decimals):<{col}} | "
                f"{_fmt_stat(rl_val, decimals):<{col}} | "
                f"{_fmt_stat(greedy_val, decimals):<{col}} | {delta_txt}\n"
            )
        else:
            lines.append(
                f"{label:<20} | {_fmt_stat(static_val, decimals):<{col}} | "
                f"{_fmt_stat(rl_val, decimals):<{col}} | {delta_txt}\n"
            )

    for label, key, higher_is_better, decimals, is_float in EFFICIENCY_ROWS:
        static_val = _stat_value(static_stats, key)
        rl_val = _stat_value(rl_stats, key)
        greedy_val = _stat_value(greedy_stats, key) if greedy_stats is not None else None
        delta = (rl_val - static_val) if (rl_val is not None and static_val is not None) else 0.0
        trend = _trend(delta, higher_is_better)
        delta_fmt = f"{delta:+.2f}" if is_float else f"{delta:+.0f}"
        delta_txt = f"{trend} {delta_fmt}" if rl_val is not None and static_val is not None else "n/a"
        if greedy_stats is not None:
            lines.append(
                f"{label:<20} | {_fmt_stat(static_val, decimals):<{col}} | "
                f"{_fmt_stat(rl_val, decimals):<{col}} | "
                f"{_fmt_stat(greedy_val, decimals):<{col}} | {delta_txt}\n"
            )
        else:
            lines.append(
                f"{label:<20} | {_fmt_stat(static_val, decimals):<{col}} | "
                f"{_fmt_stat(rl_val, decimals):<{col}} | {delta_txt}\n"
            )

    lines.append("=" * width + "\n")
    return lines


def _print_comparison_table(static_stats: dict, rl_stats: dict, greedy_stats: dict = None) -> None:
    for line in _comparison_table_lines(static_stats, rl_stats, greedy_stats):
        print(line, end="")


def _write_metrics_txt(path: str, static_stats: dict, rl_stats: dict, greedy_stats: dict = None) -> None:
    """Save the static vs RL (vs greedy) comparison table as plain text."""
    out = path
    if not os.path.isabs(out):
        out = os.path.join(PROJECT_ROOT, out.lstrip("./"))
    _dir = os.path.dirname(out)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(_comparison_table_lines(static_stats, rl_stats, greedy_stats))


def _run_simulator_episode(env, strategy: str, strategy_params=None):
    """
    Run the post-warmup RL phase via EventSimulator directly.

    On oracle-approach, gym step() always runs the greedy twin before composite,
    so pure-greedy eval must not use env.step().
    """
    if strategy_params is None:
        strategy_params = {}
    env.simulator.switch_strategy(strategy, strategy_params)
    done = False
    while not done:
        sim_done = env.simulator.step(duration_seconds=env.step_duration)
        env.current_step_idx += 1
        if env.episode_end_time and env.simulator.current_time >= env.episode_end_time:
            done = True
        elif sim_done:
            done = True
    return env.simulator.get_final_results()


def run_static_baseline(day, data_root, eval_seed: int = 42, episode_hours: float = DEFAULT_EVAL_HOURS):
    """
    Same protocol as RL: greedy warmup → composite with config weights, fixed λ1/λ2.
    episode_hours large (full-day) drains the queue so TAR is a real completion rate.
    """
    print(f"\n[1/3] 🏃 Static baseline (aligned env: warmup + fixed config λ) for {day}...")
    print(f"      eval_seed={eval_seed} (must match RL run for same scenario)")
    t0 = time.time()

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(
        data_root=data_root, day_folders=[day], episode_duration_hours=episode_hours
    )
    env.reset(seed=eval_seed)
    _enable_tier1_fairness_diagnostics(env.simulator)

    sp = get_strategy_params("composite")
    sp["normalize_scores"] = True
    sp["enable_deferral_tracking"] = True

    w_fair = sp.get("fairness_weight", "N/A")
    w_starv = sp.get("starvation_weight", "N/A")
    w_util = sp.get("utility_weight", "N/A")
    print(f"      ⚖️  Fixed weights from config.py: λ1={w_fair}, λ2={w_starv}, λ3={w_util}")

    action = np.array([float(sp["fairness_weight"]), float(sp["starvation_weight"])], dtype=np.float32)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    stats = env.simulator.get_final_results()
    print(f"      ✅ Finished in {time.time() - t0:.2f} seconds.")
    return stats


def run_greedy_full_episode(day, data_root, eval_seed: int = 42, episode_hours: float = DEFAULT_EVAL_HOURS):
    """Greedy warmup → pure greedy for the episode window (full-day if episode_hours large)."""
    print(f"\n[2/3] 🟢 Greedy baseline (warmup + pure greedy episode) for {day}...")
    print(f"      eval_seed={eval_seed}")
    t0 = time.time()

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(
        data_root=data_root, day_folders=[day], episode_duration_hours=episode_hours
    )
    env.reset(seed=eval_seed)
    _enable_tier1_fairness_diagnostics(env.simulator)
    print("      🟢 Pure greedy (no composite weights)")

    stats = _run_simulator_episode(env, "greedy", {})
    print(f"      ✅ Finished in {time.time() - t0:.2f} seconds.")
    return stats


def run_rl_agent(
    model_path,
    day,
    data_root,
    quiet=False,
    weights_log_path=None,
    print_weights=False,
    eval_seed: int = 42,
    episode_hours: float = DEFAULT_EVAL_HOURS,
):
    print(f"\n[3/3] 🧠 Running RL Agent ({model_path}) for {day}...")
    print(f"      eval_seed={eval_seed} (must match baselines for same scenario)")
    start_time = time.time()

    load_path = resolve_model_load_path(model_path)

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(
        data_root=data_root, day_folders=[day], episode_duration_hours=episode_hours
    )
    model = PPO.load(load_path)

    obs, _ = env.reset(seed=eval_seed)
    _enable_tier1_fairness_diagnostics(env.simulator)
    log_rows = []
    done = False
    step_num = 0
    while not done:
        step_num += 1
        action, _ = model.predict(obs, deterministic=True)
        a = np.ravel(action)  # Handle (2,) or (1,2) from DummyVecEnv
        obs, reward, terminated, truncated, info = env.step(a)
        lam = info.get("lambdas", [float(a[0]), float(a[1]), env.lambda3_fixed])
        if not quiet:
            print(f"      🤖 Agent chose weights: λ1 (Fairness)={lam[0]:.2f}, λ2 (Starvation)={lam[1]:.2f}")
        if weights_log_path is not None:
            log_rows.append({
                "env_step": info.get("step", len(log_rows) + 1),
                "sim_time": info.get("current_time"),
                "lambda1": float(lam[0]),
                "lambda2": float(lam[1]),
                "lambda3": float(lam[2]),
                "reward": float(reward),
            })
        if print_weights:
            print(f"      Step {step_num}: λ1={float(lam[0]):.4f}, λ2={float(lam[1]):.4f}")
        done = terminated or truncated

    if weights_log_path is not None and log_rows:
        out = weights_log_path
        if not os.path.isabs(out):
            out = os.path.join(PROJECT_ROOT, out.lstrip("./"))
        if out.lower().endswith(".txt"):
            _write_readable_steps_txt(log_rows, out)
            print(f"      📝 Weight trace: {out} ({len(log_rows)} steps)")
        else:
            _dir = os.path.dirname(out)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            fieldnames = ["env_step", "sim_time", "lambda1", "lambda2", "lambda3", "reward"]
            with open(out, "w", newline="") as cf:
                w = csv.DictWriter(cf, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(log_rows)
            base = os.path.splitext(out)[0]
            txt_out = _write_readable_steps_txt(log_rows, base + "_steps.txt")
            print(f"      📝 Weight trace CSV: {out} ({len(log_rows)} steps)")
            print(f"      📝 Step-by-step text: {txt_out}")

    stats = env.simulator.get_final_results()
    print(f"      ✅ Finished in {time.time() - start_time:.2f} seconds.")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare RL model against Static and Greedy baselines")
    parser.add_argument("--model", type=str, required=True, help="Path to the PPO .zip model")
    parser.add_argument("--day", type=str, default="496528674@qq.com_20161128", help="Test day folder")
    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia"),
                        help="Path to DiDi day folders (default: <project>/data/didi/full_didi_gaia)")
    parser.add_argument("--stratified", type=lambda x: x.lower() == "true", default=None,
                        help="Override stratified sampling: true or false (default: use config.py)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step agent action prints")
    parser.add_argument(
        "--log-weights",
        type=str,
        default=None,
        metavar="PATH",
        help="Per-step λ trace: use .txt for one readable file, or .csv for CSV plus *_steps.txt (legacy)",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Write static vs RL vs greedy metric table to this .txt (project-root relative OK)",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Only run the RL episode (skip static and greedy baselines)",
    )
    parser.add_argument(
        "--no-greedy",
        action="store_true",
        help="Skip greedy baseline (static vs RL table only)",
    )
    parser.add_argument(
        "--print-weights",
        action="store_true",
        help="Print each step to the terminal as 'Step N: λ1=..., λ2=...' (use with --quiet to only see weights)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="RNG seed for env.reset (same drop-in + warmup for all arms). Default: 42",
    )
    parser.add_argument(
        "--full-day",
        action="store_true",
        help="Drain the whole day (start at day open) instead of the 8h training window. "
             "Makes TAR a real completion rate, comparable to conference-ready 6-day eval.",
    )
    args = parser.parse_args()

    episode_hours = FULL_DAY_HOURS if args.full_day else DEFAULT_EVAL_HOURS

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(PROJECT_ROOT, args.data_root.lstrip("./"))

    try:
        resolve_model_load_path(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    if args.stratified is not None:
        import config as config_module
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified
        print(f"   📋 Stratified sampling: {'ON' if args.stratified else 'OFF'} (override)")

    print("=" * 60)
    print(f"🧪 EVALUATION BENCHMARK: {args.day}")
    window_desc = "FULL DAY (queue drain)" if args.full_day else "8h training window"
    print(f"   Fair protocol: greedy warmup + {window_desc}, eval_seed={args.eval_seed}")
    print("=" * 60)

    static_stats = None
    greedy_stats = None
    if not args.rl_only:
        static_stats = run_static_baseline(args.day, args.data_root, eval_seed=args.eval_seed, episode_hours=episode_hours)
        if not args.no_greedy:
            greedy_stats = run_greedy_full_episode(args.day, args.data_root, eval_seed=args.eval_seed, episode_hours=episode_hours)


    rl_stats = run_rl_agent(
        args.model,
        args.day,
        args.data_root,
        quiet=args.quiet,
        weights_log_path=args.log_weights,
        print_weights=args.print_weights,
        episode_hours=episode_hours,
        eval_seed=args.eval_seed,
    )

    if args.rl_only:
        print("\n" + "=" * 60)
        print("RL-only run complete (no baseline comparison).")
        for label, key, _, decimals in TIER1_FAIRNESS_ROWS:
            print(f"  {label}: {_fmt_stat(_stat_value(rl_stats, key), decimals)}")
        print(f"  tasks_completed: {rl_stats.get('completed_tasks', 'n/a')}")
        print(f"  backlog_peak: {rl_stats.get('backlog_peak', 'n/a')}")
        print(f"  avg_wait_time_minutes: {rl_stats.get('avg_wait_time_minutes', 'n/a')}")
        print(f"  avg_pickup_distance_km: {rl_stats.get('avg_pickup_distance_km', 'n/a')}")
        print("=" * 60 + "\n")
        return

    print()
    _print_comparison_table(static_stats, rl_stats, greedy_stats)
    print()

    if args.metrics_out:
        _write_metrics_txt(args.metrics_out, static_stats, rl_stats, greedy_stats)
        mo = args.metrics_out
        if not os.path.isabs(mo):
            mo = os.path.join(PROJECT_ROOT, mo.lstrip("./"))
        print(f"   📝 Metrics written to: {mo}")


if __name__ == "__main__":
    main()
