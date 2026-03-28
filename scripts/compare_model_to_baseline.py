"""
Compare RL Agent vs. Static Baseline (FATP/Composite)

Use --log-weights PATH.csv to record λ1, λ2 per step; also writes PATH_steps.txt (Step 1: λ1=..., λ2=...).
Use --print-weights to print the same lines to the terminal (good with --quiet).

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


def _write_readable_steps_txt(log_rows, csv_path: str) -> str:
    """Write traces/foo_steps.txt with Step N: λ1=..., λ2=... next to traces/foo.csv."""
    base = os.path.splitext(csv_path)[0]
    txt_path = base + "_steps.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# RL policy output per environment step (λ1 = fairness, λ2 = starvation, λ3 = fixed utility anchor)\n")
        f.write("# Columns match the CSV row order.\n\n")
        for i, row in enumerate(log_rows, start=1):
            f.write(
                f"Step {i}: λ1={row['lambda1']:.6f}, λ2={row['lambda2']:.6f}, λ3={row['lambda3']:.6f} "
                f"(reward={row['reward']:.4f}, sim_time={row['sim_time']})\n"
            )
    return txt_path


def run_static_baseline(day, data_root, eval_seed: int = 42):
    """
    Same protocol as RL: greedy warmup → composite with config weights, fixed λ1/λ2 for
    episode_duration_hours × 5-min steps. Uses the same eval_seed as RL for identical
    drop-in time and RNG (fair vs run_rl_agent).
    """
    print(f"\n[1/2] 🏃 Static baseline (aligned env: warmup + fixed config λ) for {day}...")
    print(f"      eval_seed={eval_seed} (must match RL run for same scenario)")
    t0 = time.time()

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=data_root, day_folders=[day])
    obs, _ = env.reset(seed=eval_seed)

    sp = get_strategy_params("composite")
    sp["normalize_scores"] = True
    sp["enable_deferral_tracking"] = True
    env.simulator.switch_strategy("composite", sp)

    w_fair = sp.get("fairness_weight", "N/A")
    w_starv = sp.get("starvation_weight", "N/A")
    w_util = sp.get("utility_weight", "N/A")
    print(f"      ⚖️  Fixed weights from config.py: λ1={w_fair}, λ2={w_starv}, λ3={w_util}")

    action = np.array([float(sp["fairness_weight"]), float(sp["starvation_weight"])], dtype=np.float32)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    stats = env.simulator.get_final_results()
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
):
    print(f"\n[2/2] 🧠 Running RL Agent ({model_path}) for {day}...")
    print(f"      eval_seed={eval_seed} (must match static baseline for same scenario)")
    start_time = time.time()

    # Fail fast before loading dataset / building env
    load_path = resolve_model_load_path(model_path)

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    env = AdaptiveSpatialCrowdsourcingEnv(data_root=data_root, day_folders=[day])
    # Load policy only — avoids SB3 wrapping env twice vs our controlled reset(seed=...)
    model = PPO.load(load_path)

    obs, _ = env.reset(seed=eval_seed)
    log_rows = []
    done = False
    step_num = 0
    while not done:
        step_num += 1
        action, _ = model.predict(obs, deterministic=True)
        a = np.ravel(action)  # Handle (2,) or (1,2) from DummyVecEnv
        if not quiet:
            print(f"      🤖 Agent chose weights: λ1 (Fairness)={a[0]:.2f}, λ2 (Starvation)={a[1]:.2f}")
        obs, reward, terminated, truncated, info = env.step(a)
        if weights_log_path is not None:
            lam = info.get("lambdas", [float(a[0]), float(a[1]), env.lambda3_fixed])
            log_rows.append({
                "env_step": info.get("step", len(log_rows) + 1),
                "sim_time": info.get("current_time"),
                "lambda1": float(lam[0]),
                "lambda2": float(lam[1]),
                "lambda3": float(lam[2]),
                "reward": float(reward),
            })
        if print_weights:
            print(f"      Step {step_num}: λ1={float(a[0]):.4f}, λ2={float(a[1]):.4f}")
        done = terminated or truncated

    if weights_log_path is not None and log_rows:
        out = weights_log_path
        if not os.path.isabs(out):
            out = os.path.join(PROJECT_ROOT, out.lstrip("./"))
        _dir = os.path.dirname(out)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        fieldnames = ["env_step", "sim_time", "lambda1", "lambda2", "lambda3", "reward"]
        with open(out, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(log_rows)
        txt_out = _write_readable_steps_txt(log_rows, out)
        print(f"      📝 Weight trace CSV: {out} ({len(log_rows)} steps)")
        print(f"      📝 Step-by-step text: {txt_out}")

    # NEW: Get the true final results for the whole day
    stats = env.simulator.get_final_results()
    print(f"      ✅ Finished in {time.time() - start_time:.2f} seconds.")
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compare RL model against Static Baseline")
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
        metavar="CSV_PATH",
        help="Write λ1/λ2/λ3, sim_time, reward per env step to this CSV (project-root relative OK)",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Only run the RL episode (skip static baseline); still writes --log-weights if set",
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
        help="RNG seed for env.reset (same drop-in + warmup for static baseline and RL). Default: 42",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(PROJECT_ROOT, args.data_root.lstrip("./"))

    try:
        resolve_model_load_path(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    # Override config before any data loading
    if args.stratified is not None:
        import config as config_module
        config_module.DATA_SAMPLING["use_stratified_sampling"] = args.stratified
        print(f"   📋 Stratified sampling: {'ON' if args.stratified else 'OFF'} (override)")

    print("="*60)
    print(f"🧪 EVALUATION BENCHMARK: {args.day}")
    print(f"   Fair protocol: greedy warmup + composite episode (see gym env), eval_seed={args.eval_seed}")
    print("="*60)

    static_stats = None
    if not args.rl_only:
        static_stats = run_static_baseline(args.day, args.data_root, eval_seed=args.eval_seed)
    rl_stats = run_rl_agent(
        args.model,
        args.day,
        args.data_root,
        quiet=args.quiet,
        weights_log_path=args.log_weights,
        print_weights=args.print_weights,
        eval_seed=args.eval_seed,
    )

    if args.rl_only:
        print("\n" + "="*60)
        print("RL-only run complete (no baseline comparison).")
        print(f"  final_jains_fairness_index: {rl_stats.get('final_jains_fairness_index', 'n/a')}")
        print(f"  backlog_peak: {rl_stats.get('backlog_peak', 'n/a')}")
        print(f"  avg_wait_time_minutes: {rl_stats.get('avg_wait_time_minutes', 'n/a')}")
        print("="*60 + "\n")
        return

    # Use the correct dictionary keys from get_final_results()
    jfi_delta = rl_stats['final_jains_fairness_index'] - static_stats['final_jains_fairness_index']
    backlog_delta = rl_stats['backlog_peak'] - static_stats['backlog_peak']
    wait_delta = rl_stats['avg_wait_time_minutes'] - static_stats['avg_wait_time_minutes']

    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Static Baseline':<15} | {'RL Agent':<15} | {'Improvement'}")
    print("-" * 60)

    jfi_trend = "🟢" if jfi_delta > 0 else "🔴"
    print(f"{'JFI (Fairness)':<20} | {static_stats['final_jains_fairness_index']:<15.4f} | {rl_stats['final_jains_fairness_index']:<15.4f} | {jfi_trend} {jfi_delta:+.4f}")

    backlog_trend = "🟢" if backlog_delta < 0 else "🔴"
    print(f"{'Peak Backlog':<20} | {static_stats['backlog_peak']:<15.0f} | {rl_stats['backlog_peak']:<15.0f} | {backlog_trend} {backlog_delta:+.0f}")

    wait_trend = "🟢" if wait_delta < 0 else "🔴"
    print(f"{'Avg Wait Time (m)':<20} | {static_stats['avg_wait_time_minutes']:<15.2f} | {rl_stats['avg_wait_time_minutes']:<15.2f} | {wait_trend} {wait_delta:+.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
