"""
Train PPO agent for Adaptive Spatial Crowdsourcing using Stable Baselines 3.

Usage:
    python rl/train_sb3.py [--timesteps N] [--resume PATH] [--test] [--hyperparams PATH]
    
Examples:
    # Quick test run (1000 timesteps)
    python rl/train_sb3.py --timesteps 1000
    
    # Full training run
    python rl/train_sb3.py --timesteps 50000
    
    # Resume from checkpoint
    python rl/train_sb3.py --resume rl_logs_sb3/ppo_sc_model_1000_steps.zip --timesteps 50000
    
    # Use Optuna-tuned hyperparameters (default: rl/best_hyperparameters.json)
    python rl/train_sb3.py --hyperparams best_hyperparameters.json --timesteps 50000

Each run folder gets: gym_environment_snapshot.py, hyperparams_snapshot.json, environment_spec.json
(obs/action spaces, reward_weights, timing), run_manifest.json, BASELINE_EVAL_README.txt.
After training, compare_model_to_baseline runs for final + best models (unless --no-post-eval), writing
baseline_eval_*.txt (full console log), baseline_*_model_weight_outputs.txt (per-step λ),
baseline_*_model_metrics.txt (static vs RL table).
Copy the whole folder when syncing from a remote machine.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sklearn.model_selection import train_test_split
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.gym_environment import AdaptiveSpatialCrowdsourcingEnv
from gymnasium import spaces as gym_spaces

# train_test_split(..., random_state=42, shuffle=True) is reproducible for a fixed list of day folders.
# Post-training baseline comparison uses one held-out day: this folder if it appears in the test split,
# otherwise the lexicographically first test day (deterministic, always in the test set).
DEFAULT_BASELINE_EVAL_DAY = "496528674@qq.com_20161128"


def _resolve_hyperparams_path(project_root: Path, hyperparams_arg: Optional[str]) -> Path:
    default = project_root / "rl" / "best_hyperparameters.json"
    if hyperparams_arg:
        return Path(hyperparams_arg).resolve()
    return default

def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_meta(project_root: Path) -> dict:
    meta: dict = {"commit": None, "dirty": None}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            meta["commit"] = r.stdout.strip()
        r2 = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r2.returncode == 0:
            meta["dirty"] = bool(r2.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return meta

def write_run_artifacts(log_dir: str, project_root: Path, args, hyperparams_path: Path) -> None:
    """
    Copy reward/env code and hyperparams into this run folder so rsync'd runs stay self-contained.
    Writes run_manifest.json (argv, git, file hashes) and a short note for baseline eval output.
    """
    gym_src = project_root / "rl" / "gym_environment.py"
    gym_dst = Path(log_dir) / "gym_environment_snapshot.py"
    hp_dst = Path(log_dir) / "hyperparams_snapshot.json"

    if gym_src.is_file():
        shutil.copy2(gym_src, gym_dst)
    else:
        print(f"   ⚠️  Could not snapshot {gym_src} (missing)")

    if hyperparams_path.is_file():
        shutil.copy2(hyperparams_path, hp_dst)
    else:
        print(f"   ⚠️  Could not snapshot hyperparams {hyperparams_path} (missing)")

    manifest = {
        "run_folder": os.path.basename(log_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git": _git_meta(project_root),
        "hyperparams_source": str(hyperparams_path),
        "artifacts": {
            "gym_environment_snapshot": "gym_environment_snapshot.py",
            "hyperparams_snapshot": "hyperparams_snapshot.json"
            if hyperparams_path.is_file()
            else None,
            "gym_sha256": _sha256_file(gym_dst) if gym_dst.is_file() else None,
            "hyperparams_sha256": _sha256_file(hp_dst) if hp_dst.is_file() else None,
        },
        "training": {
            "timesteps": args.timesteps,
            "resume": args.resume,
            "data_root": args.data_root,
            "train_days": args.train_days,
            "num_cpu": args.num_cpu,
            "no_parallel": args.no_parallel,
            "test": args.test,
        },
    }
    manifest_path = Path(log_dir) / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Hint for after you sync the run locally or on the server
    note = Path(log_dir) / "BASELINE_EVAL_README.txt"
    try:
        run_rel = Path(os.path.relpath(Path(log_dir).resolve(), project_root.resolve())).as_posix()
    except ValueError:
        run_rel = Path(log_dir).as_posix()
    note.write_text(
        "Training normally fills this folder automatically (baseline_eval_*.txt, weight_outputs, metrics). "
        "Re-run manually if needed.\n\n"
        "Layout (same day / --eval-seed for fair comparison):\n"
        "  environment_spec.json    (obs/action space + reward_weights; full reward code in gym snapshot)\n"
        "  baseline_final_model_weight_outputs.txt, baseline_best_model_weight_outputs.txt (per-step λ)\n"
        "  baseline_final_model_metrics.txt, baseline_best_model_metrics.txt (static vs RL table)\n"
        "  baseline_eval_final.txt, baseline_eval_best.txt (full stdout from compare script)\n\n"
        "Manual examples (run from repo root; replace YOUR_DAY_FOLDER):\n\n"
        f"  python3 scripts/compare_model_to_baseline.py \\\n"
        f"    --model {run_rel}/ppo_sc_final \\\n"
        f"    --day YOUR_DAY_FOLDER \\\n"
        f"    --data-root data/didi/full_didi_gaia \\\n"
        f"    --eval-seed 42 \\\n"
        f"    --log-weights {run_rel}/baseline_final_model_weight_outputs.txt \\\n"
        f"    --metrics-out {run_rel}/baseline_final_model_metrics.txt \\\n"
        f"    > {run_rel}/baseline_eval_final.txt 2>&1\n\n"
        f"  python3 scripts/compare_model_to_baseline.py \\\n"
        f"    --model {run_rel}/best_model/best_model \\\n"
        f"    --day YOUR_DAY_FOLDER \\\n"
        f"    --data-root data/didi/full_didi_gaia \\\n"
        f"    --eval-seed 42 \\\n"
        f"    --log-weights {run_rel}/baseline_best_model_weight_outputs.txt \\\n"
        f"    --metrics-out {run_rel}/baseline_best_model_metrics.txt \\\n"
        f"    > {run_rel}/baseline_eval_best.txt 2>&1\n",
        encoding="utf-8",
    )
    print(
        f"   Run artifacts: {gym_dst.name}, "
        f"{hp_dst.name if hp_dst.is_file() else 'no hyperparams'}, "
        f"{manifest_path.name}, {note.name}"
    )


def merge_manifest_artifacts(log_dir: str, updates: dict) -> None:
    """Add filenames under manifest['artifacts'] (e.g. environment_spec.json after env init)."""
    manifest_path = Path(log_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    manifest.setdefault("artifacts", {}).update(updates)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _gym_space_to_dict(space) -> dict:
    if isinstance(space, gym_spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape) if space.shape is not None else None,
            "dtype": str(space.dtype),
            "low": np.asarray(space.low).tolist(),
            "high": np.asarray(space.high).tolist(),
        }
    return {"type": space.__class__.__name__, "repr": str(space)}


def write_environment_spec(log_dir: str, env) -> None:
    """
    JSON snapshot of observation/action spaces and key fields tied to reward shaping.
    Reward formula itself lives in gym_environment_snapshot.py::_calculate_reward.
    """
    spec: dict = {
        "observation_space": _gym_space_to_dict(env.observation_space),
        "action_space": _gym_space_to_dict(env.action_space),
        "reward_note": (
            "reward_weights below are (fairness, starvation, latency) multipliers; "
            "see _calculate_reward() in gym_environment_snapshot.py for the full formula."
        ),
    }

    if hasattr(env, "get_attr"):
        def _ga(name: str):
            return env.get_attr(name, 0)[0]

        def _ga_opt(name: str, default):
            try:
                return env.get_attr(name, 0)[0]
            except Exception:
                return default

        spec["env_runtime"] = {
            "reward_weights": list(_ga("reward_weights")),
            "lambda3_fixed": float(_ga("lambda3_fixed")),
            "step_duration_minutes": float(_ga("step_duration")) / 60.0,
            "warmup_duration_seconds": int(_ga("warmup_duration_seconds")),
            "episode_duration_seconds": int(_ga("episode_duration_seconds")),
        }
        _sla = _ga_opt("sla_wait_time_minutes", None)
        _pen = _ga_opt("sla_violation_penalty", None)
        _djs = _ga_opt("delta_jfi_reward_scale", None)
        if _sla is not None:
            spec["env_runtime"]["sla_wait_time_minutes"] = float(_sla)
        if _pen is not None:
            spec["env_runtime"]["sla_violation_penalty"] = float(_pen)
        if _djs is not None:
            spec["env_runtime"]["delta_jfi_reward_scale"] = float(_djs)
        try:
            spec["env_runtime"]["obs_scaling"] = _json_safe(_ga("obs_scaling"))
        except Exception:
            pass
    else:
        inner = _unwrap_env_instance(env)
        if inner is not None:
            spec["env_runtime"] = {
                "reward_weights": list(inner.reward_weights),
                "lambda3_fixed": float(inner.lambda3_fixed),
                "step_duration_minutes": float(inner.step_duration) / 60.0,
                "warmup_duration_seconds": int(inner.warmup_duration_seconds),
                "episode_duration_seconds": int(inner.episode_duration_seconds),
                "sla_wait_time_minutes": float(
                    getattr(inner, "sla_wait_time_minutes", 3.0)
                ),
                "sla_violation_penalty": float(
                    getattr(inner, "sla_violation_penalty", 20.0)
                ),
                "delta_jfi_reward_scale": float(
                    getattr(inner, "delta_jfi_reward_scale", 1000.0)
                ),
                "obs_scaling": _json_safe(getattr(inner, "obs_scaling", None))
                if getattr(inner, "obs_scaling", None) is not None
                else None,
            }

    out = Path(log_dir) / "environment_spec.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    merge_manifest_artifacts(log_dir, {"environment_spec": "environment_spec.json"})
    print(f"   Wrote environment_spec.json (spaces + reward_weights / timing)")


def _unwrap_env_instance(env) -> Optional[AdaptiveSpatialCrowdsourcingEnv]:
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e if isinstance(e, AdaptiveSpatialCrowdsourcingEnv) else None


def resolve_baseline_eval_day(
    test_days: List[str],
    explicit_eval_day: Optional[str],
) -> Tuple[str, str]:
    """
    Choose a single day for compare_model_to_baseline. Must be in the held-out test_days set.

    - If --eval-day is set: must appear in test_days.
    - Else if DEFAULT_BASELINE_EVAL_DAY is in test_days: use it (stable across runs for the same split).
    - Else: sorted(test_days)[0] (deterministic fallback, always held out).
    """
    if not test_days:
        raise ValueError("test_days is empty")
    test_sorted = sorted(test_days)
    explicit = (explicit_eval_day or "").strip() or None
    if explicit:
        if explicit not in test_days:
            preview = test_sorted[:8]
            more = "..." if len(test_sorted) > 8 else ""
            raise ValueError(
                f"--eval-day {explicit!r} is not in the held-out test set. "
                f"Test days ({len(test_sorted)}): {preview}{more}"
            )
        return explicit, "explicit_cli"
    if DEFAULT_BASELINE_EVAL_DAY in test_days:
        return DEFAULT_BASELINE_EVAL_DAY, f"preferred_in_test:{DEFAULT_BASELINE_EVAL_DAY}"
    return test_sorted[0], (
        f"fallback_lexicographic_first (preferred {DEFAULT_BASELINE_EVAL_DAY!r} not in test set)"
    )


def update_run_manifest_baseline_eval(
    log_dir: str,
    baseline_eval_day: str,
    test_days: List[str],
    selection_rule: str,
) -> None:
    """Record which test day is used for post-training baseline comparison (reproducibility)."""
    manifest_path = Path(log_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    test_sorted = sorted(test_days)
    manifest["baseline_eval"] = {
        "day": baseline_eval_day,
        "selection_rule": selection_rule,
        "test_days_sorted": test_sorted,
        "train_test_split": {"random_state": 42, "shuffle": True},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def append_manifest_post_eval(
    log_dir: str,
    eval_day: str,
    eval_seed: int,
    data_root_arg: str,
    results: List[dict],
) -> None:
    """Record post-training baseline comparison outcomes in run_manifest.json."""
    manifest_path = Path(log_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    manifest["post_eval"] = {
        "eval_day": eval_day,
        "eval_seed": eval_seed,
        "data_root": data_root_arg,
        "runs": results,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_post_training_baseline_eval(
    log_dir: str,
    project_root: Path,
    eval_day: str,
    data_root_cli: str,
    eval_seed: int,
    quiet: bool,
) -> None:
    """
    Run scripts/compare_model_to_baseline.py for final and best checkpoints;
    writes weight .txt, metrics .txt, and full-console baseline_eval_*.txt under log_dir.
    """
    try:
        run_rel = Path(os.path.relpath(Path(log_dir).resolve(), project_root.resolve())).as_posix()
    except ValueError:
        run_rel = Path(log_dir).as_posix()

    script = project_root / "scripts" / "compare_model_to_baseline.py"
    if not script.is_file():
        print(f"   ⚠️  Post-eval skipped: {script} not found")
        return

    final_model = f"{run_rel}/ppo_sc_final"
    best_model = f"{run_rel}/best_model/best_model"
    best_zip = Path(log_dir) / "best_model" / "best_model.zip"

    jobs = [
        (
            "final",
            final_model,
            "baseline_final_model_weight_outputs.txt",
            "baseline_final_model_metrics.txt",
            "baseline_eval_final.txt",
        ),
    ]
    if best_zip.is_file():
        jobs.append(
            (
                "best",
                best_model,
                "baseline_best_model_weight_outputs.txt",
                "baseline_best_model_metrics.txt",
                "baseline_eval_best.txt",
            )
        )
    else:
        print(f"   ⚠️  No {best_zip}, skipping best-model post-eval")

    results: List[dict] = []
    print("\n" + "=" * 80)
    print("📋 POST-TRAINING BASELINE COMPARISON (compare_model_to_baseline.py)")
    print("=" * 80)

    for label, model_arg, weights_name, metrics_name, out_name in jobs:
        out_path = Path(log_dir) / out_name
        weights_arg = f"{run_rel}/{weights_name}"
        metrics_arg = f"{run_rel}/{metrics_name}"
        cmd = [
            sys.executable,
            str(script),
            "--model",
            model_arg,
            "--day",
            eval_day,
            "--data-root",
            data_root_cli,
            "--eval-seed",
            str(eval_seed),
            "--log-weights",
            weights_arg,
            "--metrics-out",
            metrics_arg,
        ]
        if quiet:
            cmd.append("--quiet")

        print(f"\n   Running [{label}]: model={model_arg}")
        print(f"   Log → {out_path}")
        with open(out_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )
        results.append(
            {
                "label": label,
                "model": model_arg,
                "log_weights": weights_arg,
                "metrics_out": metrics_arg,
                "stdout_txt": out_name,
                "returncode": proc.returncode,
            }
        )
        if proc.returncode != 0:
            print(f"   ❌ Post-eval [{label}] exited with code {proc.returncode} (see {out_name})")
        else:
            print(f"   ✅ Post-eval [{label}] complete")

    append_manifest_post_eval(log_dir, eval_day, eval_seed, data_root_cli, results)
    print("=" * 80 + "\n")


def make_env(data_root, day_folders, rank=0, step_duration_minutes=5, reward_weights=None,
             warmup_duration_minutes=30, episode_duration_hours=8):
    """
    Utility function for multiprocessed env.
    
    Args:
        data_root: Base path to dataset folders
        day_folders: List of folder names to randomly select from
        rank: Index of the subprocess (useful for seeding)
        step_duration_minutes: Duration of each simulation step (default: 5 minutes)
        reward_weights: Weights for reward components
        warmup_duration_minutes: Duration of warmup phase (default: 30 minutes)
        episode_duration_hours: Duration of RL episode after warmup (default: 8 hours)
    """
    def _init():
        env = AdaptiveSpatialCrowdsourcingEnv(
            dataset="didi",
            step_duration_minutes=step_duration_minutes,
            reward_weights=reward_weights or [1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=day_folders,
            warmup_duration_minutes=warmup_duration_minutes,
            episode_duration_hours=episode_duration_hours
        )
        return env
    return _init

def create_env(dataset="didi", step_duration_minutes=5, reward_weights=None, 
               data_root=None, day_folders=None, warmup_duration_minutes=30,
               episode_duration_hours=8):
    """Create and wrap environment for training (legacy single-env mode)."""
    env = AdaptiveSpatialCrowdsourcingEnv(
        dataset=dataset,
        step_duration_minutes=step_duration_minutes,
        reward_weights=reward_weights or [1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=day_folders,
        warmup_duration_minutes=warmup_duration_minutes,
        episode_duration_hours=episode_duration_hours
    )
    return env

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Spatial Crowdsourcing")
    parser.add_argument("--timesteps", type=int, default=10000, 
                       help="Total number of training timesteps (default: 10000)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--test", action="store_true",
                       help="Quick test run with minimal timesteps (1000)")
    parser.add_argument("--log-dir", type=str, default="rl_logs_sb3",
                       help="Directory for logs and checkpoints (default: rl_logs_sb3)")
    parser.add_argument("--skip-env-check", action="store_true",
                       help="Skip environment compatibility check (faster startup)")
    parser.add_argument("--data-root", type=str, default="./data/didi/full_didi_gaia",
                       help="Base path to dataset folders (default: ./data/didi/full_didi_gaia)")
    parser.add_argument("--num-cpu", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--train-days", type=int, default=24,
                       help="Number of days to use for training (default: 24)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel environments (use single env)")
    parser.add_argument("--hyperparams", type=str, default=None,
                       help="Path to best_hyperparameters.json from Optuna tuning (default: rl/best_hyperparameters.json)")
    parser.add_argument(
        "--eval-day",
        type=str,
        default=None,
        help=(
            "Day folder for post-training compare_model_to_baseline; must be in the held-out test set. "
            f"Default: {DEFAULT_BASELINE_EVAL_DAY} if that day is in the test split, else lexicographically "
            "first test day (split uses random_state=42)."
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="RNG seed for post-training baseline comparison (default: 42)",
    )
    parser.add_argument(
        "--no-post-eval",
        action="store_true",
        help="Skip compare_model_to_baseline after successful training",
    )
    parser.add_argument(
        "--post-eval-verbose",
        action="store_true",
        help="Per-step λ prints in baseline_eval_*.txt (omit --quiet); files can be very large",
    )

    args = parser.parse_args()
    
    # Override timesteps for test mode
    if args.test:
        args.timesteps = 1000
        print("🧪 Running in TEST mode (1000 timesteps)")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    hp_resolved = _resolve_hyperparams_path(project_root, args.hyperparams)
    write_run_artifacts(log_dir, project_root, args, hp_resolved)
    
    print("=" * 80)
    print("PPO TRAINING FOR ADAPTIVE SPATIAL CROWDSOURCING")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Resume from: {args.resume if args.resume else 'None (new training)'}")
    print("=" * 80)
    
    # Setup Data Paths
    print("\n[1/5] Setting up data paths...")
    data_root = args.data_root
    eval_day_folder = None

    # Get all day folders from the data root
    if os.path.exists(data_root):
        all_days = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))]
        all_days = sorted(all_days)  # Ensure consistent order before splitting
        
        print(f"   Found {len(all_days)} day folders in {data_root}")
        
        if len(all_days) < args.train_days:
            print(f"   ⚠️  Warning: Only {len(all_days)} folders available, but {args.train_days} requested for training")
            print(f"   Using all {len(all_days)} folders for training")
            train_days = all_days
            test_days = []
        else:
            # Split: Training Days, Testing Days
            # shuffle=True ensures we get a random mix of weekdays/weekends in both sets
            train_days, test_days = train_test_split(
                all_days, 
                train_size=args.train_days, 
                random_state=42, 
                shuffle=True
            )
        
        print(f"   🏋️  Training on {len(train_days)} days: {train_days[:3]}...")
        if test_days:
            print(f"   🧪 Testing on {len(test_days)} days: {test_days[:3]}...")
            eval_day_folder, eval_rule = resolve_baseline_eval_day(test_days, args.eval_day)
            print(f"   📋 Post-eval compare day: {eval_day_folder} ({eval_rule})")
            update_run_manifest_baseline_eval(log_dir, eval_day_folder, test_days, eval_rule)
    else:
        print(f"   ⚠️  Data root not found: {data_root}")
        print("   Falling back to legacy single-dataset mode")
        train_days = None
        test_days = None
        data_root = None

    # Initialize environment(s)
    print("\n[2/5] Initializing environment(s)...")
    
    if args.no_parallel or train_days is None:
        # Single environment mode (legacy or disabled parallel)
        print("   Using single environment (no parallelization)")
        env = create_env(
            dataset="didi", 
            step_duration_minutes=5,  # 5-minute steps for high-frequency decisions
            reward_weights=[1.0, 1.0, 1.0],
            data_root=data_root,
            day_folders=train_days,
            warmup_duration_minutes=30,
            episode_duration_hours=8
        )
        env = Monitor(env, log_dir)
    else:
        # Parallel environments mode
        num_cpu = args.num_cpu
        print(f"   Creating {num_cpu} parallel environments...")
        env = SubprocVecEnv([
            make_env(data_root, train_days, i, step_duration_minutes=5, 
                    reward_weights=[1.0, 1.0, 1.0],
                    warmup_duration_minutes=30,
                    episode_duration_hours=8) 
            for i in range(num_cpu)
        ])
        print(f"   ✅ Parallel environment created with {num_cpu} workers")

    write_environment_spec(log_dir, env)

    # Check environment compatibility (optional)
    if not args.skip_env_check:
        print("\n[3/5] Checking environment compatibility...")
        try:
            # For parallel envs, check the first one
            if isinstance(env, SubprocVecEnv):
                # Can't easily check SubprocVecEnv, skip for now
                print("   Skipping check for parallel environments (SubprocVecEnv)")
            else:
                check_env(env, warn=True)
                print("✅ Environment check passed!")
        except Exception as e:
            print(f"⚠️  Environment check warning: {e}")
            print("   Continuing anyway...")
    else:
        print("\n[3/5] Skipping environment check (--skip-env-check)")
    
    # Create evaluation environment
    print("\n[4/5] Creating evaluation environment...")
    if not test_days or len(test_days) == 0:
        raise ValueError(
            "No test days available for evaluation. "
            "Cannot evaluate on training data (data leakage). "
            "Ensure train_days < total_days to create a test split."
        )
    
    eval_env = create_env(
        dataset="didi", 
        step_duration_minutes=5,  # 5-minute steps for consistency
        reward_weights=[1.0, 1.0, 1.0],
        data_root=data_root,
        day_folders=test_days,
        warmup_duration_minutes=30,
        episode_duration_hours=8
    )
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Initialize or load PPO agent
    print("\n[5/5] Initializing PPO agent...")
    if args.resume:
        print(f"   Loading model from: {args.resume}")
        try:
            model = PPO.load(args.resume, env=env, verbose=1)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("   Starting fresh training instead...")
            model = create_model(env, log_dir, hyperparams_path=args.hyperparams)
    else:
        model = create_model(env, log_dir, hyperparams_path=args.hyperparams)
    
    # Setup callbacks
    print("\n[4/4] Setting up training callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, args.timesteps // 10),  # Save at least 10 times during training
        save_path=log_dir,
        name_prefix="ppo_sc_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(1000, args.timesteps // 20),  # Evaluate at least 20 times
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Train agent
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print(f"Monitor progress with: tensorboard --logdir {log_dir}")
    print("=" * 80 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.resume is None  # Reset counter if not resuming
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        
        # Save final model
        final_model_path = os.path.join(log_dir, "ppo_sc_final")
        model.save(final_model_path)
        print(f"📦 Final model saved to: {final_model_path}")
        print(f"📊 TensorBoard logs: {log_dir}")
        print(f"🏆 Best model: {os.path.join(log_dir, 'best_model', 'best_model.zip')}")

        if not args.no_post_eval and eval_day_folder is not None:
            run_post_training_baseline_eval(
                log_dir,
                project_root,
                eval_day_folder,
                args.data_root,
                args.eval_seed,
                quiet=not args.post_eval_verbose,
            )

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving current model...")
        interrupted_path = os.path.join(log_dir, "ppo_sc_interrupted")
        model.save(interrupted_path)
        print(f"📦 Model saved to: {interrupted_path}")
        print("   Resume with: --resume", interrupted_path)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Attempting to save model...")
        try:
            error_path = os.path.join(log_dir, "ppo_sc_error")
            model.save(error_path)
            print(f"📦 Model saved to: {error_path}")
        except:
            print("   Failed to save model")

def _get_net_arch(net_arch_type):
    """Map Optuna net_arch_type to policy_kwargs. Default to large [256, 256]."""
    if net_arch_type == "small":
        return dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        return dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_type == "large":
        return dict(pi=[256, 256], vf=[256, 256])
    return dict(pi=[256, 256], vf=[256, 256])  # Default: Optuna-recommended large


def create_model(env, log_dir, hyperparams_path=None):
    """
    Create a new PPO model. Loads hyperparameters from best_hyperparameters.json if available.
    Defaults to large [256, 256] network when no file is specified.
    """
    project_root = Path(__file__).resolve().parent.parent
    default_hyperparams_path = project_root / "rl" / "best_hyperparameters.json"
    path = Path(hyperparams_path) if hyperparams_path else default_hyperparams_path

    kwargs = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    net_arch_type = "large"

    if path.exists():
        try:
            with open(path) as f:
                hp = json.load(f)
            net_arch_type = hp.pop("net_arch_type", "large")
            for k, v in hp.items():
                if k in kwargs:
                    kwargs[k] = v
            print(f"   Loaded hyperparameters from {path} (net_arch: {net_arch_type})")
        except Exception as e:
            print(f"   ⚠️  Could not load {path}: {e}. Using defaults.")
    else:
        print(f"   No hyperparams file at {path}. Using defaults (net_arch: large [256, 256]).")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=_get_net_arch(net_arch_type)),
        **kwargs
    )
    return model

if __name__ == "__main__":
    main()
