#!/usr/bin/env python3
"""
Backfill a legacy rl_logs_sb3/run_* folder with the same artifacts train_sb3.py writes at startup.

Uses **current files on disk** (not git history):
  - rl/gym_environment.py  -> gym_environment_snapshot.py
  - best_hyperparameters.json (repo root) or rl/best_hyperparameters.json -> hyperparams_snapshot.json

Use this after aligning your working tree with the code that was used for an older run (e.g. stash or
checkout). Then the snapshot matches that run's reward/observation logic.

Example:
  python3 scripts/backfill_run_artifacts.py --run-dir rl_logs_sb3/run_20260402_135637

After snapshots, by default runs the same baseline comparison as `train_sb3.py` post-training
(`compare_model_to_baseline.py` for final + best checkpoints), using `--eval-day` default
`DEFAULT_BASELINE_EVAL_DAY` from `rl/train_sb3.py` so future runs stay comparable. Use `--no-baseline`
to skip (e.g. if you only want file copies).
"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_observation_static_scaling, get_strategy_params  # noqa: E402


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _default_box_spaces() -> Dict[str, Dict]:
    """Matches rl/gym_environment.py Box definitions (17-dim obs, 2-dim action)."""
    return {
        "observation_space": {
            "type": "Box",
            "shape": [17],
            "dtype": "float32",
            "low": [-math.inf] * 17,
            "high": [math.inf] * 17,
        },
        "action_space": {
            "type": "Box",
            "shape": [2],
            "dtype": "float32",
            "low": [0.0, 0.0],
            "high": [2.0, 0.5],
        },
    }


def _merge_manifest_artifacts(log_dir: Path, updates: dict) -> None:
    manifest_path = log_dir / "run_manifest.json"
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


def _resolve_hyperparams_path(project_root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else project_root / explicit
    for rel in ("best_hyperparameters.json", "rl/best_hyperparameters.json"):
        p = project_root / rel
        if p.is_file():
            return p
    return None


def _write_environment_spec_standalone(log_dir: Path) -> None:
    """
    Same structure as train_sb3.write_environment_spec for default training hyperparams,
    without constructing a full simulator (matches unwrapped env defaults).
    """
    comp = get_strategy_params("composite")
    boxes = _default_box_spaces()
    spec = {
        "observation_space": boxes["observation_space"],
        "action_space": boxes["action_space"],
        "reward_note": (
            "reward_weights below are (fairness, starvation, latency) multipliers; "
            "see _calculate_reward() in gym_environment_snapshot.py for the full formula."
        ),
        "env_runtime": {
            "reward_weights": [1.0, 1.0, 1.0],
            "lambda3_fixed": float(comp.get("utility_weight", 1.0)),
            "step_duration_minutes": 5.0,
            "warmup_duration_seconds": 30 * 60,
            "episode_duration_seconds": 8 * 60 * 60,
            "sla_wait_time_minutes": 2.8,
            "sla_violation_penalty": 20.0,
            "obs_scaling": _json_safe(get_observation_static_scaling()),
        },
    }
    out = log_dir / "environment_spec.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    _merge_manifest_artifacts(log_dir, {"environment_spec": "environment_spec.json"})
    print(f"   Wrote {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill run folder with gym/hyperparam snapshots")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory, e.g. rl_logs_sb3/run_20260402_135637",
    )
    parser.add_argument(
        "--gym-source",
        type=str,
        default="rl/gym_environment.py",
        help="Path to gym environment source (relative to repo root). Default: current rl/gym_environment.py",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=None,
        help="Hyperparameters JSON (relative to repo root). Default: best_hyperparameters.json or rl/best_hyperparameters.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing gym_environment_snapshot.py / hyperparams_snapshot.json",
    )
    parser.add_argument(
        "--skip-legacy-weight-alias",
        action="store_true",
        help="Do not copy eval_weights_*_steps.txt to baseline_*_model_weight_outputs.txt (ignored if baseline runs)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip compare_model_to_baseline (only write snapshots / manifest / environment_spec)",
    )
    parser.add_argument(
        "--eval-day",
        type=str,
        default=None,
        help="Day folder for baseline compare (default: same DEFAULT_BASELINE_EVAL_DAY as train_sb3 post-eval)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="RNG seed for baseline compare (default: 42, same as train_sb3)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/didi/full_didi_gaia",
        help="Data root passed to compare_model_to_baseline (default: data/didi/full_didi_gaia)",
    )
    parser.add_argument(
        "--post-eval-verbose",
        action="store_true",
        help="Omit --quiet on baseline runs (very large baseline_eval_*.txt)",
    )
    args = parser.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"❌ Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    gym_src = (PROJECT_ROOT / args.gym_source).resolve()
    if not gym_src.is_file():
        print(f"❌ Missing gym source: {gym_src}", file=sys.stderr)
        sys.exit(1)

    hp_path = _resolve_hyperparams_path(PROJECT_ROOT, args.hyperparams)
    gym_dst = run_dir / "gym_environment_snapshot.py"
    hp_dst = run_dir / "hyperparams_snapshot.json"

    if gym_dst.exists() and not args.force:
        print(f"⚠️  Exists (use --force): {gym_dst}")
    else:
        shutil.copy2(gym_src, gym_dst)
        print(f"   Copied {gym_src.relative_to(PROJECT_ROOT)} -> {gym_dst.name}")

    if hp_path is None:
        print("⚠️  No hyperparameters file found; skipping hyperparams_snapshot.json")
    elif hp_dst.exists() and not args.force:
        print(f"⚠️  Exists (use --force): {hp_dst}")
    else:
        shutil.copy2(hp_path, hp_dst)
        print(f"   Copied {hp_path.relative_to(PROJECT_ROOT)} -> {hp_dst.name}")

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_folder": run_dir.name,
        "created_utc": created,
        "backfill": {
            "tool": "scripts/backfill_run_artifacts.py",
            "created_utc": created,
            "gym_source": str(gym_src),
            "hyperparams_source": str(hp_path) if hp_path else None,
            "note": (
                "Artifacts generated from current working-tree files. "
                "gym_environment_snapshot.py matches rl/gym_environment.py at backfill time."
            ),
        },
        "argv": sys.argv,
        "artifacts": {
            "gym_environment_snapshot": "gym_environment_snapshot.py",
            "hyperparams_snapshot": "hyperparams_snapshot.json" if hp_dst.is_file() else None,
            "gym_sha256": _sha256_file(gym_dst),
            "hyperparams_sha256": _sha256_file(hp_dst) if hp_dst.is_file() else None,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Wrote {manifest_path.name}")

    _write_environment_spec_standalone(run_dir)

    baseline_ran = False
    eval_day_used = ""
    if not args.no_baseline:
        try:
            from rl.train_sb3 import (
                DEFAULT_BASELINE_EVAL_DAY,
                run_post_training_baseline_eval,
                update_run_manifest_baseline_eval,
            )
        except ImportError as e:
            print(f"❌ Baseline step needs training deps (numpy, stable-baselines3, …): {e}", file=sys.stderr)
            print("   Use --no-baseline to only copy snapshots.", file=sys.stderr)
            sys.exit(1)

        eval_day_used = (args.eval_day or "").strip() or DEFAULT_BASELINE_EVAL_DAY
        update_run_manifest_baseline_eval(
            str(run_dir),
            eval_day_used,
            [],
            f"backfill: same default eval day as train_sb3 post-eval ({eval_day_used})",
        )
        print(f"\n   Baseline compare: day={eval_day_used} seed={args.eval_seed} data-root={args.data_root}")
        run_post_training_baseline_eval(
            str(run_dir),
            PROJECT_ROOT,
            eval_day_used,
            args.data_root,
            args.eval_seed,
            quiet=not args.post_eval_verbose,
        )
        baseline_ran = True
    elif not args.skip_legacy_weight_alias:
        pairs = [
            ("eval_weights_final_steps.txt", "baseline_final_model_weight_outputs.txt"),
            ("eval_weights_best_steps.txt", "baseline_best_model_weight_outputs.txt"),
        ]
        for legacy, new_name in pairs:
            src = run_dir / legacy
            dst = run_dir / new_name
            if src.is_file():
                if dst.exists() and not args.force:
                    print(f"⚠️  Skip alias (exists): {new_name}")
                else:
                    shutil.copy2(src, dst)
                    print(f"   Copied {legacy} -> {new_name}")

    record = run_dir / "RUN_BACKFILL_RECORD.md"
    hp_rel = str(hp_path.relative_to(PROJECT_ROOT)) if hp_path else "none"
    gym_rel = str(gym_src.relative_to(PROJECT_ROOT))
    baseline_section = ""
    if baseline_ran:
        baseline_section = (
            f"- **Baseline compare:** ran `compare_model_to_baseline.py` (final + best if present).\n"
            f"- **Eval day:** `{eval_day_used}` (aligned with `train_sb3` default post-eval).\n"
            f"- **Eval seed:** {args.eval_seed}\n"
        )
    else:
        baseline_section = (
            "- **Baseline compare:** skipped (`--no-baseline`). "
            "Legacy `eval_weights_*_steps.txt` copies used if present.\n"
        )
    record.write_text(
        f"# Backfill record: `{run_dir.name}`\n\n"
        f"- **Backfilled at (UTC):** {created}\n"
        f"- **Gym snapshot source:** `{gym_rel}` (current file on disk at backfill time).\n"
        f"- **Hyperparams source:** `{hp_rel}`\n"
        f"{baseline_section}"
        "- **Note:** For a specific git revision of `rl/gym_environment.py`, check out that revision first, "
        "then run backfill, or edit `gym_environment_snapshot.py` manually.\n",
        encoding="utf-8",
    )
    print(f"   Wrote {record.name}")
    print("Done.")


if __name__ == "__main__":
    main()
