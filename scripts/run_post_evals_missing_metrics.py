#!/usr/bin/env python3
"""
Batch post-training baseline comparison for run folders that have checkpoints
but no baseline_eval / metrics yet.

Uses the same Python interpreter you launch this script with (``sys.executable``),
so after ``conda activate sc`` run:

    cd /path/to/sc_sim
    python scripts/run_post_evals_missing_metrics.py

Quiet compare logs still go to each run's ``baseline_eval_*.txt``; progress prints
on stdout so you see that something is running.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPARE = PROJECT_ROOT / "scripts" / "compare_model_to_baseline.py"

DEFAULT_RUNS = (
    "run_20260313_141658",
    "run_20260321_033353",
    "run_20260329_145347",
    "run_20260330_161842",
    "run_20260331_103805",
)


def _require_imports() -> None:
    try:
        import numpy  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError as e:
        print(
            "Missing dependency for compare_model_to_baseline.\n"
            "Activate your env first, e.g.:\n"
            "  conda activate sc\n"
            f"Then: {sys.executable} scripts/run_post_evals_missing_metrics.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _run_compare(
    *,
    model: str,
    day: str,
    data_root: str,
    eval_seed: int,
    log_weights: Path,
    metrics_out: Path,
    log_file: Path,
    quiet: bool,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(COMPARE),
        "--model",
        str(model),
        "--day",
        day,
        "--data-root",
        data_root,
        "--eval-seed",
        str(eval_seed),
        "--log-weights",
        str(log_weights),
        "--metrics-out",
        str(metrics_out),
    ]
    if quiet:
        cmd.append("--quiet")

    print(f"  → {' '.join(cmd[:4])} ... (full log: {log_file})")
    with open(log_file, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=fp,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        print(f"  ✗ failed (exit {proc.returncode}). Tail of {log_file}:", file=sys.stderr)
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            lines = text.strip().splitlines()
            for line in lines[-25:]:
                print(line, file=sys.stderr)
        except OSError:
            pass
        raise SystemExit(proc.returncode)


def main() -> None:
    _require_imports()

    if not COMPARE.is_file():
        print(f"Not found: {COMPARE}", file=sys.stderr)
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--day",
        default="496528674@qq.com_20161128",
        help="Held-out day folder (default: same as compare script)",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data" / "didi" / "full_didi_gaia"),
        help="DiDi data root",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Env seed (default 42)",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="RUN_DIR",
        help="Extra run folder name under rl_logs_sb3 (repeatable). Default: built-in list.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to compare (less console noise inside log files)",
    )
    args = parser.parse_args()

    runs = list(args.runs) if args.runs else list(DEFAULT_RUNS)
    data_root = args.data_root
    if not Path(data_root).is_absolute():
        data_root = str((PROJECT_ROOT / data_root.lstrip("./")).resolve())

    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {sys.executable}")
    print(f"Data:    {data_root}")
    print()

    for run in runs:
        base = PROJECT_ROOT / "rl_logs_sb3" / run
        if not base.is_dir():
            print(f"Skip (missing dir): {base}")
            continue

        print("=" * 50)
        print(f"Post-eval: {run}")
        print("=" * 50)

        final_zip = base / "ppo_sc_final.zip"
        if not final_zip.is_file():
            print(f"  Skip: no {final_zip.name}")
        else:
            _run_compare(
                model=f"rl_logs_sb3/{run}/ppo_sc_final",
                day=args.day,
                data_root=data_root,
                eval_seed=args.eval_seed,
                log_weights=base / "baseline_final_model_weight_outputs.txt",
                metrics_out=base / "baseline_final_model_metrics.txt",
                log_file=base / "baseline_eval_final.txt",
                quiet=args.quiet,
            )
            print("  final: OK")

        best_zip = base / "best_model" / "best_model.zip"
        if best_zip.is_file():
            _run_compare(
                model=f"rl_logs_sb3/{run}/best_model/best_model",
                day=args.day,
                data_root=data_root,
                eval_seed=args.eval_seed,
                log_weights=base / "baseline_best_model_weight_outputs.txt",
                metrics_out=base / "baseline_best_model_metrics.txt",
                log_file=base / "baseline_eval_best.txt",
                quiet=args.quiet,
            )
            print("  best: OK")
        else:
            print("  (no best_model.zip)")

        print()

    print("All requested post-evals finished.")


if __name__ == "__main__":
    main()
