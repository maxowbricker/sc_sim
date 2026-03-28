#!/usr/bin/env python
"""
Run compare_model_to_baseline.py for each model: stratified (config DATA_SAMPLING) then full dataset.

Fair comparison (each subprocess):
  - Static baseline uses the same gym env as RL: greedy warmup → composite episode with fixed config λ.
  - RL uses the same reset(eval_seed) and horizon (see rl/gym_environment.py).
  - Passed through: --eval-seed (default 42).

For each .zip under --models-dir (default: rl_logs_sb3/recent_run_21/rl_final_results):
  1) --stratified true  → uses config.py DATA_SAMPLING (e.g. 40k tasks / 10k workers / 288 bins)
  2) --stratified false → full raw day load, same composite params from config.py

Collates results into benchmark_batch_report_<timestamp>.txt and .csv (RL vs composite baseline, deltas).

Usage:
    python scripts/run_benchmark_batch.py
    python scripts/run_benchmark_batch.py --models-dir rl_logs_sb3/recent_run_21/rl_final_results
    python scripts/run_benchmark_batch.py --models path/to/a.zip path/to/b.zip
"""
import argparse
import csv
import subprocess
import sys
import os
import re
import glob
from datetime import datetime

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMPARE_SCRIPT = os.path.join(SCRIPT_DIR, "compare_model_to_baseline.py")

DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "rl_logs_sb3", "recent_run_21", "rl_final_results")

DAY = "496528674@qq.com_20161128"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "didi", "full_didi_gaia")


def discover_models(models_dir: str):
    """Return sorted list of paths to *.zip under models_dir."""
    if not os.path.isdir(models_dir):
        return []
    pattern = os.path.join(models_dir, "*.zip")
    return sorted(glob.glob(pattern))


def parse_comparison_output(stdout: str):
    """Extract JFI, Backlog, Wait from the comparison table."""
    # Match lines like: JFI (Fairness)       | 0.8666          | 0.6825          | ...
    jfi_match = re.search(
        r"JFI \(Fairness\)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|",
        stdout
    )
    backlog_match = re.search(
        r"Peak Backlog\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|",
        stdout
    )
    wait_match = re.search(
        r"Avg Wait Time \(m\)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|",
        stdout
    )
    if not all([jfi_match, backlog_match, wait_match]):
        return None
    static_jfi = float(jfi_match.group(1))
    rl_jfi = float(jfi_match.group(2))
    static_backlog = float(backlog_match.group(1))
    rl_backlog = float(backlog_match.group(2))
    static_wait = float(wait_match.group(1))
    rl_wait = float(wait_match.group(2))
    return {
        "static_jfi": static_jfi,
        "rl_jfi": rl_jfi,
        "static_backlog": static_backlog,
        "rl_backlog": rl_backlog,
        "static_wait": static_wait,
        "rl_wait": rl_wait,
        "delta_jfi": rl_jfi - static_jfi,
        "delta_backlog": rl_backlog - static_backlog,
        "delta_wait": rl_wait - static_wait,
    }


def run_one(model, stratified, day=DAY, data_root=DATA_ROOT, eval_seed=42):
    """Run compare script once. Returns (parsed_results, raw_stdout)."""
    model_path = model if os.path.isabs(model) else os.path.join(PROJECT_ROOT, model)
    cmd = [
        sys.executable,
        COMPARE_SCRIPT,
        "--model", model_path,
        "--day", day,
        "--data-root", data_root,
        "--stratified", "true" if stratified else "false",
        "--eval-seed", str(eval_seed),
        "--quiet",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 h per run (full-day sim can be slow)
    )
    stdout = result.stdout + result.stderr
    parsed = parse_comparison_output(stdout)
    return parsed, stdout


def main():
    parser = argparse.ArgumentParser(
        description="Run compare_model_to_baseline for each model: stratified (DATA_SAMPLING) then full dataset"
    )
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model .zip paths (overrides --models-dir)")
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing .zip models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--day", default=DAY, help="Test day folder name under data root")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Data root path")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Passed to compare_model_to_baseline (same static vs RL scenario)",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(PROJECT_ROOT, args.data_root.lstrip("./"))
    if not os.path.isabs(args.models_dir):
        args.models_dir = os.path.join(PROJECT_ROOT, args.models_dir.lstrip("./"))

    if args.models:
        models = args.models
    else:
        models = discover_models(args.models_dir)
        if not models:
            print(f"❌ No *.zip files found in {args.models_dir}")
            print("   Pass --models-dir or --models path1.zip path2.zip ...")
            return 1

    print("=" * 70)
    print(f"BENCHMARK BATCH: {len(models)} models × 2 settings = {len(models) * 2} runs")
    print("=" * 70)
    print(f"Day: {args.day}")
    print(f"eval_seed: {args.eval_seed} (fair static vs RL alignment)")
    print(f"Models: {models}")
    print("Order: (1) stratified=True  → config DATA_SAMPLING  (2) stratified=False → full load")
    print("=" * 70)

    results = []
    for i, model in enumerate(models):
        for stratified in [True, False]:
            label = "stratified" if stratified else "full-scale"
            model_short = os.path.basename(model.replace(".zip", ""))
            run_id = f"{model_short} | {label}"
            total_runs = len(models) * 2
            print(f"\n[{len(results)+1}/{total_runs}] Running: {run_id}")
            print("-" * 50)
            try:
                parsed, raw = run_one(model, stratified, args.day, args.data_root, eval_seed=args.eval_seed)
                if parsed:
                    results.append({
                        "model": model_short,
                        "stratified": stratified,
                        **parsed,
                    })
                    print(f"   Composite baseline | RL agent")
                    print(f"   JFI:     {parsed['static_jfi']:.4f}          | {parsed['rl_jfi']:.4f}   (Δ {parsed['delta_jfi']:+.4f})")
                    print(f"   Backlog: {parsed['static_backlog']:.0f}            | {parsed['rl_backlog']:.0f}   (Δ {parsed['delta_backlog']:+.0f})")
                    print(f"   Wait:    {parsed['static_wait']:.2f} m          | {parsed['rl_wait']:.2f} m   (Δ {parsed['delta_wait']:+.2f})")
                else:
                    print("   ⚠️  Could not parse output")
                    print(raw[-1500:] if len(raw) > 1500 else raw)
            except subprocess.TimeoutExpired:
                print("   ❌ Timeout (1 h)")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    # Write summary report + CSV (collated RL vs composite baseline)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(PROJECT_ROOT, f"benchmark_batch_report_{timestamp}.txt")
    csv_path = os.path.join(PROJECT_ROOT, f"benchmark_batch_report_{timestamp}.csv")

    def write_collated_tables(f):
        """RL vs composite baseline: one block per setting, full metrics + deltas."""
        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write("COLLATED: RL AGENT vs COMPOSITE BASELINE (same baseline per row for that day + setting)\n")
        f.write("=" * 90 + "\n\n")

        for stratified, title in [
            (True, "SETTING A — STRATIFIED (config DATA_SAMPLING: shrink to target_tasks / target_workers / bins)"),
            (False, "SETTING B — FULL DATASET (no stratified shrink; composite params from config.py)"),
        ]:
            rows = [r for r in results if r["stratified"] == stratified]
            if not rows:
                continue
            f.write(f"{title}\n")
            f.write("-" * 90 + "\n")
            hdr = (
                f"{'Model':<42} | {'Comp JFI':>9} | {'RL JFI':>9} | {'Δ JFI':>8} | "
                f"{'Comp BL':>9} | {'RL BL':>9} | {'Δ BL':>8} | {'Comp Wait':>9} | {'RL Wait':>9} | {'Δ Wait':>8}\n"
            )
            f.write(hdr)
            f.write("-" * 90 + "\n")
            for r in rows:
                f.write(
                    f"{r['model']:<42} | {r['static_jfi']:9.4f} | {r['rl_jfi']:9.4f} | {r['delta_jfi']:+8.4f} | "
                    f"{r['static_backlog']:9.0f} | {r['rl_backlog']:9.0f} | {r['delta_backlog']:+8.0f} | "
                    f"{r['static_wait']:9.2f} | {r['rl_wait']:9.2f} | {r['delta_wait']:+8.2f}\n"
                )
            f.write("\n")
            f.write("  Comp = composite static baseline | RL = PPO agent | BL = peak backlog | Wait = avg minutes\n\n")

        f.write("LEGEND — Good direction: Δ JFI positive (fairer), Δ BL negative (lower backlog), Δ Wait negative (shorter waits).\n")

    with open(report_path, "w") as f:
        f.write("BENCHMARK BATCH REPORT\n")
        f.write("=" * 90 + "\n")
        f.write(f"Day: {args.day}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"eval_seed (compare_model_to_baseline): {args.eval_seed}\n")
        f.write(
            "Protocol: Static baseline = same env as RL (greedy warmup + fixed config λ for episode); "
            "RL = same reset(seed) + policy. Not full-calendar-day static vs partial RL.\n"
        )
        f.write(f"Runs completed: {len(results)} / {len(models) * 2}\n")
        f.write("=" * 90 + "\n")

        write_collated_tables(f)

        f.write("\n")
        f.write("QUICK SUMMARY (RL deltas only)\n")
        f.write("-" * 70 + "\n")
        for r in results:
            setting = "stratified" if r["stratified"] else "full"
            f.write(
                f"{r['model']:<40} [{setting}]  ΔJFI {r['delta_jfi']:+.4f}  "
                f"ΔBacklog {r['delta_backlog']:+.0f}  ΔWait {r['delta_wait']:+.2f}m\n"
            )

        f.write("\n")
        f.write("DETAILED TABLE (all columns)\n")
        f.write("-" * 90 + "\n")
        f.write(
            f"{'Model':<35} | {'Setting':<10} | {'Comp JFI':<9} | {'RL JFI':<9} | {'Δ JFI':<8} | "
            f"{'Comp BL':<9} | {'RL BL':<9} | {'Comp Wait':<9} | {'RL Wait':<9}\n"
        )
        f.write("-" * 90 + "\n")
        for r in results:
            setting = "stratified" if r["stratified"] else "full"
            f.write(
                f"{r['model']:<35} | {setting:<10} | {r['static_jfi']:<9.4f} | {r['rl_jfi']:<9.4f} | {r['delta_jfi']:+8.4f} | "
                f"{r['static_backlog']:<9.0f} | {r['rl_backlog']:<9.0f} | "
                f"{r['static_wait']:<9.2f} | {r['rl_wait']:<9.2f}\n"
            )

    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([
            "model", "setting", "composite_jfi", "rl_jfi", "delta_jfi",
            "composite_backlog", "rl_backlog", "delta_backlog",
            "composite_wait_min", "rl_wait_min", "delta_wait_min",
        ])
        for r in results:
            w.writerow([
                r["model"],
                "stratified" if r["stratified"] else "full",
                r["static_jfi"], r["rl_jfi"], r["delta_jfi"],
                r["static_backlog"], r["rl_backlog"], r["delta_backlog"],
                r["static_wait"], r["rl_wait"], r["delta_wait"],
            ])

    if not results:
        print("\n⚠️  No successful runs to collate (parse failures or errors).")
        print(f"   Partial report written: {report_path}")
        return 0

    print("\n")
    print("=" * 90)
    print("COLLATED RESULTS: RL vs COMPOSITE BASELINE (also in report + CSV)")
    print("=" * 90)
    for stratified, title in [
        (True, "SETTING A — STRATIFIED"),
        (False, "SETTING B — FULL DATASET"),
    ]:
        rows = [r for r in results if r["stratified"] == stratified]
        if not rows:
            continue
        print(f"\n{title}")
        print("-" * 90)
        hdr = (
            f"{'Model':<42} | {'Comp JFI':>9} | {'RL JFI':>9} | {'Δ JFI':>8} | "
            f"{'Comp BL':>9} | {'RL BL':>9} | {'Δ BL':>8} | {'Comp Wait':>9} | {'RL Wait':>9} | {'Δ Wait':>8}"
        )
        print(hdr)
        print("-" * 90)
        for r in rows:
            print(
                f"{r['model']:<42} | {r['static_jfi']:9.4f} | {r['rl_jfi']:9.4f} | {r['delta_jfi']:+8.4f} | "
                f"{r['static_backlog']:9.0f} | {r['rl_backlog']:9.0f} | {r['delta_backlog']:+8.0f} | "
                f"{r['static_wait']:9.2f} | {r['rl_wait']:9.2f} | {r['delta_wait']:+8.2f}"
            )
    print()

    print("=" * 70)
    print(f"✅ Full report:  {report_path}")
    print(f"✅ CSV (Excel): {csv_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
