#!/usr/bin/env python
"""
Run compare_model_to_baseline.py 6 times: 3 models × 2 stratified settings.
Collates results into a summary report.

Usage:
    python scripts/run_benchmark_batch.py
    python scripts/run_benchmark_batch.py --models model1.zip model2.zip
"""
import argparse
import subprocess
import sys
import os
import re
from datetime import datetime

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMPARE_SCRIPT = os.path.join(SCRIPT_DIR, "compare_model_to_baseline.py")

# Default models
DEFAULT_MODELS = [
    "rl_logs_final/run_20260318_195154/ppo_sc_interrupted.zip",
    "rl_logs_final/run_20260318_195154/best_model/best_model.zip",
    "rl_logs_final/run_20260318_195154/ppo_sc_model_80000_steps.zip",
]

DAY = "496528674@qq.com_20161128"
DATA_ROOT = "./data/didi/full_didi_gaia"


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
    return {
        "static_jfi": float(jfi_match.group(1)),
        "rl_jfi": float(jfi_match.group(2)),
        "static_backlog": float(backlog_match.group(1)),
        "rl_backlog": float(backlog_match.group(2)),
        "static_wait": float(wait_match.group(1)),
        "rl_wait": float(wait_match.group(2)),
    }


def run_one(model, stratified, day=DAY, data_root=DATA_ROOT):
    """Run compare script once. Returns (parsed_results, raw_stdout)."""
    cmd = [
        sys.executable,
        COMPARE_SCRIPT,
        "--model", model,
        "--day", day,
        "--data-root", data_root,
        "--stratified", "true" if stratified else "false",
        "--quiet",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min per run
    )
    stdout = result.stdout + result.stderr
    parsed = parse_comparison_output(stdout)
    return parsed, stdout


def main():
    parser = argparse.ArgumentParser(description="Run benchmark comparison 6 times (3 models × 2 stratified)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model paths")
    parser.add_argument("--day", default=DAY, help="Test day folder")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Data root path")
    args = parser.parse_args()
    models = args.models

    print("=" * 70)
    print(f"BENCHMARK BATCH: {len(models)} models × 2 stratified settings = {len(models)*2} runs")
    print("=" * 70)
    print(f"Day: {args.day}")
    print(f"Models: {models}")
    print("=" * 70)

    results = []
    for i, model in enumerate(models):
        for stratified in [False, True]:
            label = "stratified" if stratified else "full-scale"
            model_short = os.path.basename(model.replace(".zip", ""))
            run_id = f"{model_short} | {label}"
            total_runs = len(models) * 2
            print(f"\n[{len(results)+1}/{total_runs}] Running: {run_id}")
            print("-" * 50)
            try:
                parsed, raw = run_one(model, stratified, args.day, args.data_root)
                if parsed:
                    results.append({
                        "model": model_short,
                        "stratified": stratified,
                        **parsed,
                    })
                    print(f"   JFI: static={parsed['static_jfi']:.4f} rl={parsed['rl_jfi']:.4f}")
                    print(f"   Backlog: static={parsed['static_backlog']:.0f} rl={parsed['rl_backlog']:.0f}")
                    print(f"   Wait: static={parsed['static_wait']:.2f} rl={parsed['rl_wait']:.2f}")
                else:
                    print("   ⚠️  Could not parse output")
                    print(raw[-1500:] if len(raw) > 1500 else raw)
            except subprocess.TimeoutExpired:
                print("   ❌ Timeout (10 min)")
            except Exception as e:
                print(f"   ❌ Error: {e}")

    # Write summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(PROJECT_ROOT, f"benchmark_batch_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("BENCHMARK BATCH REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Day: {args.day}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("FULL-SCALE (stratified=False)\n")
        f.write("-" * 50 + "\n")
        for r in results:
            if not r["stratified"]:
                f.write(f"{r['model']:40} | JFI: {r['rl_jfi']:.4f} (Δ{r['rl_jfi']-r['static_jfi']:+.4f}) | "
                        f"Backlog: {r['rl_backlog']:.0f} | Wait: {r['rl_wait']:.2f}m\n")
        f.write("\n")

        f.write("STRATIFIED (stratified=True, 5k tasks / 1.25k workers)\n")
        f.write("-" * 50 + "\n")
        for r in results:
            if r["stratified"]:
                f.write(f"{r['model']:40} | JFI: {r['rl_jfi']:.4f} (Δ{r['rl_jfi']-r['static_jfi']:+.4f}) | "
                        f"Backlog: {r['rl_backlog']:.0f} | Wait: {r['rl_wait']:.2f}m\n")
        f.write("\n")

        f.write("DETAILED TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<35} | {'Setting':<10} | {'Static JFI':<10} | {'RL JFI':<10} | {'Static Backlog':<14} | {'RL Backlog':<10} | {'Static Wait':<10} | {'RL Wait':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            setting = "stratified" if r["stratified"] else "full"
            f.write(f"{r['model']:<35} | {setting:<10} | {r['static_jfi']:<10.4f} | {r['rl_jfi']:<10.4f} | "
                    f"{r['static_backlog']:<14.0f} | {r['rl_backlog']:<10.0f} | "
                    f"{r['static_wait']:<10.2f} | {r['rl_wait']:<10.2f}\n")

    print("\n" + "=" * 70)
    print(f"✅ Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
