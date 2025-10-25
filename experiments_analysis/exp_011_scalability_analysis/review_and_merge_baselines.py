#!/usr/bin/env python3
"""
Baseline Results Review and Merge Tool
=======================================

Use this script in the morning to:
1. Review the baseline results
2. Compare with original experiment results
3. Optionally merge them into the main CSV

This is INTERACTIVE and SAFE - you review before any changes are made.
"""

import pandas as pd
from pathlib import Path

def main():
    exp_dir = Path(__file__).parent
    
    # File paths
    original_csv = exp_dir / "data" / "experiment_011_aggregate_results.csv"
    baseline_csv = exp_dir / "baselines_exp011_results.csv"
    merged_csv = exp_dir / "data" / "experiment_011_aggregate_results_WITH_BASELINES.csv"
    
    print("=" * 80)
    print("EXPERIMENT 011: BASELINE RESULTS REVIEW")
    print("=" * 80)
    print()
    
    # Check files exist
    if not baseline_csv.exists():
        print("❌ Baseline results not found!")
        print(f"   Expected: {baseline_csv}")
        print()
        print("   Did the overnight run complete?")
        return
    
    if not original_csv.exists():
        print("❌ Original results not found!")
        print(f"   Expected: {original_csv}")
        return
    
    # Load data
    print("📊 Loading data...")
    original_df = pd.read_csv(original_csv)
    baseline_df = pd.read_csv(baseline_csv)
    
    print(f"   Original: {len(original_df)} rows (Composite only)")
    print(f"   Baselines: {len(baseline_df)} rows (4 baselines × 7 worker counts)")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Group by baseline
    for baseline in baseline_df['baseline'].unique():
        subset = baseline_df[baseline_df['baseline'] == baseline]
        print(f"{baseline.upper()} ({len(subset)} simulations):")
        print(f"  JFI range: {subset['jains_fairness_index'].min():.4f} - {subset['jains_fairness_index'].max():.4f}")
        print(f"  TAR range: {subset['task_assignment_ratio'].min():.2%} - {subset['task_assignment_ratio'].max():.2%}")
        print(f"  Wait time range: {subset['mean_task_wait_time_min'].min():.2f} - {subset['mean_task_wait_time_min'].max():.2f} min")
        print()
    
    # Compare with Composite at matching worker counts
    print("=" * 80)
    print("COMPARISON: BASELINES vs. COMPOSITE (at 4K workers)")
    print("=" * 80)
    print()
    
    composite_4k = original_df[original_df['worker_count'] == 4000].iloc[0]
    baseline_4k = baseline_df[baseline_df['worker_count'] == 4000]
    
    print("Strategy          | JFI    | TAR     | Wait (min) | Util")
    print("-" * 60)
    print(f"{'Composite':<17} | {composite_4k['jains_fairness_index']:.4f} | {composite_4k['task_assignment_ratio']:.4f} | {composite_4k['mean_task_wait_time_min']:>10.2f} | {composite_4k['mean_worker_utilization']:.4f}")
    
    for _, row in baseline_4k.iterrows():
        print(f"{row['baseline_name']:<17} | {row['jains_fairness_index']:.4f} | {row['task_assignment_ratio']:.4f} | {row['mean_task_wait_time_min']:>10.2f} | {row['mean_worker_utilization']:.4f}")
    print()
    
    # Sanity checks
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    print()
    
    issues = []
    
    # Check for missing simulations
    expected_sims = 4 * 7  # 4 baselines × 7 worker counts
    if len(baseline_df) != expected_sims:
        issues.append(f"⚠️  Expected {expected_sims} simulations, got {len(baseline_df)}")
    
    # Check for failed simulations (TAR = 0 or very low)
    failed = baseline_df[baseline_df['task_assignment_ratio'] < 0.01]
    if len(failed) > 0:
        issues.append(f"⚠️  {len(failed)} simulations have TAR < 1% (possible failures)")
        for _, row in failed.iterrows():
            issues.append(f"     - {row['baseline_name']} at {row['worker_count']:,} workers")
    
    # Check for anomalies (JFI = 0)
    anomalies = baseline_df[baseline_df['jains_fairness_index'] == 0]
    if len(anomalies) > 0:
        issues.append(f"⚠️  {len(anomalies)} simulations have JFI = 0 (check results)")
    
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("Review the individual JSON files in baselines/ directory for details.")
        print()
    else:
        print("✅ No obvious issues detected")
        print()
    
    # Merge option
    print("=" * 80)
    print("MERGE OPTIONS")
    print("=" * 80)
    print()
    print("1. Create merged CSV (does NOT modify original)")
    print("2. Review baseline JSONs first")
    print("3. Exit without merging")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Merge data
        print()
        print("Merging data...")
        
        # Need to align columns
        # Original CSV has many columns, baseline CSV has fewer
        # Add missing columns to baseline_df with default values
        
        for col in original_df.columns:
            if col not in baseline_df.columns:
                if col in ['experiment_id', 'name', 'description']:
                    # Will set these below
                    continue
                elif 'time' in col.lower() or 'duration' in col.lower():
                    baseline_df[col] = 0
                elif 'pct' in col or 'ratio' in col or 'cv' in col:
                    baseline_df[col] = 0.0
                else:
                    baseline_df[col] = None
        
        # Set experiment metadata for baselines
        baseline_df['experiment_id'] = range(len(original_df) + 1, len(original_df) + len(baseline_df) + 1)
        baseline_df['name'] = baseline_df.apply(
            lambda row: f"{row['baseline_name']}_{row['worker_count']//1000}K", axis=1
        )
        baseline_df['description'] = baseline_df.apply(
            lambda row: f"{row['baseline_name']} baseline, {row['worker_count']:,} workers, 20K tasks", axis=1
        )
        baseline_df['config_name'] = baseline_df['baseline_name']
        
        # Set weights to NaN for baselines (they don't use composite weights)
        baseline_df['fairness_weight'] = None
        baseline_df['starvation_weight'] = None
        baseline_df['utility_weight'] = None
        baseline_df['soft_threshold'] = None
        
        # Reorder columns to match original
        baseline_df = baseline_df[original_df.columns]
        
        # Concatenate
        merged_df = pd.concat([original_df, baseline_df], ignore_index=True)
        
        # Save
        merged_df.to_csv(merged_csv, index=False)
        
        print(f"✅ Merged CSV saved: {merged_csv.name}")
        print(f"   Total rows: {len(merged_df)}")
        print(f"   - Original (Composite): {len(original_df)}")
        print(f"   - Baselines: {len(baseline_df)}")
        print()
        print("NEXT STEPS:")
        print("  1. Open merged CSV and verify it looks correct")
        print("  2. If happy, rename it to replace the original:")
        print(f"     mv {merged_csv.name} experiment_011_aggregate_results.csv")
        print("  3. Update analysis.ipynb to include baselines in plots")
        
    elif choice == "2":
        print()
        print(f"Baseline JSONs are in: {exp_dir / 'baselines'}")
        print("Run this script again when ready to merge.")
        
    else:
        print()
        print("No changes made.")

if __name__ == "__main__":
    main()

