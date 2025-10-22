"""
Combine Experiment 009 and 010 Results
======================================

This script safely combines the aggregate results from:
- Experiment 009: Comprehensive Parameter Sweep (42 experiments)
- Experiment 010: Extended Boundaries - Pareto High-Resolution (21 experiments)

Into a unified dataset for joint analysis (63 total experiments).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    exp_009_csv = base_dir / "exp_009_comprehensive_parameter_sweep/data/exp_009_20251019_232730/experiment_009_aggregate_results.csv"
    exp_010_csv = base_dir / "exp_010_extended_boundaries/data/exp_010_20251021_000315/experiment_010_aggregate_results.csv"
    output_dir = Path(__file__).parent / "data"
    output_csv = output_dir / "experiment_009+010_combined_results.csv"
    
    print("=" * 70)
    print("COMBINING EXPERIMENTS 009 + 010")
    print("=" * 70)
    print()
    
    # Load both datasets
    print(f"📂 Loading Experiment 009: {exp_009_csv.name}")
    df_009 = pd.read_csv(exp_009_csv)
    print(f"   ✅ Loaded {len(df_009)} experiments")
    print(f"   Columns: {list(df_009.columns)}")
    print()
    
    print(f"📂 Loading Experiment 010: {exp_010_csv.name}")
    df_010 = pd.read_csv(exp_010_csv)
    print(f"   ✅ Loaded {len(df_010)} experiments")
    print(f"   Columns: {list(df_010.columns)}")
    print()
    
    # Add source column to track which experiment each row came from
    df_009['source_experiment'] = 'exp_009'
    df_010['source_experiment'] = 'exp_010'
    
    # Add original experiment IDs before combining
    df_009['original_experiment_id'] = df_009['experiment_id']
    df_010['original_experiment_id'] = df_010['experiment_id']
    
    # Combine datasets
    print("🔗 Combining datasets...")
    df_combined = pd.concat([df_009, df_010], ignore_index=True)
    
    # Renumber experiment_id to be sequential (1-63)
    df_combined['experiment_id'] = range(1, len(df_combined) + 1)
    
    # Reorder columns to put source info near the front
    cols = df_combined.columns.tolist()
    # Move source_experiment and original_experiment_id after experiment_id
    cols.remove('source_experiment')
    cols.remove('original_experiment_id')
    new_order = [cols[0]] + ['source_experiment', 'original_experiment_id'] + cols[1:]
    df_combined = df_combined[new_order]
    
    print(f"   ✅ Combined dataset has {len(df_combined)} total experiments")
    print()
    
    # Summary statistics
    print("📊 COMBINED DATASET SUMMARY:")
    print("-" * 70)
    print(f"Total Experiments: {len(df_combined)}")
    print(f"  - From Exp 009: {len(df_009)} experiments")
    print(f"  - From Exp 010: {len(df_010)} experiments")
    print()
    
    print("Strategy Breakdown:")
    print(df_combined['strategy'].value_counts().to_string())
    print()
    
    # Composite strategy stats
    df_composite = df_combined[df_combined['strategy'] == 'composite']
    print(f"Composite Strategy Analysis ({len(df_composite)} experiments):")
    print(f"  JFI Range: {df_composite['jains_fairness_index'].min():.4f} - {df_composite['jains_fairness_index'].max():.4f}")
    print(f"  Wait Time Range: {df_composite['mean_task_wait_time_min'].min():.4f} - {df_composite['mean_task_wait_time_min'].max():.4f} min")
    print(f"  TAR Range: {df_composite['task_assignment_ratio'].min():.4f} - {df_composite['task_assignment_ratio'].max():.4f}")
    print()
    
    # Parameter ranges
    if 'fairness_weight' in df_composite.columns:
        print("Parameter Ranges (Composite only):")
        print(f"  λ₁ (Fairness):    {df_composite['fairness_weight'].min():.1f} - {df_composite['fairness_weight'].max():.1f}")
        print(f"  λ₂ (Starvation):  {df_composite['starvation_weight'].min():.1f} - {df_composite['starvation_weight'].max():.1f}")
        print(f"  λ₃ (Utility):     {df_composite['utility_weight'].min():.1f} - {df_composite['utility_weight'].max():.1f}")
        print(f"  θ (Threshold):    {df_composite['soft_threshold'].min():.1f} - {df_composite['soft_threshold'].max():.1f}")
        print()
    
    # Save combined dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_csv, index=False)
    print(f"💾 Saved combined results to:")
    print(f"   {output_csv}")
    print()
    
    # Create a summary report
    summary_file = output_dir / "COMBINED_DATASET_INFO.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMBINED DATASET: EXPERIMENTS 009 + 010\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Experiments: {len(df_combined)}\n")
        f.write(f"  - Experiment 009: {len(df_009)} experiments (Comprehensive Parameter Sweep)\n")
        f.write(f"  - Experiment 010: {len(df_010)} experiments (Pareto High-Resolution Sweep)\n\n")
        f.write("SOURCE FILES:\n")
        f.write(f"  - {exp_009_csv}\n")
        f.write(f"  - {exp_010_csv}\n\n")
        f.write("OUTPUT FILE:\n")
        f.write(f"  - {output_csv}\n\n")
        f.write("NOTES:\n")
        f.write("  - 'source_experiment' column indicates origin (exp_009 or exp_010)\n")
        f.write("  - 'original_experiment_id' preserves the original experiment ID\n")
        f.write("  - 'experiment_id' is renumbered sequentially (1-63)\n")
        f.write("  - Original data files remain unchanged\n")
    
    print(f"📝 Created summary info file:")
    print(f"   {summary_file}")
    print()
    print("✅ COMBINATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()




