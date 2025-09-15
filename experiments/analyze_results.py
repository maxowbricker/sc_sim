#!/usr/bin/env python3
"""
Results Analysis Script

Purpose: Load and analyze experiment results from JSON files
Creates summary statistics and basic visualizations

Usage:
    python analyze_results.py results/experiments/parameter_sensitivity_20250912_143022.json
    python analyze_results.py results/experiments/small_scale_validation_20250912_142015.json
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Handle different result formats
    if 'results' in data:
        # New format with metadata
        results = data['results']
        metadata = data.get('metadata', {})
    else:
        # Old format (list of results)
        results = data
        metadata = {}
    
    return results, metadata

def analyze_parameter_sensitivity(results, output_dir):
    """Analyze parameter sensitivity results."""
    df = pd.json_normalize(results)
    
    print("📊 Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Group by scenario
    scenarios = df['scenario'].unique() if 'scenario' in df.columns else ['all']
    
    for scenario in scenarios:
        if scenario == 'all':
            scenario_df = df
        else:
            scenario_df = df[df['scenario'] == scenario]
        
        print(f"\n📈 {scenario.upper()}:")
        print("-" * 40)
        
        # Basic statistics
        key_metrics = ['metrics.jfi', 'metrics.tar', 'metrics.avg_wait_time_min', 'metrics.empty_km_ratio']
        
        for metric in key_metrics:
            if metric in scenario_df.columns:
                values = scenario_df[metric]
                print(f"{metric.split('.')[-1]:<20}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    # Create plots if we have varied parameters
    if 'varied_parameter' in df.columns:
        create_parameter_plots(df, output_dir)
    
    return df

def analyze_strategy_comparison(results, output_dir):
    """Analyze strategy comparison results."""
    df = pd.json_normalize(results)
    
    print("📊 Strategy Comparison Analysis")
    print("=" * 50)
    
    # Group by strategy
    if 'strategy' in df.columns:
        strategies = df['strategy'].unique()
        
        print(f"\n📈 STRATEGY PERFORMANCE:")
        print(f"{'Strategy':<20} {'JFI':<8} {'TAR':<8} {'Wait(min)':<10} {'Runtime(s)':<10}")
        print("-" * 70)
        
        for strategy in strategies:
            strategy_df = df[df['strategy'] == strategy]
            
            # Average metrics for this strategy
            avg_jfi = strategy_df['metrics.jfi'].mean()
            avg_tar = strategy_df['metrics.tar'].mean()
            avg_wait = strategy_df['metrics.avg_wait_time_min'].mean()
            avg_runtime = strategy_df['runtime_seconds'].mean()
            
            print(f"{strategy:<20} {avg_jfi:<8.3f} {avg_tar*100:<7.1f}% {avg_wait:<9.1f} {avg_runtime:<10.2f}")
    
    # Create comparison plots
    if len(df) > 1:
        create_comparison_plots(df, output_dir)
    
    return df

def create_parameter_plots(df, output_dir):
    """Create parameter sensitivity plots."""
    print("\n📊 Creating parameter sensitivity plots...")
    
    scenarios = df['scenario'].unique()
    
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15, 5*len(scenarios)))
    if len(scenarios) == 1:
        axes = axes.reshape(1, -1)
    
    for i, scenario in enumerate(scenarios):
        scenario_df = df[df['scenario'] == scenario]
        
        if 'varied_parameter' in scenario_df.columns:
            varied_param = scenario_df['varied_parameter'].iloc[0]
            param_values = scenario_df[f'parameters.{varied_param}']
            
            # Plot 1: JFI vs Parameter
            axes[i, 0].plot(param_values, scenario_df['metrics.jfi'], 'o-', linewidth=2, markersize=8)
            axes[i, 0].set_xlabel(f'{varied_param}')
            axes[i, 0].set_ylabel('Jain\'s Fairness Index (JFI)')
            axes[i, 0].set_title(f'Fairness vs {varied_param}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot 2: TAR vs Parameter
            axes[i, 1].plot(param_values, scenario_df['metrics.tar']*100, 'o-', color='orange', linewidth=2, markersize=8)
            axes[i, 1].set_xlabel(f'{varied_param}')
            axes[i, 1].set_ylabel('Task Assignment Ratio (%)')
            axes[i, 1].set_title(f'Efficiency vs {varied_param}')
            axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'parameter_sensitivity_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_file}")
    plt.close()

def create_comparison_plots(df, output_dir):
    """Create strategy comparison plots."""
    print("\n📊 Creating strategy comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'strategy' in df.columns:
        strategies = df['strategy'].unique()
        
        # Plot 1: JFI comparison
        jfi_data = [df[df['strategy'] == s]['metrics.jfi'].values for s in strategies]
        axes[0, 0].boxplot(jfi_data, labels=strategies)
        axes[0, 0].set_title('Fairness (JFI) Comparison')
        axes[0, 0].set_ylabel('Jain\'s Fairness Index')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: TAR comparison
        tar_data = [df[df['strategy'] == s]['metrics.tar'].values * 100 for s in strategies]
        axes[0, 1].boxplot(tar_data, labels=strategies)
        axes[0, 1].set_title('Efficiency (TAR) Comparison')
        axes[0, 1].set_ylabel('Task Assignment Ratio (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Wait Time comparison
        wait_data = [df[df['strategy'] == s]['metrics.avg_wait_time_min'].values for s in strategies]
        axes[1, 0].boxplot(wait_data, labels=strategies)
        axes[1, 0].set_title('Wait Time Comparison')
        axes[1, 0].set_ylabel('Average Wait Time (min)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Runtime comparison
        runtime_data = [df[df['strategy'] == s]['runtime_seconds'].values for s in strategies]
        axes[1, 1].boxplot(runtime_data, labels=strategies)
        axes[1, 1].set_title('Runtime Comparison')
        axes[1, 1].set_ylabel('Runtime (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'strategy_comparison_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_file}")
    plt.close()

def main():
    """Main analysis function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"📊 Analyzing results from: {results_file}")
    
    # Load results
    results, metadata = load_results(results_file)
    
    # Create output directory
    output_dir = results_file.parent / f"analysis_{results_file.stem}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Determine analysis type
    experiment_type = metadata.get('experiment_type', 'unknown')
    
    if experiment_type == 'parameter_sensitivity' or any('scenario' in r for r in results if isinstance(r, dict)):
        df = analyze_parameter_sensitivity(results, output_dir)
    else:
        df = analyze_strategy_comparison(results, output_dir)
    
    # Save processed data
    csv_file = output_dir / 'processed_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Processed data saved to: {csv_file}")
    
    # Create summary report
    summary_file = output_dir / 'summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Analysis Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Results file: {results_file}\n")
        f.write(f"Analysis date: {pd.Timestamp.now()}\n")
        f.write(f"Total experiments: {len(results)}\n\n")
        
        if metadata:
            f.write("Experiment Metadata:\n")
            f.write("-" * 20 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("Key Findings:\n")
        f.write("-" * 20 + "\n")
        
        # Basic statistics
        if 'metrics.jfi' in df.columns:
            f.write(f"JFI range: {df['metrics.jfi'].min():.3f} - {df['metrics.jfi'].max():.3f}\n")
        if 'metrics.tar' in df.columns:
            f.write(f"TAR range: {df['metrics.tar'].min()*100:.1f}% - {df['metrics.tar'].max()*100:.1f}%\n")
        if 'runtime_seconds' in df.columns:
            f.write(f"Runtime range: {df['runtime_seconds'].min():.1f}s - {df['runtime_seconds'].max():.1f}s\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    print(f"\n🎯 Analysis complete! Check the output directory for detailed results.")

if __name__ == "__main__":
    main()

