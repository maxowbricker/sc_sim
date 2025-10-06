#!/usr/bin/env python3
"""
Parameter Sweep Analysis Tool
=============================

Tool for analyzing comprehensive parameter sweep results to understand
relationships between λ₁, λ₂, λ₃, and soft_threshold parameters.

Usage:
    python experiments/analyze_parameter_sweep.py ../../../results/comprehensive_parameter_sweep_YYYYMMDD_HHMMSS.json
    
Features:
- Parameter correlation analysis
- Optimal configuration identification
- Trade-off visualization preparation
- Pareto frontier analysis
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_sweep_results(filepath):
    """Load parameter sweep results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['results'])
        config = data.get('config', {})
        analysis = data.get('analysis', {})
        
        print(f"✅ Loaded {len(results_df)} experiments from {filepath}")
        print(f"🕒 Original experiment duration: {data.get('execution_time_minutes', 0):.1f} minutes")
        
        return results_df, config, analysis
    
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        sys.exit(1)

def analyze_parameter_correlations(df):
    """Analyze correlations between parameters and outcomes."""
    
    print("\n📊 PARAMETER CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Select key columns for analysis
    param_cols = ['lambda1', 'lambda2', 'lambda3', 'soft_threshold']
    metric_cols = ['jfi', 'tar', 'avg_pickup_distance', 'ewma_cv']
    
    # Calculate correlations
    corr_data = {}
    
    for metric in metric_cols:
        corr_data[metric] = {}
        for param in param_cols:
            correlation = df[param].corr(df[metric])
            corr_data[metric][param] = correlation
    
    # Display correlation table
    corr_df = pd.DataFrame(corr_data).round(3)
    print("Parameter-Metric Correlations:")
    print("(Positive = parameter increase improves metric)")
    print(corr_df.to_string())
    
    return corr_df

def find_optimal_configurations(df):
    """Find optimal configurations for different objectives."""
    
    print("\n🎯 OPTIMAL CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    objectives = {
        'Highest JFI': ('jfi', 'max'),
        'Highest TAR': ('tar', 'max'), 
        'Best Balanced (JFI>0.85 & TAR>95%)': (None, 'balanced'),
        'Shortest Pickup Distance': ('avg_pickup_distance', 'min'),
        'Lowest EWMA CV': ('ewma_cv', 'min'),
        'Best Combined Score': (None, 'combined')
    }
    
    results = {}
    
    for obj_name, (metric, direction) in objectives.items():
        print(f"\n🔍 {obj_name}:")
        
        if direction == 'balanced':
            # Find configurations that meet both criteria
            candidates = df[(df['jfi'] > 0.85) & (df['tar'] > 95)]
            if len(candidates) > 0:
                best_config = candidates.loc[candidates['jfi'].idxmax()]
                results[obj_name] = best_config
            else:
                print("   ❌ No configurations meet both criteria")
                continue
                
        elif direction == 'combined':
            # Weighted combination: 60% JFI, 40% TAR (normalized)
            df['combined_score'] = (df['jfi'] * 0.6) + (df['tar'] / 100 * 0.4)
            best_config = df.loc[df['combined_score'].idxmax()]
            results[obj_name] = best_config
            
        else:
            # Simple optimization
            if direction == 'max':
                best_config = df.loc[df[metric].idxmax()]
            else:  # min
                best_config = df.loc[df[metric].idxmin()]
            results[obj_name] = best_config
        
        # Display configuration
        config = results[obj_name]
        print(f"   λ₁={config['lambda1']:>5.1f}, λ₂={config['lambda2']:>5.1f}, λ₃={config['lambda3']:>5.1f}, thresh={config['soft_threshold']:>5.2f}")
        print(f"   JFI={config['jfi']:.3f}, TAR={config['tar']:.1f}%, Pickup={config['avg_pickup_distance']:.2f}km")
        
        if 'combined_score' in df.columns:
            print(f"   Combined Score={config.get('combined_score', 0):.3f}")
    
    return results

def analyze_parameter_ranges(df):
    """Analyze optimal parameter ranges for different performance levels."""
    
    print("\n📈 PARAMETER RANGE ANALYSIS")
    print("=" * 60)
    
    # Define performance tiers
    tiers = {
        'Excellent (JFI>0.9)': df[df['jfi'] > 0.9],
        'Good (JFI>0.85)': df[df['jfi'] > 0.85],
        'Balanced (JFI>0.85 & TAR>95%)': df[(df['jfi'] > 0.85) & (df['tar'] > 95)],
        'High Efficiency (Pickup<2km)': df[df['avg_pickup_distance'] < 2.0]
    }
    
    params = ['lambda1', 'lambda2', 'lambda3', 'soft_threshold']
    
    for tier_name, tier_df in tiers.items():
        if len(tier_df) == 0:
            continue
            
        print(f"\n🏆 {tier_name} ({len(tier_df)} configs):")
        
        for param in params:
            values = tier_df[param]
            print(f"   {param:>15}: [{values.min():>5.1f} - {values.max():>5.1f}] (avg: {values.mean():>5.1f})")

def identify_trade_offs(df):
    """Identify key trade-offs between different objectives."""
    
    print("\n⚖️  TRADE-OFF ANALYSIS")
    print("=" * 60)
    
    trade_offs = [
        ('JFI vs Efficiency', 'jfi', 'avg_pickup_distance', 'negative'),
        ('JFI vs TAR', 'jfi', 'tar', 'positive'),  
        ('TAR vs Efficiency', 'tar', 'avg_pickup_distance', 'negative'),
        ('Fairness Weight vs Efficiency', 'lambda1', 'avg_pickup_distance', 'positive'),
        ('Utility Weight vs JFI', 'lambda3', 'jfi', 'negative')
    ]
    
    for trade_name, metric1, metric2, expected_relationship in trade_offs:
        correlation = df[metric1].corr(df[metric2])
        
        relationship_strength = abs(correlation)
        if relationship_strength > 0.5:
            strength = "Strong"
        elif relationship_strength > 0.3:
            strength = "Moderate"  
        else:
            strength = "Weak"
        
        direction = "Positive" if correlation > 0 else "Negative"
        
        print(f"📊 {trade_name}:")
        print(f"   Correlation: {correlation:.3f} ({strength} {direction})")
        
        # Find extreme examples
        q75_m1 = df[metric1].quantile(0.75)
        q25_m1 = df[metric1].quantile(0.25)
        
        high_m1 = df[df[metric1] >= q75_m1]
        low_m1 = df[df[metric1] <= q25_m1]
        
        if len(high_m1) > 0 and len(low_m1) > 0:
            high_m1_avg_m2 = high_m1[metric2].mean()
            low_m1_avg_m2 = low_m1[metric2].mean()
            
            print(f"   High {metric1}: {metric2} avg = {high_m1_avg_m2:.3f}")
            print(f"   Low {metric1}: {metric2} avg = {low_m1_avg_m2:.3f}")
        
        print()

def generate_summary_report(df, config, analysis, optimal_configs):
    """Generate a comprehensive summary report."""
    
    print("\n📋 EXECUTIVE SUMMARY")
    print("=" * 60)
    
    total_experiments = len(df)
    successful_balanced = len(df[(df['jfi'] > 0.85) & (df['tar'] > 95)])
    high_jfi = len(df[df['jfi'] > 0.9])
    
    print(f"Total Experiments: {total_experiments:,}")
    print(f"Balanced Success (JFI>0.85 & TAR>95%): {successful_balanced} ({successful_balanced/total_experiments*100:.1f}%)")
    print(f"Excellent JFI (>0.9): {high_jfi} ({high_jfi/total_experiments*100:.1f}%)")
    
    print(f"\n🎯 KEY FINDINGS:")
    
    # Best overall configuration
    if 'Best Combined Score' in optimal_configs:
        best = optimal_configs['Best Combined Score']
        print(f"   • Best Overall: λ₁={best['lambda1']:.1f}, λ₂={best['lambda2']:.1f}, λ₃={best['lambda3']:.1f}, thresh={best['soft_threshold']:.2f}")
        print(f"     Performance: JFI={best['jfi']:.3f}, TAR={best['tar']:.1f}%, Pickup={best['avg_pickup_distance']:.2f}km")
    
    # Parameter insights
    high_performers = df[df['jfi'] > 0.85]
    if len(high_performers) > 0:
        print(f"   • High-performing λ₁ range: {high_performers['lambda1'].min():.1f} - {high_performers['lambda1'].max():.1f}")
        print(f"   • High-performing λ₂ range: {high_performers['lambda2'].min():.1f} - {high_performers['lambda2'].max():.1f}")
        print(f"   • High-performing λ₃ range: {high_performers['lambda3'].min():.1f} - {high_performers['lambda3'].max():.1f}")
    
    print(f"\n🔧 RECOMMENDED NEXT STEPS:")
    print(f"   1. Focus detailed experiments on high-performing parameter ranges")
    print(f"   2. Create visualization plots for parameter interactions")  
    print(f"   3. Test top configurations on different datasets")
    print(f"   4. Consider ensemble approaches combining multiple good configurations")
    
def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive parameter sweep results')
    parser.add_argument('results_file', help='Path to parameter sweep results JSON file')
    parser.add_argument('--export-csv', help='Export analysis to CSV file')
    
    args = parser.parse_args()
    
    # Load data
    df, config, analysis = load_sweep_results(args.results_file)
    
    # Run analysis
    correlations = analyze_parameter_correlations(df)
    optimal_configs = find_optimal_configurations(df)
    analyze_parameter_ranges(df)
    identify_trade_offs(df)
    generate_summary_report(df, config, analysis, optimal_configs)
    
    # Export if requested
    if args.export_csv:
        output_path = Path(args.export_csv)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Raw results exported to: {output_path}")
        
        # Export correlation matrix
        corr_path = output_path.parent / f"{output_path.stem}_correlations.csv"
        correlations.to_csv(corr_path)
        print(f"💾 Correlations exported to: {corr_path}")

if __name__ == "__main__":
    main()

