#!/usr/bin/env python3
"""
Test lambda1 effect on EWMA CV (the actual fairness metric we're optimizing)
Instead of focusing on JFI (task count equality), look at EWMA CV (idle time fairness).
"""

import pandas as pd
from models.loader import DataLoader
from simulator.simulation import run_simulation
from config import create_composite_config

def test_lambda1_ewma_focus():
    """Test λ₁ effect specifically on EWMA CV - the metric we're actually optimizing."""
    
    # Load quarter-fixed datasets
    loader = DataLoader("didi", "/Users/maxapple/Documents/GitHub/sc_sim/data/didi")
    workers_df, tasks_df = loader.get_simulation_data(
        max_workers=1000, 
        max_tasks=2000
    )
    
    # Test extreme λ₁ values
    lambda1_values = [0.1, 5.0, 25.0, 100.0]
    results = []
    
    print("🎯 Testing λ₁ Effect on EWMA CV (Actual Fairness Metric)")
    print("=" * 70)
    
    for i, λ1 in enumerate(lambda1_values, 1):
        print(f"\n🔧 Testing λ₁={λ1} ({i}/{len(lambda1_values)})")
        
        # Create config with specific λ₁
        config = create_composite_config(
            assignment_strategy="composite",
            λ1=λ1,
            λ2=1.0,  # Keep starvation constant
            λ3=0.5,  # Keep utility constant
            fairness_metric="ewma"
        )
        
        # Run simulation
        summary = run_simulation(workers_df.to_dict('records'), 
                                tasks_df.to_dict('records'), 
                                sim_config=config)
        
        # Extract key metrics
        result = {
            'λ₁': λ1,
            'completed_tasks': summary.get('completed_tasks', 0),
            'TAR': summary.get('completed_tasks', 0) / len(tasks_df) * 100,
            'JFI': summary.get('final_jains_fairness_index', 0),
            'EWMA_CV': summary.get('final_ewma_cv', 0),  # ⭐ This is what we're optimizing!
            'avg_wait': summary.get('total_wait_min', 0) / max(1, summary.get('completed_tasks', 1)),
        }
        results.append(result)
        
        print(f"   📊 TAR: {result['TAR']:.1f}%")
        print(f"   📊 JFI: {result['JFI']:.4f} (task count equality)")
        print(f"   ⭐ EWMA CV: {result['EWMA_CV']:.4f} (idle time fairness - what we're optimizing!)")
        print(f"   ⏰ Avg Wait: {result['avg_wait']:.1f}min")
    
    # Analysis
    print(f"\n{'='*70}")
    print("📈 RESULTS ANALYSIS - Focus on EWMA CV")
    print(f"{'='*70}")
    
    df = pd.DataFrame(results)
    
    print(f"{'λ₁':<8} {'TAR%':<8} {'JFI':<10} {'EWMA_CV':<12} {'Wait(min)':<10}")
    print("-" * 55)
    
    for _, row in df.iterrows():
        print(f"{row['λ₁']:<8} {row['TAR']:<8.1f} {row['JFI']:<10.4f} {row['EWMA_CV']:<12.4f} {row['avg_wait']:<10.1f}")
    
    # Check for EWMA CV differences
    ewma_min = df['EWMA_CV'].min()
    ewma_max = df['EWMA_CV'].max()
    ewma_range = ewma_max - ewma_min
    ewma_relative_change = (ewma_range / ewma_min * 100) if ewma_min > 0 else 0
    
    print(f"\n🔍 EWMA CV Analysis:")
    print(f"   Range: {ewma_min:.4f} to {ewma_max:.4f}")
    print(f"   Absolute Change: {ewma_range:.4f}")
    print(f"   Relative Change: {ewma_relative_change:.1f}%")
    
    if ewma_relative_change > 5:
        print(f"   ✅ λ₁ IS affecting fairness! {ewma_relative_change:.1f}% change in EWMA CV")
    else:
        print(f"   ❌ λ₁ effect is minimal. {ewma_relative_change:.1f}% change in EWMA CV")
        
    # JFI vs EWMA CV comparison
    jfi_min = df['JFI'].min()
    jfi_max = df['JFI'].max()
    jfi_relative_change = ((jfi_max - jfi_min) / jfi_min * 100) if jfi_min > 0 else 0
    
    print(f"\n📊 Metric Sensitivity Comparison:")
    print(f"   JFI (task equality) change: {jfi_relative_change:.1f}%")
    print(f"   EWMA CV (idle fairness) change: {ewma_relative_change:.1f}%")
    
    if ewma_relative_change > jfi_relative_change:
        print(f"   ✅ EWMA CV is more sensitive to λ₁ changes (as expected!)")
    else:
        print(f"   ⚠️  JFI is more sensitive - unexpected given our optimization target")

if __name__ == "__main__":
    test_lambda1_ewma_focus()
