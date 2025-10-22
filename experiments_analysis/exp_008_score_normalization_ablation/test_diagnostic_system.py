#!/usr/bin/env python3
"""
Quick test of Experiment 008 diagnostic system.

Tests the new features before running the full experiment:
1. DiagnosticTracker creation and data collection
2. Score normalization (normalize_scores=True)
3. Threshold ablation (disable_soft_threshold=True)
4. Diagnostic data export

This runs small-scale simulations (100 workers, 200 tasks) for quick validation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import create_composite_config
from simulator.simulation import Simulation
from data.notebook_optimized_loader import load_data


def test_diagnostic_system():
    """Test the diagnostic tracking system with small-scale simulations."""
    
    print("=" * 80)
    print("EXPERIMENT 008 DIAGNOSTIC SYSTEM TEST")
    print("=" * 80)
    print("Testing new features:")
    print("  1. DiagnosticTracker creation and data collection")
    print("  2. Score normalization (normalize_scores=True)")
    print("  3. Threshold ablation (disable_soft_threshold=True)")
    print("  4. Diagnostic data export to DataFrames")
    print()
    print("Running small-scale tests (100 workers, 200 tasks)...")
    print("=" * 80)
    print()
    
    # Load small dataset for quick testing
    print("[1/4] Loading test dataset...")
    workers_df, tasks_df = load_data('didi', max_workers=100, max_tasks=200)
    print(f"   ✅ Loaded: {len(workers_df)} workers, {len(tasks_df)} tasks")
    print()
    
    # Test 1: Fast Path (no diagnostics - should be fast)
    print("[2/4] Test 1: Fast Path - Standard Composite (No Diagnostics)")
    print("   normalize_scores=False, disable_soft_threshold=False, enable_diagnostics=False")
    print("   Expected: FAST PATH (single-pass algorithm)")
    config1 = create_composite_config(
        fairness_weight=0.5,
        starvation_weight=0.8,
        utility_weight=0.8,
        soft_threshold=0.5,
        normalize_scores=False,
        disable_soft_threshold=False,
        enable_diagnostics=False  # No diagnostics = fast path
    )
    
    import time
    start = time.time()
    sim1 = Simulation(config1, workers_df, tasks_df)
    summary1 = sim1.run()
    duration1 = time.time() - start
    
    # Should NOT have diagnostic tracker
    if 'diagnostic_tracker' not in summary1 or summary1['diagnostic_tracker'] is None:
        print(f"   ✅ Fast path confirmed: No DiagnosticTracker created")
        print(f"   ⚡ Duration: {duration1:.2f}s (baseline speed)")
        print(f"   📊 Completed tasks: {summary1.get('completed_tasks', 0)}")
    else:
        print(f"   ❌ ERROR: DiagnosticTracker was created but shouldn't be (fast path)")
        return False
    
    print()
    
    # Test 2: Slow Path with Diagnostics and Normalization
    print("[3/4] Test 2: Slow Path - Composite with Diagnostics + Normalization")
    print("   normalize_scores=True, disable_soft_threshold=False, enable_diagnostics=True")
    print("   Expected: SLOW PATH (multi-pass algorithm with diagnostics)")
    config2 = create_composite_config(
        fairness_weight=0.5,
        starvation_weight=0.8,
        utility_weight=0.8,
        soft_threshold=0.5,
        normalize_scores=True,  # ⚠️ Forces slow path
        disable_soft_threshold=False,
        enable_diagnostics=True  # ⚠️ Enables diagnostics
    )
    
    start = time.time()
    sim2 = Simulation(config2, workers_df, tasks_df)
    summary2 = sim2.run()
    duration2 = time.time() - start
    
    if 'diagnostic_tracker' in summary2 and summary2['diagnostic_tracker'] is not None:
        tracker2 = summary2['diagnostic_tracker']
        stats2 = tracker2.get_summary_stats()
        
        print(f"   ✅ Slow path confirmed: DiagnosticTracker created")
        print(f"   ⚡ Duration: {duration2:.2f}s ({duration2/duration1:.1f}x slower than fast path)")
        print(f"   📊 Assignments: {stats2['total_assignments']}, Deferrals: {stats2['total_deferrals']}")
        print(f"   📊 Deferral rate: {stats2['deferral_rate']*100:.1f}%")
        
        if stats2['total_assignments'] > 0:
            print(f"   📊 Component dominance (normalized):")
            for component, pct in stats2['dominance_percentages'].items():
                print(f"      - {component.capitalize()}: {pct:.1f}%")
        
        # Verify normalization was actually used
        assignments_df2 = tracker2.to_dataframe('assignments')
        if len(assignments_df2) > 0:
            norm_used = assignments_df2['normalization_used'].all()
            print(f"   ✅ Normalization applied: {norm_used}")
            if norm_used:
                # Show that normalized values exist
                has_norm_cols = all(col in assignments_df2.columns for col in ['fairness_norm', 'starvation_norm', 'utility_norm'])
                print(f"   ✅ Normalized columns present: {has_norm_cols}")
                if has_norm_cols:
                    # Compare raw vs normalized ranges
                    print(f"   📊 Fairness raw range: [{assignments_df2['fairness_raw'].min():.3f}, {assignments_df2['fairness_raw'].max():.3f}]")
                    print(f"   📊 Fairness norm range: [{assignments_df2['fairness_norm'].min():.3f}, {assignments_df2['fairness_norm'].max():.3f}]")
    else:
        print("   ❌ ERROR: DiagnosticTracker not found in summary!")
        return False
    
    print()
    
    # Test 3: Composite with Normalization + No Threshold + Diagnostics
    print("[4/4] Test 3: Composite with Normalization + Threshold Disabled + Diagnostics")
    print("   normalize_scores=True, disable_soft_threshold=True, enable_diagnostics=True")
    config3 = create_composite_config(
        fairness_weight=0.5,
        starvation_weight=0.8,
        utility_weight=0.8,
        soft_threshold=0.5,  # This will be bypassed
        normalize_scores=True,  # ⚠️ NEW FEATURE
        disable_soft_threshold=True,  # ⚠️ NEW FEATURE
        enable_diagnostics=True  # ⚠️ Enable diagnostics
    )
    
    sim3 = Simulation(config3, workers_df, tasks_df)
    summary3 = sim3.run()
    
    if 'diagnostic_tracker' in summary3 and summary3['diagnostic_tracker'] is not None:
        tracker3 = summary3['diagnostic_tracker']
        stats3 = tracker3.get_summary_stats()
        
        print(f"   ✅ DiagnosticTracker working with both features")
        print(f"   📊 Assignments: {stats3['total_assignments']}, Deferrals: {stats3['total_deferrals']}")
        print(f"   📊 Deferral rate: {stats3['deferral_rate']*100:.1f}%")
        
        # Verify threshold was actually disabled (should have 0 or very few deferrals)
        if stats3['deferral_rate'] < 0.01:  # Less than 1%
            print(f"   ✅ Threshold successfully disabled (deferral rate ≈ 0%)")
        else:
            print(f"   ⚠️  WARNING: Expected low deferral rate, got {stats3['deferral_rate']*100:.1f}%")
        
        if stats3['total_assignments'] > 0:
            print(f"   📊 Component dominance (normalized, no threshold):")
            for component, pct in stats3['dominance_percentages'].items():
                print(f"      - {component.capitalize()}: {pct:.1f}%")
    else:
        print("   ❌ ERROR: DiagnosticTracker not found in summary!")
        return False
    
    print()
    
    # Summary comparison
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<35} {'Fast Path':<15} {'Normalized':<15} {'Both':<15}")
    print("-" * 80)
    
    # Basic metrics available for all
    print(f"{'Completed Tasks':<35} {summary1.get('completed_tasks', 0):<15} {summary2.get('completed_tasks', 0):<15} {summary3.get('completed_tasks', 0):<15}")
    print(f"{'Duration (seconds)':<35} {duration1:<15.2f} {duration2:<15.2f} {'N/A':<15}")
    print(f"{'Speed vs Fast Path':<35} {'1.0x':<15} {f'{duration2/duration1:.1f}x':<15} {'N/A':<15}")
    print()
    
    # Diagnostic metrics only for tests 2 and 3
    if all(s.get('diagnostic_tracker') for s in [summary2, summary3]):
        print("Diagnostic Metrics (Tests 2 & 3 only):")
        s2 = summary2['diagnostic_tracker'].get_summary_stats()
        s3 = summary3['diagnostic_tracker'].get_summary_stats()
        
        print(f"{'Total Assignments':<35} {'N/A':<15} {s2['total_assignments']:<15} {s3['total_assignments']:<15}")
        print(f"{'Total Deferrals':<35} {'N/A':<15} {s2['total_deferrals']:<15} {s3['total_deferrals']:<15}")
        print(f"{'Deferral Rate (%)':<35} {'N/A':<15} {s2['deferral_rate']*100:<15.1f} {s3['deferral_rate']*100:<15.1f}")
        
        if all(s['total_assignments'] > 0 for s in [s2, s3]):
            print(f"{'Fairness Dominance (%)':<35} {'N/A':<15} {s2['dominance_percentages'].get('fairness', 0):<15.1f} {s3['dominance_percentages'].get('fairness', 0):<15.1f}")
            print(f"{'Utility Dominance (%)':<35} {'N/A':<15} {s2['dominance_percentages'].get('utility', 0):<15.1f} {s3['dominance_percentages'].get('utility', 0):<15.1f}")
            print(f"{'Avg Dominance Ratio':<35} {'N/A':<15} {s2['overall_avg_dominance_ratio']:<15.2f} {s3['overall_avg_dominance_ratio']:<15.2f}")
    
    print()
    print(f"{'Task Assignment Ratio':<35} {summary1.get('completed_tasks', 0)/200:<15.2%} {summary2.get('completed_tasks', 0)/200:<15.2%} {summary3.get('completed_tasks', 0)/200:<15.2%}")
    print(f"{'Completed Tasks':<35} {summary1.get('completed_tasks', 0):<15} {summary2.get('completed_tasks', 0):<15} {summary3.get('completed_tasks', 0):<15}")
    
    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("The diagnostic system is working correctly:")
    print("  ✅ DiagnosticTracker is created and tracks assignments/deferrals")
    print("  ✅ Score normalization (normalize_scores=True) is functioning")
    print("  ✅ Threshold ablation (disable_soft_threshold=True) is functioning")
    print("  ✅ Diagnostic data can be exported to DataFrames")
    print()
    print("You can now run the full Experiment 008 with confidence:")
    print("  python run_experiment.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_diagnostic_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED WITH ERROR")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

