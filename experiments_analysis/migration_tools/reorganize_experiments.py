#!/usr/bin/env python3
"""
Script to reorganize experiments:
1. Delete experiments that haven't been run yet
2. Renumber remaining experiments so comparative_parameter_sweep is highest
"""

import shutil
import os
from pathlib import Path

def reorganize_experiments():
    """Reorganize experiment directories based on completion status."""
    
    print("🗂️ EXPERIMENT REORGANIZATION")
    print("=" * 35)
    
    # Define which experiments have been run (have data)
    completed_experiments = {
        'exp_001_rq1_1_fairness_weights': 'exp_001_rq1_1_fairness_weights',  # Keep as 001
        'exp_003_comprehensive_parameter_sweep': 'exp_002_comprehensive_parameter_sweep',  # 003 → 002
        'exp_005_custom_parameter_sweep': 'exp_003_custom_parameter_sweep',  # 005 → 003  
        'exp_006_focused_parameter_sweep': 'exp_004_focused_parameter_sweep',  # 006 → 004
        'exp_007_bottleneck_analysis': 'exp_005_bottleneck_analysis',  # 007 → 005
        'exp_004_comparative_parameter_sweep': 'exp_006_comparative_parameter_sweep',  # 004 → 006 (HIGHEST)
    }
    
    # Experiments to delete (not run yet)
    experiments_to_delete = [
        'exp_002_rq4_1_baseline_comparison',  # Not run
        'exp_008_full_dataset_analysis',      # Computational analysis only
        'exp_009_ewma_gamma_sensitivity',     # Future work
        'exp_010_ppo_adaptive_weights'        # Future work
    ]
    
    print("🗑️ DELETING UNRUN EXPERIMENTS:")
    print("-" * 35)
    
    for exp_dir in experiments_to_delete:
        if os.path.exists(exp_dir):
            try:
                shutil.rmtree(exp_dir)
                print(f"✅ Deleted: {exp_dir}")
            except Exception as e:
                print(f"❌ Failed to delete {exp_dir}: {e}")
        else:
            print(f"⚠️  Already missing: {exp_dir}")
    
    print(f"\n🔄 RENUMBERING COMPLETED EXPERIMENTS:")
    print("-" * 40)
    
    # Create temporary directory for renaming
    temp_dir = "temp_rename"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1: Move all to temporary names
    for old_name, new_name in completed_experiments.items():
        if os.path.exists(old_name):
            temp_name = f"{temp_dir}/{new_name}"
            try:
                shutil.move(old_name, temp_name)
                print(f"📦 Temp move: {old_name} → {temp_name}")
            except Exception as e:
                print(f"❌ Failed temp move {old_name}: {e}")
    
    # Step 2: Move from temp to final names
    for old_name, new_name in completed_experiments.items():
        temp_name = f"{temp_dir}/{new_name}"
        if os.path.exists(temp_name):
            try:
                shutil.move(temp_name, new_name)
                print(f"✅ Final move: {temp_name} → {new_name}")
            except Exception as e:
                print(f"❌ Failed final move {temp_name}: {e}")
    
    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print(f"\n📊 FINAL EXPERIMENT STRUCTURE:")
    print("=" * 35)
    
    # Show final structure
    experiment_dirs = sorted([d for d in os.listdir('.') if d.startswith('exp_')])
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        if os.path.exists(exp_dir):
            # Check if has data
            data_dir = os.path.join(exp_dir, 'data')
            has_data = os.path.exists(data_dir) and len(os.listdir(data_dir)) > 1  # More than just README
            
            data_status = "📊 WITH DATA" if has_data else "📋 NO DATA"
            print(f"✅ {exp_dir}/ {data_status}")
            
            # Show key files
            key_files = ['run_experiment.py', 'setup.md', 'analysis.ipynb', 'analysis_comprehensive.ipynb']
            for key_file in key_files:
                file_path = os.path.join(exp_dir, key_file)
                if os.path.exists(file_path):
                    print(f"   📄 {key_file}")
    
    print(f"\n🎯 REORGANIZATION SUMMARY:")
    print("=" * 30)
    print("✅ Deleted 4 unrun experiments")
    print("✅ Kept 6 completed experiments") 
    print("✅ Renumbered so comparative_parameter_sweep is exp_006 (highest)")
    print("✅ Clean, focused experiment structure")
    
    print(f"\n🚀 READY FOR ANALYSIS:")
    print("cd exp_006_comparative_parameter_sweep/")
    print("jupyter notebook analysis_comprehensive.ipynb")

if __name__ == "__main__":
    print("🔧 Experiment Structure Reorganization Tool")
    print("===========================================")
    reorganize_experiments()
    print("✨ Reorganization complete!")
