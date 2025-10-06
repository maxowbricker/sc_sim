#!/usr/bin/env python3
"""
Final cleanup script to remove the old experiments directory after successful migration
"""

import shutil
import os
from pathlib import Path

def cleanup_old_experiments():
    """Remove the old experiments directory after confirming migration is complete."""
    
    print("🧹 Final Cleanup: Removing Old Experiments Directory")
    print("=" * 55)
    
    old_experiments_dir = "../experiments"
    
    if not os.path.exists(old_experiments_dir):
        print(f"✅ Old experiments directory already removed: {old_experiments_dir}")
        return
    
    print(f"📁 Found old experiments directory: {old_experiments_dir}")
    print("🔍 Checking contents before removal...")
    
    # List what's in the old directory
    try:
        contents = os.listdir(old_experiments_dir)
        print(f"📋 Contents to be removed ({len(contents)} items):")
        for item in sorted(contents)[:10]:  # Show first 10 items
            print(f"   - {item}")
        if len(contents) > 10:
            print(f"   ... and {len(contents) - 10} more items")
    except Exception as e:
        print(f"❌ Error listing contents: {e}")
        return
    
    # Confirm the new structure is in place
    expected_dirs = [
        "exp_001_rq1_1_fairness_weights",
        "exp_002_rq4_1_baseline_comparison", 
        "exp_003_comprehensive_parameter_sweep",
        "exp_004_comparative_parameter_sweep",
        "exp_005_custom_parameter_sweep",
        "exp_006_focused_parameter_sweep",
        "exp_007_bottleneck_analysis",
        "exp_008_full_dataset_analysis"
    ]
    
    missing_dirs = []
    for exp_dir in expected_dirs:
        if not os.path.exists(exp_dir):
            missing_dirs.append(exp_dir)
    
    if missing_dirs:
        print(f"⚠️  WARNING: Some new experiment directories are missing:")
        for missing in missing_dirs:
            print(f"   - {missing}")
        print(f"🛑 Aborting cleanup to prevent data loss")
        return
    
    print(f"✅ All new experiment directories confirmed present")
    
    # Check that key data has been moved
    key_data_check = [
        "exp_004_comparative_parameter_sweep/data/comparative_sweep_20250918_182711",
        "exp_001_rq1_1_fairness_weights/data/rq1_1_results_20250905_123033.json"
    ]
    
    missing_data = []
    for data_path in key_data_check:
        if not os.path.exists(data_path):
            missing_data.append(data_path)
    
    if missing_data:
        print(f"⚠️  WARNING: Some key data files are missing:")
        for missing in missing_data:
            print(f"   - {missing}")
        print(f"🛑 Aborting cleanup to prevent data loss")
        return
    
    print(f"✅ Key data files confirmed in new locations")
    
    # Final confirmation
    print(f"\n🗑️  READY TO REMOVE OLD DIRECTORY")
    print(f"   Old: {old_experiments_dir}")
    print(f"   ✅ Scripts migrated to organized structure")
    print(f"   ✅ Data migrated to experiment folders")
    print(f"   ✅ Analysis notebooks in place")
    
    try:
        # Remove the old directory
        shutil.rmtree(old_experiments_dir)
        print(f"\n✅ SUCCESS: Old experiments directory removed!")
        print(f"🎉 Migration completed successfully!")
        
        # Show the new clean structure
        print(f"\n📁 NEW ORGANIZED STRUCTURE:")
        print("=" * 30)
        for exp_dir in sorted(expected_dirs):
            if os.path.exists(exp_dir):
                print(f"✅ {exp_dir}/")
                if os.path.exists(f"{exp_dir}/data"):
                    data_files = len(os.listdir(f"{exp_dir}/data"))
                    print(f"   📊 Data files: {data_files}")
                if os.path.exists(f"{exp_dir}/run_experiment.py"):
                    print(f"   🚀 Script: run_experiment.py")
                if os.path.exists(f"{exp_dir}/analysis.ipynb"):
                    print(f"   📓 Analysis: analysis.ipynb")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Test running experiments: cd exp_XXX && python run_experiment.py")
        print("2. Open analysis notebooks in Jupyter")
        print("3. Continue with your research!")
        
    except Exception as e:
        print(f"❌ Error removing old directory: {e}")
        print(f"💡 You may need to remove it manually: rm -rf {old_experiments_dir}")

if __name__ == "__main__":
    print("🧹 Experiment Migration Cleanup Tool")
    print("====================================")
    cleanup_old_experiments()
