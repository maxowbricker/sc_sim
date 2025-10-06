#!/usr/bin/env python3
"""
Script to move all original experiment scripts from experiments/ to their organized locations
"""

import shutil
import os
from pathlib import Path

def move_scripts():
    """Move all experiment scripts to their organized locations."""
    
    print("🚀 Moving experiment scripts to organized structure...")
    print("=" * 60)
    
    # Define the mapping of original scripts to new locations
    script_mapping = {
        # Original script -> New location
        "../experiments/run_rq1_1_fairness_weights.py": "exp_001_rq1_1_fairness_weights/run_experiment.py",
        "../experiments/run_rq4_1_baseline_comparison.py": "exp_002_rq4_1_baseline_comparison/run_experiment.py", 
        "../experiments/run_comprehensive_parameter_sweep.py": "exp_003_comprehensive_parameter_sweep/run_experiment.py",
        "../experiments/run_comparative_parameter_sweep.py": "exp_004_comparative_parameter_sweep/run_experiment.py",
        "../experiments/run_custom_parameter_sweep.py": "exp_005_custom_parameter_sweep/run_experiment.py",
        "../experiments/run_focused_parameter_sweep.py": "exp_006_focused_parameter_sweep/run_experiment.py",
        "../experiments/run_bottleneck_parameter_sweep.py": "exp_007_bottleneck_analysis/run_experiment.py",
        "../experiments/run_full_dataset_analysis.py": "exp_008_full_dataset_analysis/run_experiment.py",
        
        # Utility scripts
        "../experiments/json_utils.py": "shared_utils/json_utils.py",
        "../experiments/analyze_parameter_sweep.py": "shared_utils/analyze_parameter_sweep.py",
        "../experiments/analyze_results.py": "shared_utils/analyze_results.py",
        
        # Testing scripts
        "../experiments/test_fixed_loader.py": "testing/test_fixed_loader.py",
        "../experiments/timing_test.py": "testing/timing_test.py",
        "../experiments/simple_timing_test.py": "testing/simple_timing_test.py",
        "../experiments/operational_test.py": "testing/operational_test.py",
        
        # Additional experiments
        "../experiments/run_balanced_experiment.py": "additional/run_balanced_experiment.py",
        "../experiments/run_full_dataset_experiments.py": "additional/run_full_dataset_experiments.py",
    }
    
    moved_count = 0
    failed_count = 0
    
    for original_path, new_path in script_mapping.items():
        try:
            # Check if original file exists
            if not os.path.exists(original_path):
                print(f"⚠️  {original_path} not found, skipping...")
                continue
            
            # Create directory if it doesn't exist
            new_dir = os.path.dirname(new_path)
            os.makedirs(new_dir, exist_ok=True)
            
            # Update the script paths for the new location
            with open(original_path, 'r') as f:
                content = f.read()
            
            # Update path references
            updated_content = content.replace(
                'project_root = Path(__file__).parent.parent',
                'project_root = Path(__file__).parent.parent.parent'
            )
            
            # Update result paths to go to the main results directory
            updated_content = updated_content.replace(
                'results/',
                '../../../results/'
            )
            
            # Write to new location
            with open(new_path, 'w') as f:
                f.write(updated_content)
            
            print(f"✅ Moved: {original_path} → {new_path}")
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Failed to move {original_path}: {e}")
            failed_count += 1
    
    print(f"\n📊 MIGRATION SUMMARY:")
    print("=" * 30)
    print(f"✅ Successfully moved: {moved_count} files")
    print(f"❌ Failed: {failed_count} files")
    
    if moved_count > 0:
        print(f"\n🎯 Next steps:")
        print("1. Test the moved scripts: cd exp_XXX && python run_experiment.py")
        print("2. Remove the original experiments/ directory if everything works")
        print("3. Update any remaining path references if needed")
    
    return moved_count, failed_count

if __name__ == "__main__":
    print("📁 Experiment Script Migration Tool")
    print("===================================")
    
    # Create utility directories
    for dir_name in ['shared_utils', 'testing', 'additional']:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}/")
    
    moved, failed = move_scripts()
    
    if moved > 0:
        print(f"\n🎉 Migration completed!")
        print(f"📊 Your experiment scripts are now properly organized!")
    else:
        print(f"\n⚠️  No files were moved. Check the original paths.")
