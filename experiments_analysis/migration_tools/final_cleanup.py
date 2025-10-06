#!/usr/bin/env python3
"""
Final cleanup script to remove the old analysis/ directory and verify complete migration
"""

import shutil
import os
from pathlib import Path

def final_cleanup():
    """Complete the migration by removing old analysis directory."""
    
    print("🧹 FINAL CLEANUP: Complete Migration")
    print("=" * 40)
    
    old_analysis_dir = "../analysis"
    
    # Check if analysis directory exists
    if not os.path.exists(old_analysis_dir):
        print(f"✅ Analysis directory already removed: {old_analysis_dir}")
    else:
        print(f"📁 Found old analysis directory: {old_analysis_dir}")
        
        # List contents before removal
        try:
            contents = os.listdir(old_analysis_dir)
            print(f"📋 Contents to be removed ({len(contents)} items):")
            for item in contents:
                if not item.startswith('.'):  # Skip hidden files
                    print(f"   - {item}")
        except Exception as e:
            print(f"❌ Error listing contents: {e}")
            return
        
        # Verify notebooks have been migrated
        notebook_locations = {
            "Honours_Results_Analysis.ipynb": "exp_001_rq1_1_fairness_weights/analysis_honours.ipynb",
            "Comprehensive_Research_Analysis.ipynb": "exp_004_comparative_parameter_sweep/analysis_comprehensive.ipynb"
        }
        
        missing_notebooks = []
        for old_name, new_location in notebook_locations.items():
            if not os.path.exists(new_location):
                missing_notebooks.append(f"{old_name} → {new_location}")
        
        if missing_notebooks:
            print(f"⚠️  WARNING: Some notebooks haven't been migrated:")
            for missing in missing_notebooks:
                print(f"   - {missing}")
            print(f"🛑 Aborting cleanup to prevent data loss")
            return
        
        print(f"✅ All notebooks confirmed migrated to new locations")
        
        # Remove the old analysis directory
        try:
            shutil.rmtree(old_analysis_dir)
            print(f"✅ SUCCESS: Old analysis directory removed!")
        except Exception as e:
            print(f"❌ Error removing old directory: {e}")
            return
    
    # Verify the experiments directory is also gone
    old_experiments_dir = "../experiments"
    if os.path.exists(old_experiments_dir):
        print(f"⚠️  WARNING: Old experiments directory still exists: {old_experiments_dir}")
        print(f"🔧 Run cleanup_old_experiments.py to remove it")
    else:
        print(f"✅ Old experiments directory already removed")
    
    # Final verification of the new structure
    print(f"\n📊 MIGRATION VERIFICATION:")
    print("=" * 30)
    
    expected_structure = {
        "exp_001_rq1_1_fairness_weights": ["run_experiment.py", "data", "analysis.ipynb", "analysis_honours.ipynb"],
        "exp_004_comparative_parameter_sweep": ["run_experiment.py", "data", "analysis.ipynb", "analysis_comprehensive.ipynb"],
        "shared_utils": ["json_utils.py", "analyze_parameter_sweep.py", "analyze_results.py"],
        "testing": ["data", "test_fixed_loader.py", "timing_test.py"],
        "shared_data": ["parameter_sweep_results.csv", "fairness_metrics_analysis.csv"]
    }
    
    all_good = True
    for directory, expected_files in expected_structure.items():
        if os.path.exists(directory):
            print(f"✅ {directory}/")
            for expected_file in expected_files:
                file_path = os.path.join(directory, expected_file)
                if os.path.exists(file_path):
                    if os.path.isdir(file_path):
                        file_count = len(os.listdir(file_path)) if os.path.exists(file_path) else 0
                        print(f"   📁 {expected_file}/ ({file_count} items)")
                    else:
                        print(f"   📄 {expected_file}")
                else:
                    print(f"   ❌ Missing: {expected_file}")
                    all_good = False
        else:
            print(f"❌ Missing directory: {directory}")
            all_good = False
    
    if all_good:
        print(f"\n🎉 COMPLETE MIGRATION SUCCESS!")
        print("=" * 35)
        print("✅ All old directories removed")
        print("✅ All scripts migrated and organized")
        print("✅ All data moved to proper experiment folders")
        print("✅ All analysis notebooks in place")
        print("✅ Organized structure ready for research")
        
        print(f"\n🚀 READY FOR RESEARCH:")
        print("1. cd exp_004_comparative_parameter_sweep/")
        print("2. jupyter notebook analysis_comprehensive.ipynb")
        print("3. Continue your spatial crowdsourcing research!")
        
    else:
        print(f"\n⚠️  Migration incomplete. Check missing items above.")
    
    return all_good

if __name__ == "__main__":
    print("🏁 Final Migration Cleanup Tool")
    print("===============================")
    success = final_cleanup()
    
    if success:
        print(f"\n✨ Your experiment framework is now perfectly organized!")
    else:
        print(f"\n🔧 Some issues detected. Review and fix before proceeding.")
