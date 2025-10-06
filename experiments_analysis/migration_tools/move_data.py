#!/usr/bin/env python3
"""
Script to move all result data from results/ to their organized experiment folders
"""

import shutil
import os
import glob
from pathlib import Path

def move_data():
    """Move all experiment data to their organized locations."""
    
    print("🚀 Moving experiment data to organized structure...")
    print("=" * 60)
    
    # Define the mapping of result patterns to experiment folders
    data_mapping = {
        # RQ1.1 Fairness Weights - Multiple result files
        "../results/rq1_1_results_*.json": "exp_001_rq1_1_fairness_weights/data/",
        
        # Comprehensive Parameter Sweeps 
        "../results/comprehensive_parameter_sweep_*.json": "exp_003_comprehensive_parameter_sweep/data/",
        
        # Comparative Parameter Sweep - THE BIG ONE
        "../results/comparative_sweep_20250918_182711/": "exp_004_comparative_parameter_sweep/data/comparative_sweep_20250918_182711/",
        
        # Custom Parameter Sweep
        "../results/custom_parameter_sweep_*.json": "exp_005_custom_parameter_sweep/data/",
        
        # Focused Parameter Sweep
        "../results/focused_parameter_sweep_*.json": "exp_006_focused_parameter_sweep/data/",
        
        # Bottleneck Analysis
        "../results/bottleneck_sweep_*.json": "exp_007_bottleneck_analysis/data/",
        "../results/bottleneck_sweep_*.log": "exp_007_bottleneck_analysis/data/",
        
        # Testing data
        "../results/temporal_test_*.csv": "testing/data/",
        "../results/temporal_test_*.json": "testing/data/",
        "../results/test_enhanced_metrics_*.csv": "testing/data/",
        "../results/test_enhanced_metrics_*.json": "testing/data/",
        
        # General analysis files
        "../results/parameter_sweep_results.csv": "shared_data/parameter_sweep_results.csv",
        "../results/fairness_metrics_analysis.csv": "shared_data/fairness_metrics_analysis.csv",
        "../results/processed_comparative_analysis.csv": "shared_data/processed_comparative_analysis.csv",
    }
    
    moved_count = 0
    failed_count = 0
    
    # Create necessary directories
    for dir_name in ['testing/data', 'shared_data']:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}/")
    
    for pattern, destination in data_mapping.items():
        try:
            # Handle directory moves differently than file patterns
            if pattern.endswith('/'):
                # This is a directory move
                source_dir = pattern.rstrip('/')
                if os.path.exists(source_dir):
                    # Create destination directory
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    
                    # Move the entire directory
                    shutil.move(source_dir, destination)
                    print(f"✅ Moved directory: {source_dir} → {destination}")
                    moved_count += 1
                else:
                    print(f"⚠️  Directory {source_dir} not found, skipping...")
            else:
                # This is a file pattern
                matching_files = glob.glob(pattern)
                
                if not matching_files:
                    print(f"⚠️  No files found matching {pattern}, skipping...")
                    continue
                
                # Create destination directory
                dest_dir = os.path.dirname(destination)
                os.makedirs(dest_dir, exist_ok=True)
                
                for file_path in matching_files:
                    try:
                        filename = os.path.basename(file_path)
                        dest_path = os.path.join(dest_dir, filename)
                        
                        shutil.move(file_path, dest_path)
                        print(f"✅ Moved: {file_path} → {dest_path}")
                        moved_count += 1
                        
                    except Exception as e:
                        print(f"❌ Failed to move {file_path}: {e}")
                        failed_count += 1
            
        except Exception as e:
            print(f"❌ Failed to process pattern {pattern}: {e}")
            failed_count += 1
    
    print(f"\n📊 DATA MIGRATION SUMMARY:")
    print("=" * 35)
    print(f"✅ Successfully moved: {moved_count} files/directories")
    print(f"❌ Failed: {failed_count} files/directories")
    
    # Create README files in data directories
    create_data_readmes()
    
    return moved_count, failed_count

def create_data_readmes():
    """Create README files in data directories to explain what's there."""
    
    readme_content = {
        "exp_001_rq1_1_fairness_weights/data/README.md": """# RQ1.1 Fairness Weights Data

This directory contains results from the RQ1.1 fairness weight analysis experiments.

## Files:
- `rq1_1_results_*.json`: Results from different experimental runs
- Each file contains JFI, TAR, wait times, and parameter configurations

## Analysis:
- Use the analysis notebook to explore optimal λ₁ values
- Look for configurations with TAR >95% and high JFI
""",
        
        "exp_004_comparative_parameter_sweep/data/README.md": """# Comparative Parameter Sweep Data

This directory contains the MOST COMPREHENSIVE experimental data.

## Structure:
- `comparative_sweep_20250918_182711/`: Main experiment directory
  - `temporal_data/`: Individual experiment temporal evolution data
    - `exp_XX_Strategy_Run_Y_summary.json`: Summary metrics for each experiment
    - `exp_XX_Strategy_Run_Y_*.csv`: Detailed temporal data (wait times, worker fairness, etc.)

## Key Data:
- 36 experiments: 6 Greedy baseline + 30 Composite parameter combinations
- Enhanced metrics: Supervisor's UD/FL, IOR, EWMA evolution
- 15K workers, 20K tasks dataset
- Full temporal evolution tracking

## Usage:
- Primary data source for the comprehensive analysis notebook
- Contains data for RQ1, RQ2, RQ4, RQ5, RQ10-11 analysis
""",
        
        "testing/data/README.md": """# Testing Data

This directory contains data from various testing and validation experiments.

## Files:
- `temporal_test_*`: Temporal evolution testing data
- `test_enhanced_metrics_*`: Enhanced metrics validation data

## Purpose:
- Validation of metric collection systems
- Testing of temporal data collection
- Verification of enhanced fairness metrics
""",
        
        "shared_data/README.md": """# Shared Analysis Data

This directory contains processed and aggregated data used across multiple experiments.

## Files:
- `parameter_sweep_results.csv`: Aggregated parameter sweep results
- `fairness_metrics_analysis.csv`: Fairness metrics analysis
- `processed_comparative_analysis.csv`: Processed comparative analysis data

## Usage:
- Cross-experiment analysis
- Meta-analysis of parameter effects
- Comparative studies across different experimental setups
"""
    }
    
    for filepath, content in readme_content.items():
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"📝 Created README: {filepath}")
        except Exception as e:
            print(f"❌ Failed to create README {filepath}: {e}")

if __name__ == "__main__":
    print("📁 Experiment Data Migration Tool")
    print("=================================")
    
    moved, failed = move_data()
    
    if moved > 0:
        print(f"\n🎉 Data migration completed!")
        print(f"📊 Your experiment data is now properly organized!")
        print(f"\n🔍 Check each exp_XXX/data/ directory for your results")
        print(f"📚 README files created to explain data structure")
    else:
        print(f"\n⚠️  No files were moved. Check the original paths.")
