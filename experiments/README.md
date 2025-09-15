# Spatial Crowdsourcing Experiments

This directory contains comprehensive experiment scripts for analyzing the composite strategy parameters in the spatial crowdsourcing simulator.

## Available Experiments

### 1. Individual Parameter Analysis
- **`run_rq1_1_fairness_weights.py`** - Tests different λ₁ (fairness weight) values
- Explores how fairness weighting affects JFI and task assignment ratio

### 2. Comprehensive Parameter Sweep
- **`run_comprehensive_parameter_sweep.py`** - Multi-dimensional parameter exploration
- Tests all combinations of fairness_weight, starvation_weight, utility_weight, and soft_threshold
- Perfect for overnight data collection on parameter relationships

### 3. Balanced Parameter Search (NEW!)
- **`run_balanced_experiment.py`** - Fairness-focused parameter optimization
- Specifically designed to find configurations that balance speed AND fairness
- Addresses trade-offs identified in initial parameter sweeps

### 4. Baseline Comparison
- **`run_rq4_1_baseline_comparison.py`** - Compares strategies (composite vs greedy vs random)

## Quick Start Guide

### For Comprehensive Parameter Exploration

```bash
# Quick test (30 minutes)
python experiments/run_comprehensive_parameter_sweep.py --mode quick

# Standard exploration (1-2 hours) 
python experiments/run_comprehensive_parameter_sweep.py --mode standard

# Overnight comprehensive sweep (6-8 hours)
python experiments/run_comprehensive_parameter_sweep.py --mode overnight

# Focus on specific objectives
python experiments/run_comprehensive_parameter_sweep.py --mode extensive --focus fairness
python experiments/run_comprehensive_parameter_sweep.py --mode extensive --focus efficiency
python experiments/run_comprehensive_parameter_sweep.py --mode extensive --focus balanced
```

### For Balanced Parameter Search (Fairness-Focused)

```bash
# Find configurations that balance speed AND fairness
# Runs automatically with fairness-optimized parameter ranges
python experiments/run_balanced_experiment.py

# Or run focused sweep directly
python experiments/run_focused_parameter_sweep.py --mode fine
```

### Analyzing Results

```bash
# Analyze parameter sweep results
python experiments/analyze_parameter_sweep.py results/comprehensive_parameter_sweep_20250914_123456.json

# Export to CSV for further analysis
python experiments/analyze_parameter_sweep.py results/comprehensive_parameter_sweep_20250914_123456.json --export-csv analysis_results.csv
```

## Parameter Meanings

The composite strategy uses this scoring function:
**Score = λ₁×Fairness + λ₂×Starvation + λ₃×Utility**

- **λ₁ (Fairness Weight)**: Controls EWMA fairness priority
- **λ₂ (Starvation Weight)**: Controls idle time/starvation prevention priority  
- **λ₃ (Utility Weight)**: Controls distance/efficiency priority
- **soft_threshold**: Minimum score required for immediate task assignment

## Experiment Modes (Research-Scale Datasets) - FIXED & OPTIMIZED!

| Mode | Duration | Combinations | Dataset Size | Purpose |
|------|----------|--------------|--------------|---------|
| `quick` | 12 min | 36 | 5K tasks | Fast research testing |
| `standard` | 1.4 hours | 162 | 15K tasks | Medium research scale |
| `extensive` | 6.7 hours | 400 | 50K tasks | Large research scale |
| `overnight` | 40.8 hours | 1,225 | **FULL DATASET** | Comprehensive research |

**🚀 Performance Optimizations:**
- **FIXED:** Eliminated DataFrame conversion bottleneck (`iterrows()` → `to_dict()`)
- **FASTER:** Conversion now takes seconds instead of hours
- **RELIABLE:** Won't hang overnight anymore
- Uses optimized data loader for your full 10GB dataset
- Intermediate saves prevent data loss

## Focus Areas

- **`all`**: Test all parameter combinations (default)
- **`fairness`**: Focus on fairness-optimized configurations (high λ₁)
- **`efficiency`**: Focus on efficiency-optimized configurations (high λ₃)
- **`balanced`**: Focus on balanced configurations (moderate all λs)

## Output Files

Results are saved to `results/` directory with timestamps:
- **`comprehensive_parameter_sweep_YYYYMMDD_HHMMSS.json`** - Complete results
- Intermediate saves every 50 experiments to prevent data loss

## Analysis Features

The analysis tool provides:
- **Parameter correlations** with performance metrics
- **Optimal configurations** for different objectives
- **Parameter range analysis** for high-performing configs
- **Trade-off identification** between competing objectives
- **Executive summary** with key findings and recommendations

## Example Workflow

1. **Run overnight sweep:**
   ```bash
   python experiments/run_comprehensive_parameter_sweep.py --mode overnight
   ```

2. **Analyze results:**
   ```bash
   python experiments/analyze_parameter_sweep.py results/comprehensive_parameter_sweep_20250914_235959.json
   ```

3. **Export for visualization:**
   ```bash
   python experiments/analyze_parameter_sweep.py results/comprehensive_parameter_sweep_20250914_235959.json --export-csv detailed_analysis.csv
   ```

4. **Create plots in Jupyter:**
   ```bash
   jupyter notebook analysis/Honours_Results_Analysis.ipynb
   ```

## Tips for Long Experiments

- **Use `screen` or `tmux`** for overnight runs:
  ```bash
  screen -S param_sweep
  python experiments/run_comprehensive_parameter_sweep.py --mode overnight
  # Ctrl+A, D to detach
  ```

- **Monitor progress** by checking intermediate saves:
  ```bash
  ls -la results/comprehensive_parameter_sweep_*.json
  ```

- **Resume analysis** anytime - results are saved periodically

## Success Criteria

The experiments use multiple success criteria:
- **Balanced Success**: JFI > 0.85 AND TAR > 95%
- **High JFI**: Jain's Fairness Index > 0.85
- **High TAR**: Task Assignment Ratio > 95%
- **Efficiency**: Average pickup distance < 2.0 km
