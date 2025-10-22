# Combined Experiments 009 + 010

**Status**: ✅ **COMPLETE**  
**Combined Date**: October 21, 2025  
**Total Experiments**: 42 (21 from each source)

---

## 📋 Overview

This directory contains the **combined results** from two complementary parameter sweep experiments:

- **Experiment 009**: Comprehensive Parameter Sweep (Post-Normalization)
- **Experiment 010**: Extended Boundaries - Pareto High-Resolution

By combining these datasets, we have a **unified view** of 42 experiments exploring the fairness-efficiency trade-off space with `normalize_scores=True`.

---

## 🎯 Why Combine These Experiments?

### **Experiment 009** (21 experiments)
- **8 groups** (A-H): Baseline, Grid, Ablation, Threshold, Balanced, High/Low edges
- Broad exploration of λ₁ (0.0-5.0), λ₂ (0.0-2.0), λ₃ (0.1-2.0), θ (0.1-0.9)
- Fixed: `normalize_scores=True`, `gamma=0.5`, `enable_diagnostics=False`

### **Experiment 010** (21 experiments)
- **High-resolution Pareto frontier mapping**
- Focused grid: λ₁ ∈ [2.5, 3.0, 3.5, 4.0, 4.5] × λ₃ ∈ [0.5, 1.0, 1.5, 2.0]
- Fixed: λ₂=0.5, θ=0.5, same normalization settings as 009
- **Purpose**: Fill the critical gap in the fairness-utility trade-off space

### **Combined Dataset**
- **No redundancy**: The two experiments explored different regions
- **Complementary coverage**: 009 provides breadth, 010 provides depth in the sweet spot
- **Unified analysis**: Enables comprehensive Pareto frontier analysis

---

## 📁 Data Files

### Main Dataset
```
data/experiment_009+010_combined_results.csv
```
- **42 rows** (1 Greedy baseline + 41 Composite configurations)
- **24 columns** including:
  - `experiment_id`: Sequential 1-42
  - `source_experiment`: Origin identifier (`exp_009` or `exp_010`)
  - `original_experiment_id`: Original ID from source experiment
  - Configuration: `fairness_weight`, `starvation_weight`, `utility_weight`, `soft_threshold`
  - Metrics: `jains_fairness_index`, `mean_task_wait_time_min`, `task_assignment_ratio`, etc.

### Metadata
```
data/COMBINED_DATASET_INFO.txt
```
- Generation timestamp
- Source file paths
- Summary statistics
- Combination notes

---

## 📊 Dataset Summary

### Experiments by Source
- **Exp 009**: 21 experiments
- **Exp 010**: 21 experiments
- **Total**: 42 experiments

### Strategy Distribution
- **Greedy Baseline**: 1 experiment
- **Composite Strategy**: 41 experiments

### Parameter Ranges (Composite Only)

| Parameter | Symbol | Min | Max | Description |
|-----------|--------|-----|-----|-------------|
| **Fairness Weight** | λ₁ | 0.1 | 5.0 | EWMA fairness component |
| **Starvation Weight** | λ₂ | 0.5 | 2.0 | Max idle time penalty |
| **Utility Weight** | λ₃ | 0.1 | 2.0 | Pickup distance optimization |
| **Soft Threshold** | θ | 0.5 | 0.5 | Assignment quality filter (fixed in most experiments) |

### Metric Ranges (Composite Only)

| Metric | Min | Max | Description |
|--------|-----|-----|-------------|
| **Jain's Fairness Index (JFI)** | 0.2634 | 0.2953 | Worker equity (higher = more fair) |
| **Mean Task Wait Time** | 2.56 min | 3.09 min | Average time tasks wait for assignment |
| **Task Assignment Ratio (TAR)** | 86.22% | 86.26% | % of tasks successfully assigned |

---

## 🔍 How to Use This Dataset

### Quick Start

```python
import pandas as pd

# Load combined dataset
df = pd.read_csv('data/experiment_009+010_combined_results.csv')

# Filter to composite strategies only
df_composite = df[df['strategy'] == 'composite']

# Top 5 by fairness
top_fairness = df_composite.nlargest(5, 'jains_fairness_index')
print(top_fairness[['name', 'fairness_weight', 'utility_weight', 'jains_fairness_index']])

# Top 5 by efficiency
top_efficiency = df_composite.nsmallest(5, 'mean_task_wait_time_min')
print(top_efficiency[['name', 'fairness_weight', 'utility_weight', 'mean_task_wait_time_min']])

# Pareto frontier analysis
import matplotlib.pyplot as plt
plt.scatter(df_composite['mean_task_wait_time_min'], 
            df_composite['jains_fairness_index'],
            c=df_composite['fairness_weight'],
            cmap='viridis')
plt.xlabel('Mean Task Wait Time (min)')
plt.ylabel('Jain\'s Fairness Index')
plt.colorbar(label='λ₁ (Fairness Weight)')
plt.show()
```

### Filtering by Source

```python
# Analyze Exp 009 results only
df_009 = df[df['source_experiment'] == 'exp_009']

# Analyze Exp 010 results only
df_010 = df[df['source_experiment'] == 'exp_010']

# Compare coverage
print(f"Exp 009 λ₁ range: {df_009['fairness_weight'].min():.1f} - {df_009['fairness_weight'].max():.1f}")
print(f"Exp 010 λ₁ range: {df_010['fairness_weight'].min():.1f} - {df_010['fairness_weight'].max():.1f}")
```

---

## 🚀 Recommended Analyses

### 1. **Pareto Frontier Analysis**
- Plot JFI vs. Wait Time with λ₁ as color/size
- Identify the "knee" of the curve (optimal trade-off)
- Compare to Greedy baseline

### 2. **Parameter Space Heatmap**
- Create λ₁ × λ₃ heatmap for JFI
- Create λ₁ × λ₃ heatmap for Wait Time
- Identify regions of high fairness with acceptable efficiency

### 3. **Configuration Comparison**
- Spider/radar charts for top 5 configs by different objectives
- Multi-objective optimization ranking (weighted sum of normalized metrics)

### 4. **Statistical Analysis**
- Correlation matrix between parameters and metrics
- Regression analysis: predict JFI from λ₁, λ₂, λ₃
- Sensitivity analysis: which parameter has the strongest effect?

### 5. **Source Comparison**
- Compare metric distributions between Exp 009 and 010
- Validate consistency where experiments overlap
- Assess whether high-resolution sweep (010) revealed new insights

---

## 🔗 Related Files

### Source Experiments
- **Experiment 009**: `../exp_009_comprehensive_parameter_sweep/`
  - Original data: `data/exp_009_20251019_232730/`
  - Analysis: `analysis.ipynb`
  - Documentation: `setup.md`, `README.md`

- **Experiment 010**: `../exp_010_extended_boundaries/`
  - Original data: `data/exp_010_20251021_000315/`
  - Documentation: `setup.md`, `README.md`, `QUICK_START.md`

### Combination Script
- `combine_experiments.py`: Reproducible script for merging datasets

---

## ⚠️ Important Notes

1. **Original data preserved**: Source CSV files in Exp 009 and 010 remain unchanged
2. **Experiment IDs renumbered**: The combined dataset uses sequential IDs (1-42)
   - Use `original_experiment_id` to reference source experiments
   - Use `source_experiment` to identify origin
3. **Column differences**: Some columns are only present in one source:
   - `mean_worker_idle_time_min`: Only in Exp 009
   - `empty_km_ratio`, `ewma_cv`, `max_wait_time`: Only in Exp 010
   - Missing values are represented as `NaN`
4. **Fixed parameters**: All experiments use `normalize_scores=True`, `gamma=0.5`

---

## 📈 Key Findings Preview

*(To be completed after comprehensive analysis)*

### Preliminary Observations

1. **Fairness range is narrow** (JFI: 0.263-0.295)
   - All composite strategies achieve similar fairness
   - Greedy baseline significantly worse (expected)

2. **Wait time variation is modest** (2.56-3.09 min)
   - ~20% range across all configurations
   - Suggests robustness of normalized composite strategy

3. **TAR is remarkably stable** (86.22-86.26%)
   - Only 0.04 percentage point variation
   - System load and capacity are well-matched

4. **Parameter exploration is comprehensive**:
   - λ₁: Wide range (0.1-5.0) with high-resolution in critical zone (2.5-4.5)
   - λ₃: Good coverage (0.1-2.0) with focus on practical range (0.5-2.0)
   - λ₂: Moderate coverage (0.5-2.0)

---

## 🎯 Next Steps

1. **Create unified analysis notebook** combining insights from both experiments
2. **Identify optimal configuration(s)** based on multi-objective criteria
3. **Validate findings** with additional replications (if needed)
4. **Document recommendations** for production deployment

---

**Generated**: October 21, 2025  
**Script**: `combine_experiments.py`  
**Original Experiments**: exp_009 (Oct 19-20), exp_010 (Oct 20-21)




