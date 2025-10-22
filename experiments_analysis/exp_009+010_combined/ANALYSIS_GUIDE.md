# Combined Experiments 009+010: Analysis Guide

**Created**: October 21, 2025  
**Notebook**: `analysis_combined.ipynb`  
**Total Cells**: 38  
**Data**: 42 experiments (21 from each source)

---

## 📋 What's Included

This comprehensive analysis notebook implements plots and analyses from the original `ANALYSIS_PLAN.md`, adapted for the combined 009+010 dataset.

### Analysis Sections

1. **Setup & Data Loading** (4 cells)
   - Import libraries
   - Load combined dataset
   - Data preview and validation

2. **Summary Statistics** (2 cells)
   - Group-level statistics by source
   - Overall composite performance metrics

3. **Key Profile Identification** (1 cell)
   - Greedy Baseline
   - Top Fairness configuration
   - Top Efficiency configuration
   - Sweet Spot (balanced) configuration

4. **Trade-off & Pareto Analysis** (7 cells)
   - **Plot 2**: Pareto Frontier (JFI vs. Wait Time)
   - **Plot 6**: Efficiency Frontier (Distance vs. Wait Time)
   - Correlation analyses
   - Trade-off quantification

5. **Parameter Space Exploration** (4 cells)
   - **Plot 1**: Parameter Heatmap (λ₁ vs. λ₃) using Exp 010 grid
   - High-resolution parameter mapping
   - Optimal parameter region identification

6. **Multi-Metric Configuration Comparison** (3 cells)
   - **Plot 3**: Radar Chart (multi-metric profiles)
   - Normalized metric comparison
   - Holistic performance evaluation

7. **Advanced Statistical Analysis** (5 cells)
   - **Plot 15**: Correlation Matrix (all metrics)
   - **Plot 16**: PCA Biplot (dimensionality reduction)
   - Metric interdependencies
   - Latent performance factors

8. **Golden Nugget Questions** (4 cells)
   - **Q1**: Optimal λ₁/λ₃ ratio?
   - **Q2**: Minimum λ₁ to beat Greedy?
   - **Q3**: Fairness-efficiency trade-off slope?
   - **Q4**: Worker idle time paradox status?

9. **Final Summary & Recommendations** (2 cells)
   - Top 5 by fairness
   - Top 5 by efficiency
   - Top 3 balanced configurations
   - Production recommendations

10. **Key Findings Section** (1 cell)
    - Summary of major findings
    - Actionable recommendations
    - Next steps

---

## 🚀 How to Run the Analysis

### Quick Start

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_009+010_combined
source ../../venv/bin/activate
jupyter notebook analysis_combined.ipynb
```

### Running All Cells

1. Open the notebook in Jupyter
2. Click **Kernel** → **Restart & Run All**
3. Wait for all cells to execute (~2-3 minutes)
4. All plots will be saved to `figures/` directory

### Running Specific Sections

Navigate to the section you want and run cells individually using:
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `Alt + Enter`: Run cell and insert new cell below

---

## 📊 Expected Outputs

### Generated Figures (in `figures/` directory)

1. `plot_02_pareto_frontier.png` - Fairness vs. efficiency trade-off
2. `plot_06_efficiency_frontier.png` - Distance vs. wait time
3. `plot_01_parameter_heatmap.png` - λ₁ × λ₃ parameter space
4. `plot_03_radar_chart.png` - Multi-metric configuration profiles
5. `plot_15_correlation_matrix.png` - Metric correlations
6. `plot_16_pca_biplot.png` - Principal component analysis
7. `golden_nugget_1_ratio.png` - Parameter ratio analysis

### Key Outputs in Notebook

- **Pareto-efficient configurations**: List of non-dominated configs
- **Correlation coefficients**: Metric relationships
- **Statistical summaries**: Mean, std, min, max for all metrics
- **Top performer tables**: Best configs by different objectives
- **Regression analyses**: Quantified trade-offs

---

## 🔍 What to Look For

### 1. Pareto Frontier Analysis (Section 4)
- **Question**: Where is the "knee" of the Pareto curve?
- **Look for**: Configurations where fairness gains level off vs. wait time cost
- **Key insight**: Optimal balance point between competing objectives

### 2. Parameter Heatmaps (Section 5)
- **Question**: Are there "sweet spot" regions in parameter space?
- **Look for**: Dark green regions (high JFI) with acceptable wait times
- **Key insight**: Practical parameter ranges for production

### 3. Correlation Matrix (Section 7)
- **Question**: Which metrics are redundant or surprising?
- **Look for**: Strong correlations (|r| > 0.7) or unexpected patterns
- **Key insight**: Metric selection for future experiments

### 4. Golden Nugget 1 (Section 8)
- **Question**: Is there a magic λ₁/λ₃ ratio?
- **Look for**: Clusters or trends in the ratio vs. JFI plot
- **Key insight**: Simplified parameter selection rule

### 5. Golden Nugget 3 (Section 8)
- **Question**: What's the exact fairness-efficiency trade-off?
- **Look for**: Slope value (minutes per JFI unit)
- **Key insight**: Quantify the "cost" of fairness

---

## 💡 Tips for Interpretation

### Fairness (JFI)
- **Range**: 0.263 - 0.295 in this dataset
- **Interpretation**: Higher = more equitable task distribution
- **Context**: Narrow range suggests robust fairness across configs

### Wait Time
- **Range**: ~2.5 - 3.1 minutes
- **Interpretation**: Lower = faster response
- **Context**: ~20% variation across all configurations

### Task Assignment Ratio (TAR)
- **Range**: ~86.2% - 86.3%
- **Interpretation**: % of tasks successfully assigned
- **Context**: Remarkably stable (only 0.1% variation!)

### Pareto Efficiency
- **Dominated configs**: Can improve both JFI and wait time
- **Pareto configs**: Cannot improve one without sacrificing the other
- **Focus on**: Pareto-efficient configs only for production

---

## 🎯 Filling in the Findings

As you run the notebook, **fill in the "Key Takeaways" sections** after each major plot with:

1. **Observed Pattern**: What does the plot show?
2. **Statistical Evidence**: Relevant correlation, p-value, or R²
3. **Practical Implication**: What does this mean for system design?

Example:
```markdown
### Key Takeaways - Plot 2

**Observed Pattern**: 
- Pareto frontier shows clear trade-off: JFI 0.29+ requires wait time > 2.9 min
- 12 configurations are Pareto-efficient

**Statistical Evidence**:
- Correlation (JFI vs Wait): r = 0.65, p < 0.001

**Practical Implication**:
- For production: Choose from Pareto set based on fairness priority
- Recommend: λ₁=2.5, λ₃=0.5 for best fairness with acceptable wait
```

---

## 🔄 Regenerating the Notebook

If you need to regenerate or modify the notebook:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_009+010_combined
python generate_comprehensive_notebook.py
```

This will overwrite `analysis_combined.ipynb`. To keep the old version:

```bash
cp analysis_combined.ipynb analysis_combined_backup.ipynb
python generate_comprehensive_notebook.py
```

---

## 📚 Related Files

- **Data Source**: `data/experiment_009+010_combined_results.csv`
- **Combination Script**: `combine_experiments.py`
- **Generation Script**: `generate_comprehensive_notebook.py`
- **Main README**: `README.md`
- **Original Plan**: `../exp_009_comprehensive_parameter_sweep/ANALYSIS_PLAN.md`

---

## ⚠️ Known Limitations

1. **Temporal Data**: Plots A, B, D from ANALYSIS_PLAN.md require event-level temporal data not collected
2. **Worker-Level Data**: Plot C (idle time distribution) requires per-worker data not available in combined dataset
3. **Threshold Variation**: Limited threshold variation (most experiments use θ=0.5)
4. **Single Runs**: No replications, so no statistical error bars

---

## 🎓 Next Steps After Analysis

1. **Document Findings**: Update README.md with key results
2. **Select Production Config**: Choose optimal config based on findings
3. **Validate**: Run replications (3-5 runs) of top config
4. **Publish**: Use generated figures in thesis/paper
5. **Future Work**: Design Experiment 011 based on insights

---

**Happy Analyzing!** 📊✨

For questions or issues, refer to the original ANALYSIS_PLAN.md or check the inline comments in the notebook cells.




