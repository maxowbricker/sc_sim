# Experiment 014: EWMA & Trade-Off Exploration Analysis Plan
## Validating EWMA and Mapping the Fairness-Efficiency Frontier

**Last Updated**: October 23, 2025  
**Status**: Ready for Analysis  
**Data**: 43 experiments (3 Baselines, 25 Pareto Sweep, 15 Gamma Sensitivity)

---

## 🎯 Primary Goals

1. **Map the Fairness-Efficiency Trade-off** (RQ1): Find optimal λ₁ vs λ₃ balance
2. **Validate EWMA Metric** (RQ2.1): Compare EWMA-based fairness to traditional metrics
3. **Test Gamma Sensitivity** (RQ2.2): Determine if γ parameter matters
4. **Benchmark Composite Strategy**: Compare to Greedy, LAF, and EWMA-Only baselines

**Context**: 4K workers / 20K tasks / 15-min expiry (validated from Exp 012)

---

## 📊 Experiment Summary

### Group 1: Baselines (3 experiments)
| Strategy | Description | Expected Behavior |
|----------|-------------|-------------------|
| **Greedy** | Proximity-only | High efficiency, low fairness |
| **LAF** | Least Allocated First | High fairness, low efficiency |
| **EWMA-Only** | EWMA fairness (γ=0.5) | Middle ground? |

### Group 2: Pareto Sweep (25 experiments)
- **λ₁ (Fairness)**: [2.5, 3.0, 3.5, 4.0, 4.5]
- **λ₃ (Utility)**: [0.5, 1.0, 1.5, 2.0, 2.5]
- **Fixed**: λ₂=0.5, θ=0.0

### Group 3: Gamma Sensitivity (15 experiments)
- **γ values**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Configurations**:
  - Balanced: λ₁=3.5, λ₃=1.0
  - HighFairness: λ₁=4.5, λ₃=0.5
  - Efficiency: λ₁=2.5, λ₃=2.0

---

## 📈 Analysis Sections & Plots

### **SECTION 1: Executive Summary & Baseline Comparison** ⭐
*Setting the stage: What are we comparing against?*

#### **Plot 1: Baseline Performance Overview (Grouped Bar Chart)**
**Type**: Grouped bar chart (3 strategies × 4 metrics)  
**Purpose**: Quick visual comparison of all baselines

**Implementation**:
- X-axis: 4 key metrics (TAR, JFI, Wait Time, % Zero-Task Workers)
- Y-axis: Metric value
- Bars: Grouped by strategy (Greedy, LAF, EWMA-Only)
- Annotations: Numeric values on bars

**Expected Insight**:
- Greedy: Highest efficiency, lowest fairness
- LAF: Highest fairness, highest wait times
- EWMA-Only: Where does it land?

---

#### **Plot 2: Wait Time vs Fairness Trade-off (Baseline Scatter)**
**Type**: Scatter plot with annotations  
**Purpose**: Visualize the fundamental trade-off

**Implementation**:
- X-axis: Jain's Fairness Index (higher is better)
- Y-axis: Mean Wait Time in minutes (lower is better)
- Points: 3 baselines (large markers, different colors)
- Annotations: Strategy names
- Quadrants: Mark "ideal" (top-left) and "poor" (bottom-right)

**Expected Insight**:
- Confirm: Fairness and efficiency are in tension
- Question: Can composite strategy reach the "Pareto frontier"?

---

### **SECTION 2: Pareto Frontier - The λ₁ vs λ₃ Trade-off** 🎯
*Core RQ1: Finding the optimal balance*

#### **Plot 3: Pareto Heatmap - JFI across λ₁ and λ₃**
**Type**: 2D heatmap with annotations  
**Purpose**: Visual map of fairness across parameter space

**Implementation**:
- X-axis: λ₃ (Utility weight) [0.5, 1.0, 1.5, 2.0, 2.5]
- Y-axis: λ₁ (Fairness weight) [2.5, 3.0, 3.5, 4.0, 4.5]
- Color: JFI value (dark = higher fairness)
- Annotations: Numeric JFI values in each cell
- Mark: Best fairness configuration

**Expected Insight**:
- Does JFI increase monotonically with λ₁?
- What is the optimal λ₁/λ₃ ratio for fairness?

---

#### **Plot 4: Pareto Heatmap - Wait Time across λ₁ and λ₃**
**Type**: 2D heatmap with annotations  
**Purpose**: Visual map of efficiency across parameter space

**Implementation**:
- X-axis: λ₃ (Utility weight) [0.5, 1.0, 1.5, 2.0, 2.5]
- Y-axis: λ₁ (Fairness weight) [2.5, 3.0, 3.5, 4.0, 4.5]
- Color: Mean Wait Time (light = lower wait, better)
- Annotations: Numeric wait time values
- Mark: Best efficiency configuration

**Expected Insight**:
- Does wait time decrease with higher λ₃?
- Is there a "sweet spot" that balances both metrics?

---

#### **Plot 5: Pareto Frontier Curve**
**Type**: Scatter plot with Pareto frontier line  
**Purpose**: Identify non-dominated solutions

**Implementation**:
- X-axis: Jain's Fairness Index (JFI)
- Y-axis: Mean Wait Time (inverted: lower is better)
- Points: All 25 Pareto sweep runs
- Color: Mapped to λ₁/λ₃ ratio
- Size: Proportional to TAR
- Line: Connect Pareto-optimal points
- Overlay: 3 baseline strategies for comparison

**Expected Insight**:
- Which configurations are Pareto-optimal?
- Does composite strategy dominate all baselines?
- What is the "knee" of the curve?

---

#### **Plot 6: Multi-Objective Performance (Parallel Coordinates)**
**Type**: Parallel coordinates plot  
**Purpose**: Show trade-offs across multiple dimensions

**Implementation**:
- Axes (6 total): TAR, JFI, Gini, Wait Time, Utilization, % Zero Workers
- Lines: Each of the 25 Pareto sweep runs
- Color: Gradient from high-fairness (λ₁=4.5) to high-efficiency (λ₃=2.5)
- Highlight: Top 3 "balanced" configurations

**Expected Insight**:
- Which configurations excel across all metrics?
- Are there clear trade-offs or compromise solutions?

---

### **SECTION 3: Weight Sensitivity Analysis**
*Understanding the impact of λ₁ and λ₃*

#### **Plot 7: Impact of λ₁ (Fairness Weight) - Line Plots**
**Type**: Multi-line plot (one line per λ₃ value)  
**Purpose**: Isolate the effect of fairness weight

**Implementation**:
- X-axis: λ₁ values [2.5, 3.0, 3.5, 4.0, 4.5]
- Y-axis: JFI
- Lines: One for each λ₃ [0.5, 1.0, 1.5, 2.0, 2.5]
- Legend: λ₃ values

**Expected Insight**:
- Does JFI increase monotonically with λ₁?
- Is the effect consistent across different λ₃ values?

---

#### **Plot 8: Impact of λ₃ (Utility Weight) - Line Plots**
**Type**: Multi-line plot (one line per λ₁ value)  
**Purpose**: Isolate the effect of utility weight

**Implementation**:
- X-axis: λ₃ values [0.5, 1.0, 1.5, 2.0, 2.5]
- Y-axis: Mean Wait Time (minutes)
- Lines: One for each λ₁ [2.5, 3.0, 3.5, 4.0, 4.5]
- Legend: λ₁ values

**Expected Insight**:
- Does wait time decrease with higher λ₃?
- How much does λ₁ interfere with efficiency gains?

---

#### **Plot 9: Weight Ratio Analysis**
**Type**: Scatter plot with trend line  
**Purpose**: Test if λ₁/λ₃ ratio is the key factor

**Implementation**:
- X-axis: λ₁/λ₃ ratio (ranges from 0.5 to 9.0)
- Y-axis (dual): 
  - Left: JFI (blue)
  - Right: Wait Time (red)
- Points: All 25 Pareto sweep runs
- Trend lines: Linear regression for both metrics

**Expected Insight**:
- Is performance determined by the ratio or absolute values?
- What is the optimal λ₁/λ₃ ratio for balance?

---

### **SECTION 4: EWMA Validation (RQ2.1)** 📊
*Does the EWMA fairness metric work as intended?*

#### **Plot 10: EWMA vs Traditional Fairness (Correlation)**
**Type**: Scatter plot with regression line  
**Purpose**: Validate EWMA captures fairness

**Implementation**:
- X-axis: EWMA Coefficient of Variation (from summary)
- Y-axis: Jain's Fairness Index
- Points: All composite strategy runs (25 Pareto + 15 Gamma)
- Regression line: Show correlation coefficient (R²)
- Annotations: Highlight outliers

**Expected Insight**:
- Are EWMA CV and JFI strongly correlated?
- Does EWMA provide unique information beyond JFI?

---

#### **Plot 11: EWMA-Only vs Composite Strategy**
**Type**: Grouped bar chart  
**Purpose**: Compare EWMA-Only baseline to best composite configs

**Implementation**:
- X-axis: 5 metrics (JFI, Gini, Wait Time, TAR, % Zero Workers)
- Bars (grouped):
  - EWMA-Only baseline (γ=0.5)
  - Best Balanced composite (λ₁=3.5, λ₃=1.0, γ=0.5)
  - Best HighFairness composite (λ₁=4.5, λ₃=0.5, γ=0.5)
- Normalize to 0-100 scale for comparison

**Expected Insight**:
- Does integrating EWMA into composite score improve over EWMA-Only?
- What is gained by the composite approach?

---

### **SECTION 5: Gamma Sensitivity Analysis (RQ2.2)** 🔬
*Critical test: Does the smoothing factor matter?*

#### **Plot 12: Gamma Impact on Fairness (Line Plot)**
**Type**: Multi-line plot (3 configurations)  
**Purpose**: Visualize gamma's effect on JFI

**Implementation**:
- X-axis: γ values [0.1, 0.3, 0.5, 0.7, 0.9]
- Y-axis: Jain's Fairness Index
- Lines (3): Balanced, HighFairness, Efficiency
- Markers: Show individual data points
- Error bands: If any variance

**Expected Insight**:
- Is there a statistically significant trend?
- Or is JFI flat across gamma (robust to parameter)?

---

#### **Plot 13: Gamma Impact on Wait Time (Line Plot)**
**Type**: Multi-line plot (3 configurations)  
**Purpose**: Test if gamma affects efficiency

**Implementation**:
- X-axis: γ values [0.1, 0.3, 0.5, 0.7, 0.9]
- Y-axis: Mean Wait Time (minutes)
- Lines (3): Balanced, HighFairness, Efficiency
- Markers: Individual data points

**Expected Insight**:
- Does gamma impact wait time?
- Is the effect consistent across configurations?

---

#### **Plot 14: Gamma Sensitivity Summary (Small Multiples)**
**Type**: 3×3 grid of small line plots  
**Purpose**: Comprehensive gamma impact across all metrics

**Implementation**:
- Rows (3): Balanced, HighFairness, Efficiency configs
- Columns (3): JFI, Wait Time, TAR
- Each subplot: γ on X-axis, metric on Y-axis
- Highlight: Best γ for each config (if exists)

**Expected Insight**:
- Is there an optimal γ for each configuration?
- Or is the system gamma-invariant (key finding!)?

---

### **SECTION 6: Composite vs Baselines Deep Dive** 🏆
*Justifying the composite approach*

#### **Plot 15: Performance Distribution Comparison**
**Type**: Box plots (from percentile data)  
**Purpose**: Compare task and worker distributions

**Implementation**:
- X-axis: 5 strategies (Greedy, LAF, EWMA-Only, Best Composite, Best Fairness)
- Y-axis: 
  - Subplot A: Tasks per Worker distribution (P10, P50, P90)
  - Subplot B: Wait Time distribution (P10, P50, P95)
- Show: Mean as diamond marker

**Expected Insight**:
- Does composite strategy reduce inequality in task distribution?
- Are wait times more consistent?

---

#### **Plot 16: Efficiency Overhead Analysis**
**Type**: Bar chart with percentage labels  
**Purpose**: Quantify the cost of fairness

**Implementation**:
- X-axis: 4 strategies (Greedy, Best Composite, LAF, EWMA-Only)
- Y-axis: 
  - Bar 1: Mean Wait Time (minutes)
  - Bar 2: % Increase vs Greedy (annotated)
- Color: Gradient from green (low overhead) to red (high overhead)

**Expected Insight**:
- How much wait time do we trade for fairness?
- Is the composite strategy "worth it" compared to Greedy?

---

#### **Plot 17: Fairness Improvement Analysis**
**Type**: Bar chart with percentage labels  
**Purpose**: Quantify fairness gains over baselines

**Implementation**:
- X-axis: 4 strategies (Greedy, Best Composite, LAF, EWMA-Only)
- Y-axis:
  - Bar 1: Jain's Fairness Index (JFI)
  - Bar 2: % Improvement vs Greedy (annotated)
- Reference line: JFI = 0.80 (acceptable threshold)

**Expected Insight**:
- How much fairness do we gain with composite strategy?
- Does composite strategy match LAF's fairness?

---

### **SECTION 7: Operational Insights** 🔧
*Practical considerations for deployment*

#### **Plot 18: Task Assignment Ratio Consistency**
**Type**: Violin plot (approximated from percentiles)  
**Purpose**: Ensure TAR is stable across all strategies

**Implementation**:
- X-axis: 6 categories (3 Baselines, 3 Composite groups)
- Y-axis: Task Assignment Ratio (TAR %)
- Show: Distribution approximation
- Reference line: 90% TAR threshold

**Expected Insight**:
- Are all strategies achieving >90% TAR?
- Is TAR affected by fairness/efficiency trade-offs?

---

#### **Plot 19: Zero-Task Worker Analysis**
**Type**: Stacked bar chart  
**Purpose**: Understand fairness from allocation perspective

**Implementation**:
- X-axis: 8 key configurations
- Y-axis: 
  - Stack 1: % Workers with 0 tasks (red)
  - Stack 2: % Workers with 1-3 tasks (yellow)
  - Stack 3: % Workers with 4+ tasks (green)
- Annotations: Total % workers with 0 tasks

**Expected Insight**:
- Which strategies minimize zero-task workers?
- Is this correlated with JFI?

---

#### **Plot 20: Runtime Performance**
**Type**: Grouped bar chart  
**Purpose**: Ensure computational feasibility

**Implementation**:
- X-axis: 4 strategies (Greedy, Composite, LAF, EWMA-Only)
- Y-axis: Mean Runtime (seconds)
- Bars: Show mean and range (if multiple runs)
- Annotations: Tasks completed per second

**Expected Insight**:
- What is the computational overhead of composite strategy?
- Are all strategies feasible for real-time deployment?

---

### **SECTION 8: Recommendations & Summary** 📋
*Actionable insights for deployment*

#### **Summary Table 1: Baseline Comparison**
**Type**: Formatted table with color coding  
**Purpose**: Quick reference for strategy selection

**Columns**:
- Strategy
- TAR
- JFI
- Gini
- Wait Time (min)
- % Zero Workers
- Runtime (s)
- **Recommendation** (🏆 Best / ✅ Good / ⚠️ Use with Caution)

---

#### **Summary Table 2: Top 5 Composite Configurations**
**Type**: Ranked table with highlights  
**Purpose**: Recommend optimal parameter settings

**Columns**:
- Rank
- λ₁, λ₂, λ₃, γ
- JFI
- Wait Time
- TAR
- **Use Case** (Fairness Priority / Balanced / Efficiency Priority)

---

#### **Decision Matrix: When to Use Each Strategy**
**Type**: Text flowchart / decision guide  
**Purpose**: Practical deployment guidance

**Decision Nodes**:
1. **What is your primary constraint?**
   - **Minimize Wait Time** → Use Greedy OR Efficiency Composite (λ₁=2.5, λ₃=2.0)
   - **Maximize Fairness** → Use LAF OR HighFairness Composite (λ₁=4.5, λ₃=0.5)
   - **Balanced Performance** → Use Balanced Composite (λ₁=3.5, λ₃=1.0)

2. **Do you need EWMA-based fairness tracking?**
   - **Yes** → Use Composite with γ=0.5 (default, robust)
   - **No** → Use LAF or Greedy

3. **What is your worker density?**
   - **High (many workers)** → Fairness matters more (increase λ₁)
   - **Low (few workers)** → Efficiency matters more (increase λ₃)

---

## 📋 Key Research Questions Addressed

### **RQ1: Fairness-Efficiency Trade-off**
- **Plots**: 3, 4, 5, 6, 7, 8, 9 (Pareto heatmaps, frontier, sensitivity)
- **Answer**: [To be determined - optimal λ₁/λ₃ ratio]

### **RQ2.1: EWMA Validation**
- **Plots**: 10, 11 (EWMA correlation, EWMA-Only comparison)
- **Answer**: [To be determined - is EWMA useful?]

### **RQ2.2: Gamma Sensitivity**
- **Plots**: 12, 13, 14 (Gamma impact on fairness, wait time, all metrics)
- **Answer**: [To be determined - preliminary: gamma has minimal impact]

### **RQ3: Composite vs Baselines**
- **Plots**: 1, 2, 15, 16, 17 (Baseline overview, distribution comparison)
- **Answer**: [To be determined - does composite dominate?]

---

## 🎯 Expected Findings

**Hypothesis 1**: Composite strategy lies on Pareto frontier between Greedy and LAF
- **Test**: Plot 5 (Pareto Frontier Curve)

**Hypothesis 2**: Gamma has minimal impact on performance (PRELIMINARY CONFIRMED)
- **Test**: Plots 12, 13, 14 (Gamma Sensitivity)
- **Implication**: EWMA is robust to parameter choice (positive finding!)

**Hypothesis 3**: Optimal λ₁/λ₃ ratio depends on fairness priority
- **Test**: Plot 9 (Weight Ratio Analysis)

**Hypothesis 4**: EWMA-Only baseline performs worse than optimized composite
- **Test**: Plot 11 (EWMA-Only vs Composite)

**Hypothesis 5**: All strategies achieve >90% TAR (system capacity validated)
- **Test**: Plot 18 (TAR Consistency)

---

## 📁 Implementation Notes

**Input Data**: `experiment_014_aggregate_results.csv` (43 rows)

**Output**:
1. **Jupyter Notebook** (`analysis.ipynb`) with:
   - Executive summary
   - 20 plots organized into 8 sections
   - Statistical tests (correlations, significance)
   - **Clear recommendations** for deployment

2. **Key Figures** (exports):
   - Plot 3, 4 (Heatmaps) - for presentations
   - Plot 5 (Pareto Frontier) - core finding
   - Summary Tables - for quick reference

3. **Results Summary** (`RESULTS.md`):
   - Optimal parameter recommendations
   - Strategy comparison table
   - Deployment guidance

---

## 🎓 Analysis Principles

1. **RQ-Focused**: Every plot directly addresses a research question
2. **Comparative**: Always benchmark against baselines (Greedy, LAF, EWMA-Only)
3. **Trade-off Aware**: Acknowledge no single "best" - depends on priorities
4. **Statistically Rigorous**: Test significance of findings (especially gamma)
5. **Deployment-Ready**: Provide actionable recommendations with use cases

---

**Total Plots**: 20 visualizations + 2 summary tables + 1 decision matrix  
**Priority**: CRITICAL (validates core thesis contributions - EWMA + Composite)  
**Estimated Analysis Time**: 2-3 hours  
**Status**: READY TO IMPLEMENT 🚀

---

## 🔍 Special Considerations

### Gamma Analysis
- If gamma shows no significant effect (as preliminary results suggest):
  - This is a **positive finding** (system is robust)
  - Recommend γ=0.5 as default (middle ground)
  - Simplifies deployment (one less parameter to tune)

### Pareto Frontier
- Identify "knee" of curve (optimal balanced solution)
- Flag configurations that are "dominated" (worse on all metrics)
- Consider multi-objective optimization score for ranking

### Baseline Comparison
- Be fair to baselines (acknowledge their strengths)
- LAF may have better fairness but at what cost?
- Greedy may be acceptable if fairness isn't critical

---

**Next Step**: Implement notebook with all 8 sections and 20 plots! 📊

