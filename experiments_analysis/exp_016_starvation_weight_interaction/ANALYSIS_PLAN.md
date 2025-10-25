# Experiment 016: Starvation Weight Interaction Analysis Plan
## Validating λ₂=0.5 Across Fairness-Utility Trade-offs

**Last Updated**: October 24, 2025  
**Status**: Experiment Running (6/28 complete)  
**Data**: 28 experiments (3 Baselines + 25 Starvation Sweep)

---

## 🎯 Primary Goals

1. **Validate λ₂=0.5 Universality**: Confirm that λ₂=0.5 remains optimal across different λ₁/λ₃ configurations
2. **Test Interaction Effects**: Determine if optimal starvation weight depends on fairness-utility balance
3. **Quantify Starvation Mitigation**: Measure how λ₂ reduces task wait time extremes (P95/P99)
4. **Define Safe Operating Range**: Identify acceptable λ₂ range for robust performance

---

## 📊 Experiment Summary

### Baseline Strategies (3)
| Exp | Strategy | Config | Purpose |
|-----|----------|--------|---------|
| 1 | Greedy | proximity-only | Efficiency reference |
| 2 | LAF | fairness-first | Simple fairness baseline |
| 3 | EWMA-Only | EWMA fairness (γ=0.5) | Advanced fairness baseline |

### Starvation Weight Sweep (25)
**5 Representative Configurations × 5 λ₂ Values**

| Config | λ₁ | λ₃ | Description | Rationale |
|--------|----|----|-------------|-----------|
| Best_JFI | 4.5 | 0.5 | High fairness focus | Max JFI from Exp 014 |
| Balanced | 3.5 | 1.0 | Standard balanced | Reference config |
| Mid_Range | 3.5 | 2.5 | Moderate efficiency | Near-optimal balance |
| Efficiency_Leaning | 2.5 | 2.0 | Efficiency-focused | Leaning toward utility |
| Best_Efficiency | 2.5 | 2.5 | Maximum efficiency | Lowest wait time from Exp 014 |

**λ₂ Values Tested**: [0.0, 0.5, 1.0, 1.5, 2.0]

---

## 📈 Analysis Sections & Plots

### **SECTION 1: Executive Summary & Baseline Comparison** ⭐
*Setting performance benchmarks*

#### **Plot 1: Baseline Performance Overview (4-Panel)**
**Type**: Grouped bar charts (2×2 grid)  
**Purpose**: Establish reference performance across strategies

**Panels**:
- **A (Top-Left)**: Task Assignment Ratio (TAR)
- **B (Top-Right)**: Jain's Fairness Index (JFI)
- **C (Bottom-Left)**: Mean Wait Time
- **D (Bottom-Right)**: P95 Wait Time (starvation indicator)

**Bars**: Greedy, LAF, EWMA-Only (3 baselines)

**Expected Insight**:
- Greedy: Lowest wait time, moderate fairness
- LAF: Highest JFI, longer wait times
- EWMA-Only: Advanced fairness, highest wait times
- Establish context for composite strategy performance

---

### **SECTION 2: Starvation Weight Impact Analysis** ⭐⭐ KEY SECTION
*Core research questions: Does optimal λ₂ vary? Is there interaction?*

#### **Plot 2: λ₂ Impact Heatmap - JFI**
**Type**: 5×5 heatmap  
**Purpose**: Visualize JFI across all (Config, λ₂) combinations

**Implementation**:
- X-axis: λ₂ values [0.0, 0.5, 1.0, 1.5, 2.0]
- Y-axis: 5 configurations (Best_JFI → Best_Efficiency)
- Color: JFI value (0.60-0.85, higher is better)
- Annotations: Cell values, mark optimal λ₂ per config

**Expected Insight**:
- **Hypothesis**: Each row (config) should have peak JFI near λ₂=0.5
- If optimal λ₂ varies by config → Interaction effect exists
- If optimal λ₂ is consistent → λ₂=0.5 is universal

---

#### **Plot 3: λ₂ Impact Heatmap - Mean Wait Time**
**Type**: 5×5 heatmap  
**Purpose**: Visualize efficiency across all (Config, λ₂) combinations

**Implementation**:
- X-axis: λ₂ values
- Y-axis: 5 configurations
- Color: Mean Wait Time (2-8 min, lower is better)
- Annotations: Cell values, mark optimal λ₂ per config

**Expected Insight**:
- Does λ₂ significantly affect mean wait time?
- Or is wait time primarily controlled by λ₁/λ₃ balance?
- Validate that starvation component doesn't hurt efficiency

---

#### **Plot 4: λ₂ Impact Heatmap - P95 Wait Time**
**Type**: 5×5 heatmap  
**Purpose**: Measure starvation mitigation effectiveness

**Implementation**:
- X-axis: λ₂ values
- Y-axis: 5 configurations
- Color: P95 Wait Time (8-15 min, lower is better)
- Annotations: Cell values, mark optimal λ₂ per config

**Expected Insight**:
- **KEY METRIC**: This shows starvation mitigation in action
- λ₂=0.0 should have highest P95 (no starvation protection)
- λ₂=0.5-1.0 should show reduced extremes
- λ₂>1.0 should show diminishing returns

---

#### **Plot 5: Optimal λ₂ Identification (Line Plot)**
**Type**: Multi-line plot showing performance vs λ₂  
**Purpose**: Find optimal λ₂ for each configuration

**Implementation**:
- X-axis: λ₂ [0.0, 0.5, 1.0, 1.5, 2.0]
- Y-axis: Composite Score (weighted: 0.4×JFI + 0.4×(1/Wait) + 0.2×(1/P95))
- Lines: 5 separate lines (one per config)
- Markers: Peak points (optimal λ₂)

**Decision Rule**:
- For each config, find λ₂ maximizing composite score
- Check if all 5 peaks are at λ₂=0.5 (validates universality)

**Expected Insight**:
- **RQ3.4 Answer**: If all peaks at λ₂≈0.5 → Universal optimum confirmed
- If peaks vary → Document interaction effects

---

### **SECTION 3: Configuration-Specific Deep Dives**
*Understanding how λ₂ behaves under different primary objectives*

#### **Plot 6: Best_JFI Config - λ₂ Sensitivity**
**Type**: 3-panel line plots  
**Purpose**: Detailed λ₂ analysis for high-fairness config

**Panels**:
- **A**: JFI vs λ₂
- **B**: Mean Wait Time vs λ₂
- **C**: P95 Wait Time vs λ₂

**Config**: λ₁=4.5, λ₃=0.5 (fairness-focused)

**Expected Insight**:
- Does adding starvation mitigation hurt fairness?
- What's the trade-off between JFI and wait time reduction?

---

#### **Plot 7: Best_Efficiency Config - λ₂ Sensitivity**
**Type**: 3-panel line plots (same layout as Plot 6)  
**Purpose**: Detailed λ₂ analysis for high-efficiency config

**Config**: λ₁=2.5, λ₃=2.5 (efficiency-focused)

**Expected Insight**:
- Is starvation mitigation still valuable for efficiency-focused systems?
- Does λ₂ have less impact when already optimizing for low wait times?

---

#### **Plot 8: Balanced Config - λ₂ Sensitivity**
**Type**: 3-panel line plots (same layout as Plot 6)  
**Purpose**: Detailed λ₂ analysis for balanced config

**Config**: λ₁=3.5, λ₃=1.0 (standard balanced)

**Expected Insight**:
- Reference case: How does λ₂ behave in "typical" configuration?
- Validate λ₂=0.5 for the most commonly used setup

---

### **SECTION 4: Interaction Effect Analysis** ⭐
*RQ10.2: Are there interaction effects between λ₂ and (λ₁, λ₃)?*

#### **Plot 9: λ₂ Effect by Configuration (Grouped Bar)**
**Type**: Grouped bar chart  
**Purpose**: Compare λ₂ impact across configurations

**Implementation**:
- X-axis: 5 configurations
- Y-axis: % Reduction in P95 Wait Time (λ₂=0.5 vs λ₂=0.0)
- Bars: Grouped by metric (P95 reduction, JFI change, Mean Wait change)
- Horizontal line: 0% (no change)

**Statistical Test**: One-way ANOVA  
- H₀: λ₂ effect is consistent across all configurations
- H₁: λ₂ effect varies significantly by configuration

**Expected Insight**:
- If all bars similar height → No interaction (additive effects)
- If bars vary significantly → Interaction exists (optimal λ₂ depends on λ₁/λ₃)

---

#### **Plot 10: Interaction Surface Plot (3D)**
**Type**: 3D surface or contour plot  
**Purpose**: Visualize multi-parameter relationship

**Axes**:
- X: λ₁ (Fairness weight) [2.5, 3.5, 4.5]
- Y: λ₂ (Starvation weight) [0.0, 0.5, 1.0, 1.5, 2.0]
- Z: JFI or P95 Wait Time
- Hold λ₃ constant at representative values

**Expected Insight**:
- Is surface smooth (weak interaction) or wavy (strong interaction)?
- Does optimal λ₂ path follow a ridge?

---

### **SECTION 5: Starvation Mitigation Quantification**
*Measuring the benefit of λ₂ component*

#### **Plot 11: λ₂=0.0 vs λ₂=0.5 Comparison (Scatter)**
**Type**: Scatter plot with parity line  
**Purpose**: Quantify improvement from adding starvation mitigation

**Implementation**:
- X-axis: Metric with λ₂=0.0
- Y-axis: Same metric with λ₂=0.5
- Points: 5 configurations
- Line: y=x (parity line)
- Annotate: % improvement

**Metrics to Compare**:
- P95 Wait Time (expect reduction)
- Mean Wait Time (expect slight reduction)
- JFI (expect minimal change)

**Expected Insight**:
- **Key Question**: How much does λ₂=0.5 improve worst-case wait times?
- Is improvement consistent across all configurations?

---

#### **Plot 12: Starvation Reduction vs λ₂ (Line Plot)**
**Type**: Multi-line plot  
**Purpose**: Show diminishing returns for increasing λ₂

**Implementation**:
- X-axis: λ₂ [0.0, 0.5, 1.0, 1.5, 2.0]
- Y-axis: P95 Wait Time (minutes)
- Lines: 5 configurations
- Markers: λ₂=0.5 (current default)

**Expected Insight**:
- Does P95 decrease significantly beyond λ₂=0.5?
- Or is there diminishing returns (validate λ₂=0.5 sufficiency)?

---

### **SECTION 6: Safe Operating Range**
*Define acceptable λ₂ bounds for robust deployment*

#### **Plot 13: Performance Degradation Analysis**
**Type**: Multi-line plot with confidence bands  
**Purpose**: Identify safe λ₂ range

**Implementation**:
- X-axis: λ₂ [0.0, 0.5, 1.0, 1.5, 2.0]
- Y-axis: % Deviation from optimal (composite score)
- Lines: Mean across 5 configs
- Bands: Min-Max range
- Horizontal lines: ±5% and ±10% thresholds

**Decision Rule**:
- "Safe range" = λ₂ values within 5% of optimal
- "Acceptable range" = λ₂ values within 10% of optimal

**Expected Insight**:
- Is λ₂∈[0.3, 0.7] acceptable? (±0.2 from default)
- How sensitive is performance to λ₂ miscalibration?

---

#### **Plot 14: Robustness Scatter (JFI vs Wait Time)**
**Type**: Scatter plot  
**Purpose**: Show trade-off space for different λ₂

**Implementation**:
- X-axis: Mean Wait Time
- Y-axis: JFI
- Points: All 25 starvation sweep experiments
- Color: λ₂ value (gradient)
- Size: Config type
- Pareto frontier: Connect non-dominated points

**Expected Insight**:
- Do different λ₂ values explore different parts of the trade-off space?
- Or do they all cluster (indicating λ₂ is orthogonal to fairness-efficiency)?

---

### **SECTION 7: Comparison to Baselines & Prior Work**

#### **Plot 15: Best Composite vs Baselines (Radar Chart)**
**Type**: Radar/spider chart  
**Purpose**: Show comprehensive performance profile

**Strategies**:
- Greedy Baseline
- LAF Baseline
- EWMA-Only Baseline
- Best Composite (Config=Balanced, λ₂=0.5)

**Metrics (Normalized 0-1)**:
1. TAR (higher is better)
2. JFI (higher is better)
3. Inverse Mean Wait Time (higher is better)
4. Inverse P95 Wait Time (higher is better)
5. Worker Utilization (higher is better)
6. Inverse Gini (higher = more equitable)

**Expected Insight**:
- Does composite strategy dominate baselines across all metrics?
- Where does composite strategy excel vs struggle?

---

#### **Plot 16: Comparison to Exp 009 Findings**
**Type**: Side-by-side bar chart  
**Purpose**: Validate consistency with prior λ₂ testing

**Implementation**:
- X-axis: Experiment source (Exp 009 vs Exp 016)
- Y-axis: Optimal λ₂ identified
- Bars: Different configurations tested
- Error bars: Performance range

**Expected Insight**:
- Are Exp 016 findings consistent with Exp 009?
- Did testing across more configurations reveal new insights?

---

### **SECTION 8: Statistical Validation & Summary**

#### **Table 1: Optimal λ₂ by Configuration**
**Type**: Summary table with color coding  
**Purpose**: Clear decision support

**Columns**:
- Configuration (λ₁, λ₃)
- Optimal λ₂ (by composite score)
- JFI at optimal λ₂
- Mean Wait at optimal λ₂
- P95 Wait at optimal λ₂
- % Improvement vs λ₂=0.0
- **Recommendation** (🟢 Use λ₂=0.5 / 🟡 Consider tuning / 🔴 Avoid)

**Color Coding**:
- 🟢 **Universal**: Optimal λ₂ = 0.5 (±0.1)
- 🟡 **Context-Dependent**: Optimal λ₂ varies by 0.2+
- 🔴 **Inconsistent**: No clear optimal

---

#### **Table 2: Statistical Tests Summary**
**Type**: Statistical results table  
**Purpose**: Formal hypothesis testing

**Tests**:
1. **ANOVA**: Does optimal λ₂ vary significantly across configs?
   - F-statistic, p-value, interpretation

2. **Correlation Analysis**: λ₂ effect vs (λ₁, λ₃) values
   - Pearson r, significance

3. **Pairwise Comparisons**: λ₂=0.5 vs other values
   - t-tests for each config, Bonferroni correction

**Expected Results**:
- p > 0.05 → No significant interaction (λ₂=0.5 is universal)
- p < 0.05 → Interaction exists (optimal λ₂ depends on config)

---

### **SECTION 9: Deployment Recommendations** ⭐ FINAL OUTPUT

#### **Decision Guide: Setting λ₂**
**Type**: Text flowchart / decision tree  
**Purpose**: Actionable deployment guidance

**Recommendations**:

**1. Standard Deployment**
- **Use λ₂=0.5** (validated across all configurations)
- Expected: 10-20% reduction in P95 wait times vs λ₂=0.0
- Trade-off: Minimal impact on mean performance (<2%)

**2. High-Fairness Systems** (λ₁>4.0)
- Recommended λ₂: [Based on analysis]
- Rationale: [From Plot 6 findings]

**3. High-Efficiency Systems** (λ₃>2.0)
- Recommended λ₂: [Based on analysis]
- Rationale: [From Plot 7 findings]

**4. Safe Operating Range**
- Acceptable: λ₂ ∈ [0.3, 0.7] (within 5% of optimal)
- Avoid: λ₂ < 0.2 or λ₂ > 1.0

---

## 📋 Key Research Questions Addressed

### **RQ3.4: Does optimal λ₂ vary with different λ₁/λ₃ configurations?**
- **Primary Plots**: 2, 3, 4, 5, 9
- **Answer**: [To be determined from analysis]
- **Expected**: No significant variation (λ₂=0.5 is universal)

### **RQ10.2: Are there interaction effects between λ₂ and (λ₁, λ₃)?**
- **Primary Plots**: 9, 10
- **Statistical Test**: ANOVA, correlation analysis
- **Answer**: [To be determined from analysis]
- **Expected**: Weak or no interaction (additive effects)

### **Secondary Questions**:
- **How much does λ₂ reduce task starvation?**
  - Plots: 11, 12 (P95 reduction analysis)
  
- **What is the safe λ₂ range for deployment?**
  - Plots: 13, 14 (Robustness analysis)
  
- **How does composite strategy compare to baselines?**
  - Plot: 15 (Radar chart comparison)

---

## 🎯 Expected Findings & Hypotheses

**Hypothesis 1: λ₂=0.5 is Universal**
- **Test**: Optimal λ₂ should be 0.5 (±0.1) for all 5 configurations
- **Plot**: 5 (Optimal λ₂ Identification)
- **Validation**: Table 1, Statistical Test 1 (ANOVA p>0.05)

**Hypothesis 2: No Strong Interaction**
- **Test**: λ₂ effect on P95 should be consistent across configs
- **Plot**: 9 (λ₂ Effect by Configuration)
- **Validation**: ANOVA shows no significant difference

**Hypothesis 3: Starvation Mitigation is Orthogonal**
- **Test**: λ₂ reduces P95 without affecting JFI or mean wait significantly
- **Plots**: 11 (λ₂=0.0 vs 0.5 comparison)
- **Expected**: P95 down 10-20%, JFI change <5%, Mean Wait change <5%

**Hypothesis 4: Diminishing Returns Beyond λ₂=0.5**
- **Test**: P95 reduction plateaus for λ₂>1.0
- **Plot**: 12 (Starvation Reduction vs λ₂)
- **Expected**: Marginal benefit <5% for λ₂: 0.5→1.0

**Hypothesis 5: Robustness to Miscalibration**
- **Test**: Performance within 5% of optimal for λ₂∈[0.3, 0.7]
- **Plot**: 13 (Performance Degradation)
- **Expected**: Wide safe operating range

---

## 📊 Data Quality Checks

Before analysis, verify:
- [ ] All 28 simulations completed successfully
- [ ] Aggregate CSV has 28 rows (3 baselines + 25 starvation sweep)
- [ ] No missing values in key metrics (JFI, Mean Wait, P95 Wait)
- [ ] TAR >75% for all experiments (reasonable assignment)
- [ ] All 5 configs represented in starvation sweep
- [ ] All 5 λ₂ values tested for each config

Expected Performance Ranges:
- TAR: 75-95%
- JFI: 0.60-0.85
- Mean Wait: 2-8 minutes
- P95 Wait: 8-15 minutes

---

## 📁 Implementation Notes

**Input Data**: `experiment_016_aggregate_results.csv`

**Output**:
1. **Jupyter Notebook** (`analysis.ipynb`) with:
   - Executive summary
   - 16 plots organized into 9 sections
   - 2 summary tables
   - Statistical tests
   - **Clear deployment recommendation**

2. **Key Figures** (exports):
   - Plot 2-4 (Heatmaps) - interaction visualization
   - Plot 5 (Optimal λ₂) - key finding
   - Plot 9 (Interaction effect) - statistical result
   - Plot 15 (Radar chart) - comprehensive comparison
   - Table 1 (Summary) - quick reference

3. **Results Summary** (`RESULTS.md`):
   - RQ3.4 and RQ10.2 answers
   - λ₂=0.5 validation status
   - Deployment guidance
   - Comparison to Exp 009

---

## 🎓 Analysis Principles

1. **Hypothesis-Driven**: Every plot tests a specific hypothesis about λ₂ behavior
2. **Configuration-Aware**: Analyze each of 5 configs separately before generalizing
3. **Statistically Rigorous**: Use formal tests to validate universality claims
4. **Deployment-Focused**: Provide clear, actionable recommendations
5. **Conservative**: Err on side of caution when recommending parameter changes

---

**Total Plots**: 16 focused visualizations + 2 summary tables  
**Priority**: HIGH (validates core thesis assumption about λ₂)  
**Estimated Analysis Time**: 2-2.5 hours  
**Status**: READY TO IMPLEMENT (once experiment completes)

---

## 🚀 Next Steps

1. **Wait for Experiment Completion** (~3 hours remaining)
2. **Validate Data Quality** (run data checks)
3. **Create Analysis Notebook** (implement this plan)
4. **Generate Key Findings** (answer RQ3.4 and RQ10.2)
5. **Update Research Questions Framework** (mark as validated)
6. **Compare to Experiment 009** (consistency check)
7. **Prepare Thesis Section** (starvation weight validation)

---

**This analysis will definitively answer whether λ₂=0.5 is a universal optimal or if it requires context-specific tuning.**

