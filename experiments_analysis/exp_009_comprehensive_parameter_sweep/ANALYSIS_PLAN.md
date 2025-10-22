# Experiment 009: Comprehensive Analysis Plan
## Post-Normalization Parameter Sweep Analysis Framework

**Last Updated**: October 20, 2025  
**Status**: Ready for Implementation  
**Data**: 42 experiments (1 Greedy + 41 Composite configurations)

---

## 🎯 Analysis Objectives

This is the **final base experiment** before moving to advanced techniques. The goal is to extract maximum insight about the fairness-efficiency trade-off space now that we've resolved the Worker Idle Time Paradox through score normalization.

### Primary Research Questions
1. **RQ1 (Fairness-Efficiency Trade-off)**: What is the optimal balance between fairness (λ₁), starvation prevention (λ₂), and efficiency (λ₃)?
2. **RQ2 (Normalization Validation)**: Has score normalization eliminated the idle time paradox while preserving fairness gains?
3. **RQ3 (Parameter Sensitivity)**: Which parameters have the strongest impact on system performance?
4. **RQ4 (Emergent Behaviors)**: Are there non-linear interactions or sweet spots in the parameter space?

---

## 📊 Analysis Plots - Organized by Category

---

## 🔄 **SECTION 1: Temporal Evolution Plots**
*Track system behavior and metrics over the course of simulation*

⚠️ **Data Availability**: These plots require event-level temporal data not collected in Experiment 009

### **Plot A: Task Wait Time Evolution (System Responsiveness)**
**Type**: Line Plot (Smoothed Average over Time)  
**Purpose**: Show how quickly the system stabilizes under traffic load

**Data Requirements** ⚠️:
- **Temporal event-level data**: Task wait times recorded at each assignment event
- **Granularity**: Per-event or windowed (e.g., every 100 tasks)
- **Current Status**: ❌ Not collected in Experiment 009 (aggregate-only)

**Implementation Notes**:
- Would require modification to `DiagnosticTracker` to record per-event metrics
- X-axis: Event index or simulation time
- Y-axis: Rolling mean task wait time (window = 50-100 events)
- Lines: 4 key profiles (see Key Profiles section below)

**Expected Insights**:
- Greedy should stabilize fastest at lowest wait time
- High Fairness should have longer wait but stabilize after initial burst
- Compare Sweet Spot vs. Starvation Ablation: Does L2=0.0 cause late-simulation spikes?

---

### **Plot B: Worker Fairness Index Evolution (Temporal Equity)**
**Type**: Line Plot (JFI Over Time)  
**Purpose**: Prove the system actively maintains equity (not just a final snapshot)

**Data Requirements** ⚠️:
- **Temporal fairness calculations**: JFI computed over rolling time windows
- **Granularity**: Every N events (e.g., 500 events)
- **Current Status**: ❌ Not collected in Experiment 009

**Implementation Notes**:
- X-axis: Simulation time or event index
- Y-axis: Jain's Fairness Index (rolling window)
- Lines: 4 key profiles

**Expected Insights**:
- Greedy: Low, flat JFI (no history consideration)
- High Fairness: Rapid jump to high JFI, maintains it
- Proves Composite strategy actively corrects fairness over time

---

### **Plot D: Score Component Dominance Evolution (Mechanism Validation)**
**Type**: Stacked Area Plot or Multi-Line Proportion Plot  
**Purpose**: Validate the internal mechanics of the composite scoring function

**Data Requirements** ⚠️:
- **Per-assignment component dominance**: Which component (F/S/U) had highest weighted value at each assignment
- **Temporal tracking**: Over the course of simulation
- **Current Status**: ❌ Not collected (diagnostic_tracker disabled for performance)

**Implementation Notes**:
- X-axis: Simulation time or event index
- Y-axis: Percentage of assignments dominated by each component (sums to 100%)
- Stacked areas: Fairness (blue), Starvation (orange), Utility (green)
- Focus: One configuration (e.g., Sweet Spot)

**Expected Insights**:
- Initial high Utility dominance (finding nearest worker)
- Gradual shift to Fairness/Starvation as environment stabilizes
- Confirms system shifts from short-term efficiency to long-term equity

---

## 📊 **SECTION 2: Distribution & Spread Analysis**
*Examine how metrics are distributed across workers, tasks, or configurations*

### **Plot C: Worker Idle Time Distribution (Worker Experience)**
**Type**: Density Plot / Overlapping Histograms  
**Purpose**: Show worker-level equity from the worker's perspective

**Data Requirements** ⚠️:
- **Per-worker idle time data**: Individual idle time for each of 15,000 workers
- **Current Status**: ❌ Not collected in Experiment 009 (no worker-level CSVs saved)

**Implementation Notes**:
- X-axis: Worker idle time (minutes, filtered to 0-60 for clarity)
- Y-axis: Density / Frequency
- Overlay: 4 key profiles with different colors/alpha

**Expected Insights**:
- Successful fairness strategy should compress distribution (shorter tail)
- Fewer workers severely starved (> 30 minutes idle)
- If Starvation Ablation (L2=0.0) shows longer tail vs. Sweet Spot, confirms L2 works

---

### **Plot 18: Wait Time Distribution Comparison**
**Type**: Overlapping Density Plots  
**Data**: Using max_wait_time from full_summary  
✅ **Available with current data**

**Implementation**:
- X-axis: Maximum wait time (minutes)
- Y-axis: Density
- Overlays: Greedy, Low Fairness, Sweet Spot, High Fairness
- Shade: Fill under curves with alpha

**Potential Insight**:
- Does fairness reduce extreme wait times?
- Distribution shape differences (skew, kurtosis)

---

## ⚖️ **SECTION 3: Trade-off & Pareto Analysis**
*Explore competing objectives and identify optimal balances*

### **Plot 2: Pareto Frontier - Multi-Objective Optimization**
**Type**: Scatter Plot with Pareto Front Line  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: JFI (maximize)
- Y-axis: Mean Task Wait Time (minimize, inverted axis)
- Points: All experiments, colored by group
- Overlay: Pareto frontier curve
- Annotate: Key configurations (Greedy, Sweet Spot, High Fairness)

**Potential Insight**:
- Quantify the fairness-efficiency trade-off slope
- Identify dominated configurations (never optimal)
- Find knee point (best balance)

---

### **Plot 6: Efficiency Frontier - Travel Distance vs. Wait Time**
**Type**: Scatter Plot  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Mean Pickup Distance (km)
- Y-axis: Mean Task Wait Time (min)
- Points: Sized by JFI, colored by strategy group
- Ideal region: Bottom-left (low distance, low wait, high JFI)

**Potential Insight**:
- Are wait time and travel distance correlated?
- Can we achieve low wait time without excessive empty-km travel?

---

### **Plot 8: Task Assignment Ratio vs. Fairness**
**Type**: Scatter Plot with Trend Line  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: JFI
- Y-axis: Task Assignment Ratio (%)
- Color: Strategy group
- Trend: Linear regression + confidence interval

**Potential Insight**:
- Is there a penalty to TAR when prioritizing fairness?
- Does high fairness lead to more expired tasks?

---

### **Plot 11: Fairness Loss vs. Utility Difference (Supervisor Metrics)**
**Type**: Scatter Plot  
**Data**: All experiments with supervisor metrics  
✅ **Available with current data**

**Implementation**:
- X-axis: Utility Difference (from full_summary)
- Y-axis: Fairness Loss
- Color: JFI
- Size: TAR

**Potential Insight**:
- How do supervisor-level metrics correlate with global JFI?
- Are there configurations that minimize both simultaneously?

---

### **Plot 13: Empty-KM Share vs. Fairness**
**Type**: Scatter Plot  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: JFI
- Y-axis: Empty-KM Ratio (%)
- Color: Mean pickup distance
- Trend line

**Potential Insight**:
- Does prioritizing fairness increase wasted travel (empty kilometers)?
- Can we achieve fairness without efficiency loss?

---

## 🎛️ **SECTION 4: Parameter Space Exploration**
*Map the parameter landscape and identify optimal regions*

### **Plot 1: Parameter Space Heatmap - Fairness vs. Efficiency**
**Type**: 2D Heatmap (λ₁ vs. λ₃)  
**Data**: Group B (L1 × L3 Grid Sweep, 12 experiments)  
✅ **Available with current data**

**Implementation**:
- X-axis: λ₃ (Utility weight) - [0.5, 1.0, 2.0, 4.0]
- Y-axis: λ₁ (Fairness weight) - [0.0, 0.5, 1.0]
- Color: JFI (heatmap 1) or Mean Wait Time (heatmap 2)
- Create both heatmaps side-by-side

**Potential Insight**: 
- Identify the Pareto frontier visually
- Find parameter combinations that achieve high JFI without severe wait time penalty

---

### **Plot 4: Starvation Weight Ablation Study**
**Type**: Grouped Bar Chart or Line Plot  
**Data**: Group H (Low Utility, varying L2, 4 experiments)  
✅ **Available with current data**

**Implementation**:
- X-axis: λ₂ (Starvation weight) - [0.5, 1.0, 1.5, 2.0]
- Y-axes (dual): 
  - Primary: Max Wait Time (bars)
  - Secondary: JFI (line with markers)
- Purpose: Isolate L2's effect on preventing extreme wait times

**Potential Insight**:
- Does increasing L2 reduce max wait time (proves starvation prevention)?
- What's the diminishing returns threshold for L2?

---

### **Plot 5: Threshold Sensitivity Analysis**
**Type**: Faceted Line Plot (θ on X-axis)  
**Data**: Groups D, E, F (Balanced with varying thresholds, 12 experiments)  
✅ **Available with current data**

**Implementation**:
- X-axis: θ (Threshold) - [0.3, 0.5, 0.7]
- Y-axis: Key metrics (faceted subplots)
  - Subplot 1: JFI
  - Subplot 2: Mean Wait Time
  - Subplot 3: Peak Backlog
- Lines: Different (L1, L3) combinations

**Potential Insight**:
- Is the soft threshold still causing problems post-normalization?
- Optimal threshold value for normalized scoring

---

### **Plot 7: Composite Score Weight Triangle**
**Type**: Ternary Plot (3-Component Composition)  
**Data**: Subset with fixed threshold (e.g., θ=0.5)  
✅ **Available with current data**

**Implementation**:
- 3 axes: λ₁, λ₂, λ₃ (normalized to sum to 1)
- Points: Experiments, colored by JFI
- Gradient: Shows which regions of weight space optimize fairness

**Potential Insight**:
- Visualize the 3D parameter space in 2D
- Identify optimal weight ratios (not just absolute values)

---

### **Plot 17: Parameter Sweep Summary - Faceted Grid**
**Type**: Small Multiples (Facet Grid)  
**Data**: All composite experiments  
✅ **Available with current data**

**Implementation**:
- Grid: 3x3 or 4x4 small plots
- Each cell: Different parameter combination
- Within cell: Bar chart of key metrics
- Color: Performance tier (good/medium/poor)

**Potential Insight**:
- At-a-glance comparison of entire parameter space
- Quickly identify best/worst regions

---

## 🎯 **SECTION 5: Multi-Metric Configuration Comparison**
*Compare configurations holistically across all performance dimensions*

### **Plot 3: Spider/Radar Chart - Configuration Profiles**
**Type**: Radar Chart (Multi-Metric Comparison)  
**Data**: 4-6 key configurations  
✅ **Available with current data**

**Metrics (Normalized to 0-1 scale)**:
1. JFI (higher is better)
2. Task Assignment Ratio (higher is better)
3. Inverse Wait Time (higher is better)
4. Inverse Pickup Distance (higher is better)
5. EWMA CV (lower is better, inverted for display)
6. Empty-KM Ratio (lower is better, inverted)

**Configurations**:
- Greedy Baseline
- Sweet Spot (e.g., L1=0.5, L3=1.0)
- High Fairness (L1=2.0, L3=0.5)
- Low Fairness (L1=0.0, L3=2.0)

**Potential Insight**:
- Holistic performance comparison
- Identify configurations with balanced performance across all metrics

---

### **Plot 14: Statistical Significance - Greedy vs. Best Composite**
**Type**: Box Plot with Significance Stars  
**Data**: Greedy vs. top 3 composite configurations  
✅ **Available with current data** (note: single-run, so showing point estimates)

**Metrics (separate subplots)**:
- JFI
- Mean Wait Time
- TAR
- Pickup Distance

**Implementation**:
- X-axis: Configuration names
- Y-axis: Metric value
- Annotations: Effect size (Cohen's d) or percentage difference

**Potential Insight**:
- Quantify improvement over baseline
- Validate findings with effect sizes

---

### **Plot 19: Top 5 Configurations Comparison Table**
**Type**: Ranked Table with Sparklines  
**Data**: Top 5 by JFI, Top 5 by Wait Time, Top 5 by TAR  
✅ **Available with current data**

**Implementation**:
- Rows: Configurations
- Columns: Metrics with inline sparklines showing percentile rank
- Highlight: Best in each column

**Potential Insight**:
- Quick reference for best performers
- Identify multi-objective winners

---

## 🔧 **SECTION 6: System Behavior & Operational Diagnostics**
*Understand system-level operational characteristics*

### **Plot 9: Backlog Behavior - Peak vs. Variance**
**Type**: 2D Scatter with Annotations  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Peak Backlog (max simultaneous unassigned tasks)
- Y-axis: Mean Wait Time (proxy for typical backlog)
- Annotations: Configurations with extreme values

**Potential Insight**:
- Are there configurations with unstable backlog (high peak, low mean)?
- Does fairness smoothing reduce peak backlog?

---

### **Plot 10: Runtime Performance Analysis**
**Type**: Box Plot or Violin Plot  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Strategy groups (A, B, C, etc.)
- Y-axis: Duration (seconds)
- Overlay: Individual points

**Potential Insight**:
- Is there a performance cost to normalization?
- Do certain parameter combinations run slower?

---

### **Plot 12: EWMA CV (Coefficient of Variation) Analysis**
**Type**: Grouped Bar Chart  
**Data**: All composite strategies, grouped by main parameter sweep  
✅ **Available with current data**

**Implementation**:
- X-axis: Experiment groups
- Y-axis: EWMA CV (lower = more equitable EWMA distribution)
- Bars: Colored by group

**Potential Insight**:
- Does EWMA CV correlate with JFI?
- Is EWMA CV a better fairness metric than JFI for our use case?

---

## 🧮 **SECTION 7: Advanced Statistical & Meta-Analysis**
*Deep-dive analytical techniques to uncover latent patterns*

### **Plot 15: Correlation Matrix - All Metrics**
**Type**: Heatmap (Correlation Coefficients)  
**Data**: All 42 experiments  
✅ **Available with current data**

**Metrics**:
- JFI, TAR, Wait Time, Pickup Distance, Travel Distance, EWMA CV, Fairness Loss, Utility Difference, Peak Backlog, Duration

**Implementation**:
- Color scale: -1 (red) to +1 (blue)
- Annotate cells with correlation values

**Potential Insight**:
- Identify unexpected correlations
- Understand metric interdependencies
- Validate metric redundancy

---

### **Plot 16: Principal Component Analysis (PCA) Biplot**
**Type**: 2D PCA Scatter with Loading Vectors  
**Data**: All 42 experiments  
✅ **Available with current data**

**Implementation**:
- Axes: PC1 (x) and PC2 (y)
- Points: Experiments, colored by group
- Arrows: Original metric vectors (loadings)

**Potential Insight**:
- Reduce dimensionality of performance space
- Identify latent performance factors
- Group similar configurations

---

## 💡 **SECTION 8: Synthesis & Actionable Recommendations**
*Translate findings into practical guidance*

### **Plot 20: Recommendation Decision Tree**
**Type**: Flow Chart / Decision Diagram  
**Data**: Synthesized from all analyses  
✅ **Can be created from analysis results**

**Implementation**:
- Start: "What is your priority?"
- Branches: 
  - Efficiency → Greedy
  - Balanced → Sweet Spot configuration
  - Maximum Fairness → High Fairness configuration
- Nodes: Include actual parameter values and expected performance

**Potential Insight**:
- Actionable recommendations for practitioners
- Clear mapping from objective to configuration

---

## 📊 Plot Summary

**Total Plots**: 20 comprehensive visualizations organized into 8 thematic sections

**Data Availability**:
- ✅ **17 plots** can be implemented with current aggregate data
- ⚠️ **3 plots** (A, B, D) require temporal event-level data for future experiments
- ⚠️ **1 plot** (C) requires per-worker data

**Section Breakdown**:
1. **🔄 Temporal Evolution** (3 plots) - Requires future data collection
2. **📊 Distribution Analysis** (2 plots) - 1 available, 1 requires worker-level data
3. **⚖️ Trade-off & Pareto** (5 plots) - All available
4. **🎛️ Parameter Space** (5 plots) - All available  
5. **🎯 Multi-Metric Comparison** (3 plots) - All available
6. **🔧 System Diagnostics** (3 plots) - All available
7. **🧮 Statistical Analysis** (2 plots) - All available
8. **💡 Recommendations** (1 plot) - Synthesis from analysis

**Implementation Priority**:
- **High Priority** (Core findings): Plots 1-8 (Parameter space, trade-offs, Pareto frontier)
- **Medium Priority** (Validation): Plots 9-16 (Diagnostics, correlations, statistical tests)
- **Low Priority** (Synthesis): Plots 17-20 (Summary tables, decision trees)

**Estimated Visualization Time**: 2-3 hours for all available plots

---

## 🔑 Key Profiles (For Comparative Plots)

For any plot comparing "key configurations," use these 4:

1. **Greedy Baseline** (exp_001)
   - Strategy: greedy
   - Purpose: Efficiency reference

2. **Sweet Spot** (TBD from analysis)
   - Likely: L1=0.5, L2=0.8, L3=1.0, θ=0.5
   - Purpose: Best balance of fairness and efficiency

3. **High Fairness** (exp_031 or similar)
   - L1=2.0, L2=0.8, L3=0.5
   - Purpose: Maximum equity

4. **Starvation Ablation** (TBD - find L2=0.0 equivalent)
   - L1=0.5, L2=0.0, L3=1.0
   - Purpose: Test starvation component necessity
   - *Note*: May need to identify closest config if exact match not run

---

## 📊 Summary Statistics Tables

### **Table 1: Experimental Group Summary**
| Group | Description | Count | Mean JFI | Mean Wait Time | Mean TAR |
|-------|-------------|-------|----------|----------------|----------|
| A | Greedy | 1 | ... | ... | ... |
| B | L1×L3 Grid | 12 | ... | ... | ... |
| ... | ... | ... | ... | ... | ... |

### **Table 2: Best Performers by Objective**
| Objective | Exp ID | Config | JFI | Wait Time | TAR | Trade-offs |
|-----------|--------|--------|-----|-----------|-----|------------|
| Max JFI | ... | ... | ... | ... | ... | ... |
| Min Wait | ... | ... | ... | ... | ... | ... |
| Max TAR | ... | ... | ... | ... | ... | ... |
| Best Balance | ... | ... | ... | ... | ... | ... |

### **Table 3: Parameter Sensitivity Rankings**
Rank parameters by their effect size on each metric using regression or ANOVA.

---

## 🧪 Statistical Analyses

### **Analysis 1: ANOVA - Parameter Main Effects**
- **DV**: JFI, Wait Time, TAR
- **IVs**: λ₁, λ₂, λ₃, θ
- **Purpose**: Identify which parameters have statistically significant effects

### **Analysis 2: Regression - Predictive Model**
- **Model**: JFI ~ λ₁ + λ₂ + λ₃ + θ + interactions
- **Purpose**: Build predictive equation for JFI given parameters

### **Analysis 3: Effect Size Calculations**
- Cohen's d for Greedy vs. Best Composite
- Purpose: Quantify magnitude of improvement

---

## 🎯 Golden Nugget Hunting - Specific Questions

1. **Is there a λ₁/λ₃ ratio that consistently performs well?**
   - Test: Correlation between ratio and JFI

2. **Does the soft threshold still matter post-normalization?**
   - Compare Groups D, E, F (same L1/L3, different θ)

3. **Is λ₂ (starvation) redundant with λ₁ (fairness)?**
   - Compare Group H (varying L2) vs. similar configs with fixed L2

4. **What's the minimum fairness weight needed to beat Greedy's JFI?**
   - Find lowest λ₁ where JFI > Greedy's JFI

5. **Are there non-linear sweet spots?**
   - Look for discontinuities or peaks in heatmaps

6. **Does normalization have a performance cost?**
   - Compare duration across experiments

7. **What's the actual fairness-efficiency trade-off slope?**
   - Quantify: For every 1% increase in JFI, how many seconds of wait time?

---

## 📁 Data Requirements for Future Temporal Plots

To implement Plots A, B, and D in future experiments, we need:

### **Modification to DiagnosticTracker** (`metrics/diagnostic_tracker.py`)
Add temporal recording methods:
```python
def record_temporal_snapshot(self, timestamp, event_index, metrics_dict):
    """Record system state at regular intervals."""
    # metrics_dict contains: mean_wait_time, jfi, backlog, component_dominance
```

### **Modification to Simulation** (`simulator/simulation.py`)
Add periodic sampling:
```python
if event_index % SNAPSHOT_INTERVAL == 0:
    snapshot_metrics = compute_current_metrics()
    diagnostic_tracker.record_temporal_snapshot(now, event_index, snapshot_metrics)
```

### **Data Structure**
Output format: `exp_XXX_temporal_metrics.csv`
```
event_index,timestamp,mean_wait_time,jfi,backlog,fairness_dominant_pct,starvation_dominant_pct,utility_dominant_pct
0,2025-01-01 00:00:00,0.0,0.5,5,0.2,0.1,0.7
500,2025-01-01 01:30:00,2.3,0.65,8,0.35,0.15,0.5
...
```

### **For Worker-Level Data (Plot C)**
Save per-worker final state:
```
worker_id,total_idle_time_min,tasks_assigned,total_distance_km
W001,15.3,45,123.5
W002,8.7,52,145.2
...
```

---

## 📝 Implementation Checklist

- [ ] Load combined results CSV
- [ ] Identify key profiles (Sweet Spot, etc.)
- [ ] Generate all 20 plots
- [ ] Create summary statistics tables
- [ ] Perform statistical analyses (ANOVA, regression)
- [ ] Answer golden nugget questions
- [ ] Write findings summary
- [ ] Create recommendations section
- [ ] Document limitations (missing temporal data)
- [ ] Propose next steps

---

## 📋 Expected Deliverables

1. **Comprehensive Jupyter Notebook** with:
   - Executive summary
   - All 20 plots (where data permits)
   - Statistical analyses
   - Findings and insights
   - Recommendations

2. **Standalone Figures** (high-res exports):
   - Publication-ready versions of key plots
   - Saved in `exp_009.../figures/` directory

3. **Results Summary Markdown**:
   - Key findings in bullet points
   - Parameter recommendations
   - Comparison to previous experiments

4. **Next Steps Document**:
   - Proposed Experiment 010 (if needed)
   - Data collection improvements for temporal analysis

---

## 🎓 Analysis Principles

1. **Maximize Insight Density**: Every plot should answer a specific question
2. **Layered Complexity**: Start simple (univariate), build to complex (multivariate)
3. **Visual Hierarchy**: Most important findings in first 5 plots
4. **Statistical Rigor**: Support visual insights with quantitative tests
5. **Actionable Output**: End with clear recommendations for practitioners

---

**Status**: Ready to implement in Jupyter Notebook  
**Estimated Analysis Time**: 3-4 hours  
**Priority**: HIGH (last base experiment before advanced techniques)

