# Experiment 012: Worker-to-Task Ratio Analysis Plan
## Finding the Optimal Worker Count

**Last Updated**: October 22, 2025  
**Status**: Ready for Analysis  
**Data**: 11 experiments (Fixed 20K tasks, 2K-15K workers)

---

## 🎯 Primary Goal

**Find the optimal number of workers for a 20K task workload** - balancing:
1. High Task Assignment Ratio (TAR >90%)
2. Meaningful tasks-per-worker for fairness tracking
3. Realistic wait times and worker utilization
4. Computational efficiency

---

## 📊 Experiment Summary

| Exp | Workers | Tasks/Worker | TAR | Status |
|-----|---------|--------------|-----|--------|
| 1 | 2K | 10.0 | 94.2% | ✅ SUCCESS |
| 2 | 3K | 6.7 | 94.3% | ✅ SUCCESS |
| 3 | 4K | 5.0 | 94.3% | ✅ SUCCESS |
| 4 | 5K | 4.0 | 94.3% | ✅ SUCCESS |
| 5 | 6K | 3.3 | 94.3% | ✅ SUCCESS |
| 6 | 7K | 2.9 | 94.3% | ✅ SUCCESS |
| 7 | 8K | 2.5 | 94.3% | ✅ SUCCESS |
| 8 | 9K | 2.2 | 94.3% | ✅ SUCCESS |
| 9 | 10K | 2.0 | 94.3% | ✅ SUCCESS |
| 10 | 12K | 1.7 | 94.3% | ✅ SUCCESS |
| 11 | 15K | 1.3 | 94.3% | ✅ SUCCESS |

**All experiments succeeded with ~94% TAR!** 🎉

**Configuration**: λ₁=2.0, λ₂=0.5, λ₃=1.0, θ=0.0, Stratified Temporal Sampling

---

## 📈 Analysis Sections & Plots

### **SECTION 1: The Optimal Worker Count Decision** ⭐
*Primary decision-making plots*

#### **Plot 1: The Knee Plot - TAR & Tasks/Worker (Dual-Axis)**
**Type**: Dual-axis line plot  
**Purpose**: THE key plot for optimal worker recommendation

**Implementation**:
- X-axis: Worker count (2K to 15K)
- Y-axis (Left): TAR (%) - line with markers
- Y-axis (Right): Tasks/Worker ratio - line with markers (different color)
- Horizontal line: 90% TAR threshold
- Shaded region: "Optimal zone" recommendation
- Annotations: Mark recommended worker count

**Decision Rule**:
- Find MINIMUM worker count where TAR >90%
- While maintaining tasks/worker >2.0 (meaningful fairness tracking)

**Expected Insight**:
- If TAR is flat at ~94% across all counts → Choose LOWEST (2K) for max tasks/worker
- If TAR increases with workers → Find knee of curve
- **This plot answers RQ1 directly**

---

#### **Plot 2: Multi-Objective Performance Dashboard (2×2 Grid)**
**Type**: 4-panel line plots  
**Purpose**: Show how all key metrics change with worker count

**Panels**:
- **A (Top-Left)**: Mean Wait Time vs Worker Count
- **B (Top-Right)**: Jain's Fairness Index (JFI) vs Worker Count
- **C (Bottom-Left)**: Gini Coefficient vs Worker Count
- **D (Bottom-Right)**: Mean Worker Utilization vs Worker Count

**Annotations**: Mark optimal worker count on all panels

**Expected Insight**:
- Does wait time decrease significantly with more workers?
- Does fairness (JFI/Gini) improve or degrade?
- What is the utilization at the optimal worker count?

---

### **SECTION 2: Efficiency & Fairness Trade-offs**
*Understanding the cost of different worker counts*

#### **Plot 3: Fairness vs Efficiency Scatter**
**Type**: Scatter plot with connecting line  
**Purpose**: Visualize the fundamental trade-off

**Implementation**:
- X-axis: Gini Coefficient (fairness, lower is better)
- Y-axis: Mean Wait Time (efficiency, lower is better)
- Points: 11 experiments, sized by worker count
- Annotate: Worker count labels
- Mark: Pareto frontier (bottom-left is best)

**Expected Insight**:
- Can we achieve BOTH low Gini AND low wait time?
- Or is there an inherent trade-off?
- Where is the "sweet spot"?

---

#### **Plot 4: Worker Utilization vs Density**
**Type**: Line plot with percentile bands  
**Purpose**: Understand resource efficiency

**Implementation**:
- X-axis: Worker count
- Y-axis: Worker utilization (%)
- Lines: Mean, P10, P90
- Fill: Shaded area between P10-P90
- Horizontal lines: 40% and 80% (acceptable range)

**Expected Insight**:
- At what worker count does utilization drop below 40%?
- Is low utilization acceptable for better fairness?

---

### **SECTION 3: Distribution Metrics Deep Dive**
*Leveraging Tier 1 & 2 metrics*

#### **Plot 5: Worker Task Distribution (Inequality)**
**Type**: Multi-line plot  
**Purpose**: Show how task distribution inequality changes

**Implementation**:
- X-axis: Worker count
- Y-axis: Inequality metric
- Lines:
  - Gini Coefficient (tasks_per_worker_gini)
  - Coefficient of Variation (tasks_per_worker_cv)
  - % Workers with Zero Tasks (pct_workers_zero_tasks)
- Lower is better (more equitable)

**Expected Insight**:
- Does Gini have a U-shape? (Hypothesis: lowest at 6K-8K)
- At what worker count do we see significant zero-task workers?

---

#### **Plot 6: Wait Time Distribution Comparison**
**Type**: Box plot equivalent (from percentiles)  
**Purpose**: Show wait time variability across scales

**Implementation**:
- X-axis: Worker count
- Y-axis: Task wait time (minutes)
- Show: Mean, P90, P95, Max (as error bars/whiskers)
- Overlay: Coefficient of Variation as text labels

**Expected Insight**:
- Does wait time variability decrease with more workers?
- Are extreme waits (P95, Max) reduced at higher densities?

---

### **SECTION 4: Operational Metrics**
*System behavior insights*

#### **Plot 7: Deferrals & Assignment Dynamics**
**Type**: Dual-axis bar + line  
**Purpose**: Understand assignment stress

**Implementation**:
- X-axis: Worker count
- Y-axis (Left): % Tasks Deferred (bars)
- Y-axis (Right): Mean Deferrals per Task (line)
- Annotate: Max deferrals per task (as text)

**Expected Insight**:
- Do deferrals decrease significantly with more workers?
- Or is deferral rate stable across all counts (indicating robust θ=0.0)?

---

#### **Plot 8: Computational Performance**
**Type**: Bar chart with overlay  
**Purpose**: Validate scalability hypothesis

**Implementation**:
- X-axis: Worker count
- Y-axis (Left): Runtime (minutes) - bars
- Y-axis (Right): Tasks completed per minute - line
- Show: Linear reference line for comparison

**Expected Insight**:
- Does runtime scale sub-linearly (as hypothesized)?
- Is there a computational bottleneck at high worker counts?

---

### **SECTION 5: Detailed Comparisons**
*Side-by-side analysis of key scenarios*

#### **Plot 9: Three Scenarios Comparison (Radar Chart)**
**Type**: Radar/spider chart  
**Purpose**: Compare performance profiles

**Scenarios**:
- **High Demand** (2K workers, 10 tasks/worker)
- **Balanced** (6K workers, 3.3 tasks/worker) 
- **Oversupply** (15K workers, 1.3 tasks/worker)

**Metrics (Normalized 0-1)**:
1. TAR (higher is better)
2. JFI (higher is better)
3. Inverse Mean Wait Time (higher is better)
4. Worker Utilization (higher is better)
5. Inverse Gini (higher = more equitable)
6. Tasks/Worker ratio (higher is better)

**Expected Insight**:
- Which scenario is most balanced across all objectives?
- Are there clear winners for specific use cases?

---

### **SECTION 6: Statistical Analysis & Recommendations**

#### **Plot 10: Correlation Heatmap**
**Type**: Correlation matrix  
**Purpose**: Validate expected relationships

**Variables**:
- Worker count, Tasks/Worker ratio
- TAR, JFI, Gini
- Mean Wait Time, Mean Pickup Distance
- Worker Utilization, % Zero Tasks
- Deferrals per Task

**Expected Insight**:
- Confirm: Worker count negatively correlated with utilization
- Confirm: More workers → lower wait time
- Validate: Gini and JFI inversely correlated

---

#### **Plot 11: Marginal Benefit Analysis**
**Type**: Bar chart of deltas  
**Purpose**: Quantify diminishing returns

**Implementation**:
- X-axis: Worker count increase (2K→3K, 3K→4K, ..., 12K→15K)
- Y-axis: % Improvement
- Bars (grouped):
  - Reduction in Mean Wait Time
  - Improvement in JFI
  - Reduction in Gini
- Horizontal line: 5% improvement threshold

**Expected Insight**:
- At what point does adding workers yield <5% improvement?
- Which metrics show diminishing returns first?

---

### **SECTION 7: Recommendation & Summary**

#### **Summary Table: Performance by Worker Count**
**Type**: Formatted table with color coding  
**Purpose**: At-a-glance comparison

**Columns**:
- Worker Count
- Tasks/Worker
- TAR
- Mean Wait Time
- JFI
- Gini
- Worker Utilization
- Runtime
- **Recommendation Tier** (🟢/🟡/🔴)

**Tiers**:
- 🟢 **Excellent**: TAR >90%, Utilization 40-80%, Gini <0.3, Tasks/Worker >2.5
- 🟡 **Good**: TAR >85%, Utilization 30-85%, Gini <0.4, Tasks/Worker >1.5
- 🔴 **Poor**: Otherwise

---

#### **Decision Tree: Choosing Worker Count**
**Type**: Text flowchart / decision guide  
**Purpose**: Actionable recommendation

**Decision Nodes**:
1. **What is your priority?**
   - **Fairness Accuracy** → Choose 6K-8K (high tasks/worker, low Gini)
   - **Lowest Wait Time** → Choose 12K-15K (max workers)
   - **Cost Efficiency** → Choose minimum achieving >90% TAR
   - **Balanced** → Choose knee of curve

2. **What are your constraints?**
   - **Budget Limited** → Minimum viable (2K-3K)
   - **Performance Priority** → High surplus (10K-15K)
   - **Realistic Utilization** → 40-60% range (6K-8K)

---

## 📋 Key Research Questions Addressed

**RQ1: Optimal Worker Count**
- **Plots**: 1, 2, 11 (Knee plot, Dashboard, Marginal benefit)
- **Answer**: [To be determined from analysis]

**RQ2: Fairness Accuracy vs Density**
- **Plots**: 5, 9 (Inequality metrics, Radar chart)
- **Answer**: [To be determined]

**RQ3: Wait Time Realism**
- **Plots**: 2A, 6 (Wait time mean, distribution)
- **Answer**: [To be determined]

**RQ4: Worker Utilization**
- **Plots**: 2D, 4 (Utilization dashboard, percentiles)
- **Answer**: [To be determined]

**RQ5: System Scalability**
- **Plots**: 8 (Computational performance)
- **Answer**: [To be determined]

---

## 🎯 Expected Findings

**Hypothesis 1**: TAR is flat at ~94% across all counts (CONFIRMED in preliminary results)
- **Implication**: No TAR-based constraint, choose for other reasons

**Hypothesis 2**: Optimal worker count is **6K-8K**
- **Rationale**: 2.5-3.3 tasks/worker, moderate utilization, good fairness

**Hypothesis 3**: Gini has a U-shape minimum around 6K-8K
- **Test**: Plot 5 (Worker Task Distribution)

**Hypothesis 4**: Wait time decreases logarithmically
- **Test**: Plot 2A (Mean Wait Time vs Worker Count)

**Hypothesis 5**: Utilization decreases linearly
- **Test**: Plot 2D, 4 (Utilization plots)

**Hypothesis 6**: Runtime scales sub-linearly
- **Test**: Plot 8 (Computational Performance)

---

## 📊 Data Quality Checks

Before analysis, verify:
- ✅ All 11 experiments completed successfully
- ✅ TAR >85% for all experiments
- ✅ No missing values in key metrics
- ✅ Worker counts match expected (2K, 3K, 4K, 5K, 6K, 7K, 8K, 9K, 10K, 12K, 15K)

---

## 📁 Implementation Notes

**Input Data**: `experiment_012_aggregate_results.csv`

**Output**:
1. **Jupyter Notebook** (`analysis.ipynb`) with:
   - Executive summary
   - 11 plots organized into 7 sections
   - Statistical tests
   - **Clear recommendation** with justification

2. **Key Figures** (exports):
   - Plot 1 (Knee plot) - for presentations
   - Plot 2 (Dashboard) - comprehensive overview
   - Summary table - for quick reference

3. **Results Summary** (`RESULTS.md`):
   - Optimal worker count recommendation
   - Performance trade-offs
   - Deployment guidance

---

## 🎓 Analysis Principles

1. **Decision-Focused**: Every plot should inform the optimal worker count decision
2. **Practical**: Provide clear, actionable recommendations
3. **Trade-off Aware**: Acknowledge that "optimal" depends on priorities
4. **Evidence-Based**: Support recommendations with quantitative analysis

---

**Total Plots**: 11 focused visualizations + 1 summary table  
**Priority**: HIGH (directly informs deployment decisions)  
**Estimated Analysis Time**: 1-1.5 hours  
**Status**: READY TO IMPLEMENT




