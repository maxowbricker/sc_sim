# Experiment 011: Scalability Analysis Plan
## Worker Density Impact on Performance & Fairness

**Last Updated**: October 21, 2025  
**Status**: Ready for Implementation  
**Data**: 7 experiments (Fixed 20K tasks, varying worker counts)

---

## 🎯 Analysis Objectives

This experiment studies **scalability**: how system performance and fairness metrics change when we vary the number of workers serving a **fixed workload** of 20,000 tasks.

### Primary Research Questions
1. **RQ1 (Performance Scaling)**: How do efficiency metrics (wait time, TAR, pickup distance) improve with more workers?
2. **RQ2 (Fairness at Scale)**: Does fairness (JFI, Gini) improve, degrade, or remain stable as worker count increases?
3. **RQ3 (Distribution Metrics Validation)**: Do the new Tier 1 & 2 metrics reveal important distributional patterns?
4. **RQ4 (Optimal Worker Density)**: What is the ideal tasks-per-worker ratio for this configuration?
5. **RQ5 (Diminishing Returns)**: At what point does adding more workers yield minimal benefit?

---

## 📊 Experimental Design Summary

| Exp | Workers | Tasks | Tasks/Worker | Purpose |
|-----|---------|-------|--------------|---------|
| 1   | 2,000   | 20,000 | 10.0        | High worker scarcity |
| 2   | 4,000   | 20,000 | 5.0         | Moderate scarcity |
| 3   | 6,000   | 20,000 | 3.3         | Approaching balance |
| 4   | 8,000   | 20,000 | 2.5         | Balanced (baseline) |
| 5   | 10,000  | 20,000 | 2.0         | Moderate surplus |
| 6   | 12,000  | 20,000 | 1.7         | High surplus |
| 7   | 15,000  | 20,000 | 1.3         | Maximum surplus |

**Fixed Configuration**: λ₁=2.0, λ₂=0.8, λ₃=1.0, θ=0.5 (Balanced Pareto-efficient)

---

## 📊 Analysis Plots - Organized by Category

---

## 📈 **SECTION 1: Core Scalability Curves**
*Primary scaling relationships - the heart of the analysis*

### **Plot 1: Primary Scalability Dashboard (2x2 Grid)**
**Type**: 4-panel line plot grid  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- **Panel A (Top-Left)**: Mean Task Wait Time vs. Worker Count
  - X-axis: Worker count (2K to 15K)
  - Y-axis: Mean wait time (minutes)
  - Expected: Exponential decay curve
  
- **Panel B (Top-Right)**: Jain's Fairness Index vs. Worker Count
  - X-axis: Worker count
  - Y-axis: JFI (0-1)
  - Expected: Sigmoid curve stabilizing at high counts
  
- **Panel C (Bottom-Left)**: Task Assignment Ratio vs. Worker Count
  - X-axis: Worker count
  - Y-axis: TAR (%)
  - Expected: Rapid increase, plateau near 100%
  
- **Panel D (Bottom-Right)**: Mean Pickup Distance vs. Worker Count
  - X-axis: Worker count
  - Y-axis: Pickup distance (km)
  - Expected: Logarithmic decrease (diminishing returns)

**Annotations**:
- Mark "optimal" worker count (knee point)
- Add tasks/worker ratio as secondary x-axis labels

**Potential Insight**:
- At what worker count do we see diminishing returns?
- Is there a "sweet spot" for worker density?
- Does fairness require surplus capacity?

---

### **Plot 2: Tasks Per Worker Impact**
**Type**: Dual-axis scatter plot with trend lines  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Tasks/Worker ratio (10.0 → 1.3)
- Y-axis (Primary): Mean Wait Time (line + markers)
- Y-axis (Secondary): JFI (line + markers, different color)
- Invert x-axis so high scarcity (10.0) is on left
- Shade "optimal zone" (e.g., 2-4 tasks/worker)

**Potential Insight**:
- What's the optimal workload per worker?
- Does JFI stabilize at a specific ratio?
- Quantify the fairness-efficiency relationship at different densities

---

### **Plot 3: Computational Efficiency Scaling**
**Type**: Dual-axis line plot  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis (Primary): Runtime (minutes, bars)
- Y-axis (Secondary): Tasks completed per second (line, markers)
- Show: Does throughput scale linearly with workers?

**Potential Insight**:
- Is there a computational penalty for large worker pools?
- Does the simulation scale efficiently?
- Validate O(n log k) spatial indexing performance

---

## 📊 **SECTION 2: Distribution Analysis (New Metrics Showcase)**
*Leverage Tier 1 & 2 metrics to reveal hidden patterns*

### **Plot 4: Worker Task Distribution Evolution**
**Type**: Box plot with overlaid violin distributions  
**Data**: All 7 experiments  
✅ **Available with current data** (using new v2.0 metrics!)

**Implementation**:
- X-axis: Worker count
- Y-axis: Tasks per worker (actual distribution reconstructed from percentiles)
- Show: p10, p50, p90, mean (from `tasks_per_worker_*` metrics)
- Overlay: Violin plot showing full distribution shape
- Annotate: % workers with zero tasks, % with single task

**Metrics Used**:
- `tasks_per_worker_mean`, `tasks_per_worker_std`
- `tasks_per_worker_p10`, `tasks_per_worker_p50`, `tasks_per_worker_p90`
- `pct_workers_zero_tasks`, `pct_workers_single_task`

**Potential Insight**:
- Does worker surplus lead to more idle workers?
- At what scale does the "zero tasks" problem emerge?
- Is task distribution more equitable at higher densities?

---

### **Plot 5: Worker Inequality Metrics**
**Type**: Multi-line plot  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis: Inequality metric value
- Lines:
  - **Gini Coefficient** (tasks_per_worker_gini)
  - **Coefficient of Variation** (tasks_per_worker_cv)
  - **EWMA CV** (ewma_cv)
- Lower is better (more equitable)

**Potential Insight**:
- Does fairness improve (lower Gini) with more workers?
- Are Gini and CV correlated?
- Does EWMA CV tell a different story than task count Gini?

---

### **Plot 6: Wait Time Distribution Characteristics**
**Type**: Faceted line plots (small multiples)  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- 3 subplots stacked vertically:
  - **Subplot A**: Mean, Std Dev (tasks_wait_time metrics)
  - **Subplot B**: P90, P95, Max (percentile metrics)
  - **Subplot C**: Coefficient of Variation (cv_task_wait_time)
- X-axis: Worker count (shared across subplots)

**Potential Insight**:
- Does wait time variability decrease with more workers?
- At what scale do extreme waits (max) become rare?
- Is the system more predictable (lower CV) with surplus workers?

---

### **Plot 7: Pickup Distance Distribution**
**Type**: Bar chart with error bars  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis: Pickup distance (km)
- Bars: Mean pickup distance
- Error bars: ±1 std dev (from `std_pickup_distance_km`)
- Markers: P90 and Max values

**Metrics Used**:
- `mean_pickup_distance_km` (existing)
- `std_pickup_distance_km`, `p90_pickup_distance_km`, `max_pickup_distance_km` (new!)

**Potential Insight**:
- Does distance variability decrease with density?
- Are extreme pickups (max) common in sparse scenarios?

---

## ⚖️ **SECTION 3: Trade-off Analysis at Scale**
*How do competing objectives behave across worker densities?*

### **Plot 8: Fairness-Efficiency Frontier Across Scales**
**Type**: Scatter plot with connecting line  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Jain's Fairness Index
- Y-axis: Mean Task Wait Time (minutes)
- Points: 7 experiments, sized by worker count, colored by tasks/worker
- Connect with line showing trajectory
- Annotate: Worker counts at each point

**Potential Insight**:
- Does the fairness-efficiency trade-off curve change shape at different scales?
- Can we achieve both high fairness AND low wait time with enough workers?

---

### **Plot 9: Multi-Objective Performance Radar**
**Type**: Radar/spider chart  
**Data**: 3 key worker counts (2K, 8K, 15K)  
✅ **Available with current data**

**Metrics (Normalized 0-1)**:
1. JFI (higher is better)
2. TAR (higher is better)
3. Inverse Mean Wait Time (higher is better)
4. Inverse Pickup Distance (higher is better)
5. Worker Utilization (higher is better)
6. Inverse Gini Coefficient (higher = more equitable)

**Potential Insight**:
- How does the performance profile change from scarcity to surplus?
- Are some metrics more sensitive to scale than others?

---

## 🎯 **SECTION 4: Worker Utilization & Operational Efficiency**
*Understand resource usage patterns*

### **Plot 10: Worker Utilization Analysis**
**Type**: Stacked area chart + percentile bands  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis: Utilization (%)
- Show mean, p10, p90 (from `mean_worker_utilization`, `p10_worker_utilization`, `p90_worker_utilization`)
- Fill area between p10-p90 (utilization spread)

**Metrics Used** (NEW v2.0):
- `mean_worker_utilization`
- `std_worker_utilization`
- `p10_worker_utilization`, `p90_worker_utilization`

**Potential Insight**:
- At what worker count does utilization drop below 50%?
- Is utilization variance high in sparse scenarios?
- What's the "acceptable" utilization range?

---

### **Plot 11: Idle Workers vs. Idle Time**
**Type**: Scatter plot with quadrants  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: % Workers with Zero Tasks
- Y-axis: Mean Worker Idle Time (minutes)
- Points: 7 experiments, sized by worker count
- Quadrants: Mark "good" (low idle workers, low idle time) region

**Potential Insight**:
- Is having idle workers worse than having workers with low utilization?
- Does surplus capacity manifest as zero-task workers or partial idle time?

---

### **Plot 12: Worker Idle Time Distribution**
**Type**: Faceted line plots (small multiples)  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- 3 subplots stacked vertically:
  - **Subplot A**: Mean, Std Dev (mean/std_worker_idle_time_min)
  - **Subplot B**: P90, Max (p90/max_worker_idle_time_min)
  - **Subplot C**: Coefficient of Variation (cv_worker_idle_time)
- X-axis: Worker count (shared across subplots)
- Mirror structure of Plot 6 (Wait Time Distribution)

**Metrics Used** (NEW v2.0):
- `mean_worker_idle_time_min`
- `std_worker_idle_time_min`
- `p90_worker_idle_time_min`
- `max_worker_idle_time_min`
- `cv_worker_idle_time`

**Potential Insight**:
- Does idle time variability INCREASE with more workers? (Paradoxical!)
- At high worker counts, do some workers get severely starved (high p90/max)?
- Is idle time more equitable (lower CV) in balanced scenarios?
- Compare to utilization metrics: does low mean idle = high utilization?

**Key Question**: 
- With 15K workers serving 20K tasks, do we see bimodal distribution (busy vs. completely idle)?

---

### **Plot 13: Travel Efficiency at Scale**
**Type**: Grouped bar chart  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Bars (grouped):
  - Total Travel Distance (km)
  - Empty-KM Ratio (%)
  - Mean Pickup Distance (km)

**Potential Insight**:
- Does total system travel decrease with more workers?
- Does empty-km ratio improve (closer workers available)?

---

## 🔢 **SECTION 5: Task Assignment Dynamics**
*New Tier 2 metrics in action*

### **Plot 14: Task Deferral Analysis**
**Type**: Dual-axis bar + line chart  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis (Primary): Total Deferrals (bars)
- Y-axis (Secondary): % Tasks Deferred (line with markers)
- Annotate: Mean deferrals per task, max deferrals per task

**Metrics Used** (NEW v2.0):
- `total_deferrals`
- `pct_tasks_deferred`
- `mean_deferrals_per_task`
- `max_deferrals_per_task`

**Potential Insight**:
- Do deferrals drop dramatically with more workers?
- Is deferral rate a good proxy for system stress?
- At what worker count do deferrals become negligible?

---

### **Plot 15: Assignment Timing Characteristics**
**Type**: Box plot equivalent (from percentile data)  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- X-axis: Worker count
- Y-axis: Assignment delay (seconds)
- Show: Mean, std, p90 (from `mean_assignment_delay_sec`, `std_assignment_delay_sec`, `p90_assignment_delay_sec`)

**Metrics Used** (NEW v2.0):
- `mean_assignment_delay_sec`
- `std_assignment_delay_sec`
- `p90_assignment_delay_sec`

**Potential Insight**:
- How quickly does the system respond to new tasks at different scales?
- Is assignment delay a bottleneck in sparse scenarios?

---

## 🧮 **SECTION 6: Statistical Analysis**
*Quantify relationships and validate findings*

### **Plot 16: Correlation Heatmap (Key Metrics)**
**Type**: Correlation matrix heatmap  
**Data**: All 7 experiments, subset of key metrics  
✅ **Available with current data**

**Metrics**:
- Worker count, Tasks/Worker ratio
- JFI, Gini, TAR
- Mean Wait Time, P90 Wait Time
- Mean Pickup Distance, Empty-KM Ratio
- Worker Utilization, % Zero Tasks
- Total Deferrals, Mean Deferrals/Task

**Potential Insight**:
- Are Gini and JFI redundant?
- Is worker utilization inversely correlated with worker count?
- Validate expected relationships (more workers → lower wait time)

---

### **Plot 17: Regression Analysis - Predictive Curves**
**Type**: Scatter + fitted curves  
**Data**: All 7 experiments  
✅ **Can be computed from current data**

**Implementation**:
- Multiple subplots (2x2):
  - **A**: Wait Time ~ Worker Count (exponential decay fit)
  - **B**: JFI ~ Worker Count (sigmoid fit)
  - **C**: Pickup Distance ~ Worker Count (logarithmic fit)
  - **D**: TAR ~ Worker Count (logistic fit)
- Show: Data points, fitted curve, R² value, equation

**Potential Insight**:
- Can we predict performance at untested worker counts?
- Validate exponential/logarithmic scaling assumptions
- Identify which metrics are most predictable

---

## 📉 **SECTION 7: Diminishing Returns & Threshold Analysis**
*Find optimal operating points*

### **Plot 18: Marginal Benefit Analysis**
**Type**: Bar chart showing deltas  
**Data**: Computed from differences between consecutive experiments  
✅ **Can be computed from current data**

**Implementation**:
- X-axis: Worker count increase (2K→4K, 4K→6K, etc.)
- Y-axis: Marginal improvement (% change)
- Bars (grouped):
  - Reduction in mean wait time
  - Increase in JFI
  - Reduction in deferrals
- Highlight: Point of diminishing returns (smallest bar)

**Potential Insight**:
- At what point does adding 2K workers yield < 10% improvement?
- Which metrics show diminishing returns first?
- What's the "minimum viable" worker count for good performance?

---

### **Plot 19: Threshold Identification Dashboard**
**Type**: Annotated line plots with shaded regions  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- 4 subplots showing key metrics vs. worker count
- Shade "acceptable performance" regions:
  - Wait Time < 5 minutes
  - JFI > 0.7
  - TAR > 95%
  - Worker Utilization 40-80%
- Annotate: First worker count meeting all thresholds

**Potential Insight**:
- What's the minimum worker count for production deployment?
- How many workers do we need for "excellent" performance?

---

## 📊 **SECTION 8: Comprehensive Summary**
*Synthesis and recommendations*

### **Plot 20: Performance Tier Matrix**
**Type**: Heatmap table  
**Data**: All 7 experiments  
✅ **Available with current data**

**Implementation**:
- Rows: 7 experiments (worker counts)
- Columns: Key metrics
- Cells: Color-coded by performance tier
  - 🟢 Green: Excellent
  - 🟡 Yellow: Acceptable
  - 🔴 Red: Poor
- Define thresholds for each metric

**Potential Insight**:
- At-a-glance performance summary
- Identify "balanced" worker counts (all green)

---

### **Plot 21: Recommendation Decision Tree**
**Type**: Flow chart / infographic  
**Data**: Synthesized from all analyses  
✅ **Can be created from analysis results**

**Implementation**:
- Decision nodes:
  - "What are your constraints?"
    - Cost-constrained → Minimum viable (first acceptable count)
    - Performance-priority → High surplus (15K+)
    - Balanced → Optimal knee point (likely 6K-8K)
- Include: Expected performance metrics at each recommendation

**Potential Insight**:
- Clear guidance for practitioners
- Quantify cost vs. performance trade-off

---

## 📊 Plot Summary

**Total Plots**: 21 comprehensive visualizations organized into 8 thematic sections

**Data Availability**:
- ✅ **ALL 21 plots** can be implemented with collected data
- ✅ Leverages all 78+ metrics including new Tier 1 & 2 distributions
- ✅ **Complete coverage** of v2.0 metric suite (including idle time distribution)

**Section Breakdown**:
1. **📈 Core Scalability** (3 plots) - Primary scaling relationships
2. **📊 Distribution Analysis** (4 plots) - Showcase new metrics
3. **⚖️ Trade-off Analysis** (2 plots) - Multi-objective at scale
4. **🎯 Worker Utilization** (4 plots) - Resource efficiency (now includes idle time!)
5. **🔢 Task Assignment** (2 plots) - Assignment dynamics
6. **🧮 Statistical Analysis** (2 plots) - Correlations & predictions
7. **📉 Diminishing Returns** (2 plots) - Optimal thresholds
8. **📊 Summary** (2 plots) - Synthesis & recommendations

**Implementation Priority**:
- **High Priority** (Core insights): Plots 1-5, 12, 14, 18, 19 (scalability curves, distributions, diminishing returns)
- **Medium Priority** (Validation): Plots 6-13, 16-17 (detailed metrics, statistics)
- **Low Priority** (Synthesis): Plots 20-21 (summary tables, recommendations)

**Estimated Visualization Time**: 2-3 hours for all plots

---

## 🔑 Key Findings to Validate

Based on expected behavior, confirm or refute:

1. **Exponential Decay**: Wait time should decrease exponentially with worker count
2. **Logarithmic Distance**: Pickup distance follows log(workers) decrease (spatial coverage)
3. **Sigmoid Fairness**: JFI improves rapidly initially, plateaus at surplus
4. **Utilization Inversely Proportional**: Utilization ≈ (tasks × avg_duration) / (workers × time)
5. **Diminishing Returns**: Adding workers beyond 10K yields < 5% improvement

---

## 🎯 Golden Nugget Questions

1. **What is the optimal tasks/worker ratio for this configuration?**
   - Likely answer: 2-4 tasks/worker (balanced workload)

2. **At what worker count does TAR exceed 99%?**
   - Measure: First experiment with TAR > 0.99

3. **Does fairness (Gini) improve monotonically with worker count?**
   - Test: Is Gini strictly decreasing?

4. **Is there a "saturation point" where more workers don't help?**
   - Look for: Flat regions in scalability curves

5. **Do idle workers (zero tasks) appear only at high densities?**
   - Compare: `pct_workers_zero_tasks` across experiments

6. **What's the computational cost of large worker pools?**
   - Analyze: Runtime vs. worker count relationship

7. **Are deferrals a good early warning signal?**
   - Correlate: Deferral rate with overall performance

8. **Does the new Gini coefficient match JFI trends?**
   - Validate: Correlation between `tasks_per_worker_gini` and `jfi`

9. **Is assignment delay negligible at all scales?**
   - Check: `mean_assignment_delay_sec` across experiments

10. **Does idle time inequality INCREASE with surplus workers?**
    - Compare: `cv_worker_idle_time` at 2K vs 15K workers
    - Expected: Higher CV at high worker counts (more idle workers with zero tasks)

11. **What worker count minimizes total system cost (wait time × tasks + idle time × workers)?**
    - Compute: Cost function for each experiment

---

## 📈 Expected Scaling Laws

Document empirical relationships:

### **Wait Time Scaling**
\( \text{Wait Time} \approx a \cdot e^{-b \cdot \text{Workers}} + c \)

### **JFI Scaling**
\( \text{JFI} \approx \frac{L}{1 + e^{-k(\text{Workers} - x_0)}} \)

### **Pickup Distance Scaling**
\( \text{Distance} \approx a - b \cdot \log(\text{Workers}) \)

**Goal**: Fit these models and extract parameters \(a, b, c, k, L, x_0\)

---

## 📋 Statistical Analyses

### **Analysis 1: Spearman Rank Correlation**
- **Variables**: All metrics vs. worker count
- **Purpose**: Confirm monotonic relationships
- **Expected**: Strong negative correlation for wait time, positive for JFI/TAR

### **Analysis 2: Non-Linear Regression**
- **Models**: Exponential, logarithmic, sigmoid fits
- **Purpose**: Predictive equations for untested scales
- **Validate**: R² > 0.95 for primary metrics

### **Analysis 3: Diminishing Returns Quantification**
- **Metric**: Marginal benefit per 2K workers
- **Purpose**: Identify optimal stopping point
- **Threshold**: < 5% improvement

### **Analysis 4: Cost-Benefit Analysis**
- **Assumptions**: 
  - Worker cost: $X per hour
  - Task wait cost: $Y per minute
- **Calculate**: Optimal worker count minimizing total cost

---

## 📁 Data Structure

**Input**: `exp_011_TIMESTAMP/experiment_011_results.csv`

**Key Columns**:
- `experiment_id`, `worker_count`, `task_count`, `tasks_per_worker_ratio`
- Core metrics: `jains_fairness_index`, `task_assignment_ratio`, `mean_task_wait_time_min`
- Distribution metrics: `tasks_per_worker_gini`, `cv_task_wait_time`, `std_pickup_distance_km`
- Utilization: `mean_worker_utilization`, `pct_workers_zero_tasks`
- Deferrals: `total_deferrals`, `mean_deferrals_per_task`
- Assignment: `mean_assignment_delay_sec`

---

## 📝 Implementation Checklist

- [ ] Load experiment results CSV
- [ ] Verify all 7 experiments completed successfully
- [ ] Generate Section 1: Core Scalability Curves (Plots 1-3)
- [ ] Generate Section 2: Distribution Analysis (Plots 4-7)
- [ ] Generate Section 3: Trade-off Analysis (Plots 8-9)
- [ ] Generate Section 4: Worker Utilization (Plots 10-13) - **includes new idle time plot!**
- [ ] Generate Section 5: Task Assignment Dynamics (Plots 14-15)
- [ ] Generate Section 6: Statistical Analysis (Plots 16-17)
- [ ] Generate Section 7: Diminishing Returns (Plots 18-19)
- [ ] Generate Section 8: Summary (Plots 20-21)
- [ ] Fit scaling law equations
- [ ] Perform statistical tests
- [ ] Answer golden nugget questions
- [ ] Create summary table
- [ ] Write recommendations section

---

## 📋 Expected Deliverables

1. **Comprehensive Jupyter Notebook** (`analysis.ipynb`) with:
   - Executive summary
   - All 20 plots
   - Fitted scaling equations
   - Statistical test results
   - Findings and recommendations

2. **Standalone Figures** (high-res exports):
   - Key plots for presentations
   - Saved in `exp_011.../figures/` directory

3. **Results Summary** (`RESULTS.md`):
   - Optimal worker count recommendation
   - Scaling law equations
   - Diminishing returns analysis
   - Cost-benefit guidance

4. **Comparison to Exp 009** (`COMPARISON.md`):
   - How does the Balanced config perform at different scales?
   - Validate that chosen parameters are robust across scales

---

## 🎓 Analysis Principles

1. **Focus on Scalability**: Every plot should illuminate how metrics change with scale
2. **Showcase New Metrics**: Highlight the value of Tier 1 & 2 distribution statistics
3. **Practical Recommendations**: End with actionable guidance on worker provisioning
4. **Quantitative Rigor**: Fit equations, not just trends
5. **Cost-Aware**: Consider operational costs, not just performance

---

**Status**: Ready to implement in Jupyter Notebook  
**Estimated Analysis Time**: 2-3 hours  
**Priority**: HIGH (validates scalability of fairness approach)  
**Unique Value**: First experiment leveraging **COMPLETE v2.0 metric suite** (all 78+ metrics including idle time distribution)

