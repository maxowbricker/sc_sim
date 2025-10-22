# Experiment 013: High-Resolution Fairness-Efficiency Trade-off Mapping

**Status**: READY TO RUN  
**Scheduled**: Friday/Weekend, October 24-25, 2025  
**Duration**: ~8 hours (73 experiments × 6.6 minutes)

---

## Research Context

**Primary Question**: What is the precise shape of the fairness-efficiency trade-off curve when varying λ₁ (fairness) and λ₃ (utility)?

**Motivation**: 
- Map the Pareto frontier at high resolution
- Identify "sweet spots" where small weight changes yield large benefits
- Validate that λ₁=2.0, λ₃=1.0 from Exp 009 is indeed optimal
- Explore balance points where λ₁=λ₃ (equal weighting)
- Provide precise guidance for parameter selection in future work

**Key Innovation**: **High-resolution grid sweep** (10×7) focused on the promising region identified by Experiments 009 and 012.

---

## Experimental Design

### Fixed Parameters (All Experiments)
- **Dataset**: 4K workers, 20K tasks (validated optimal from Exp 012)
- **Sampling**: Stratified temporal sampling (12 bins)
- **λ₂ (Starvation)**: 0.5 (validated "safety net" from Exp 009)
- **θ (Soft Threshold)**: 0.0 (DISABLED - validated by Exp 011 & 012)
- **normalize_scores**: True (validated by Exp 008)
- **gamma**: 0.5 (EWMA smoothing)
- **enable_diagnostics**: False (fast path)

### Variable Parameters: λ₁ × λ₃ Grid Sweep

#### λ₁ (Fairness Weight) [10 values]
```
[2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0]
```
**Focus Region**: 2.5 to 5.0 (high-fairness configurations)

#### λ₃ (Utility Weight) [7 values]
```
[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
```
**Focus Region**: 0.5 to 2.0 (balanced to high-utility)

#### Grid: 10 × 7 = 70 Composite Experiments

---

## Special Configuration Points

### 1. Greedy Baseline [1 experiment]
- **Purpose**: Efficiency reference (maximum utility, zero fairness)
- **Config**: Pure greedy assignment strategy

### 2. Balance Points [2 experiments]
- **λ₁=2.5, λ₃=2.5**: Equal weighting (moderate)
- **λ₁=2.25, λ₃=2.75**: Near-equal weighting (slightly efficiency-biased)
- **Purpose**: Test hypothesis that equal weights provide balanced outcomes

---

## Experimental Matrix

### Group A: Greedy Baseline (1 experiment)

| Exp | λ₁ | λ₂ | λ₃ | Strategy | Purpose |
|-----|-----|-----|-----|----------|---------|
| 1 | N/A | N/A | N/A | greedy | Efficiency reference |

### Group B: High-Resolution λ₁ × λ₃ Grid (70 experiments)

**Full Grid**:

| λ₃ → | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 |
|------|-----|------|-----|------|-----|------|-----|
| **λ₁ ↓** | | | | | | | |
| 2.5  | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 | Exp 8 |
| 2.75 | Exp 9 | Exp 10 | Exp 11 | Exp 12 | Exp 13 | Exp 14 | Exp 15 |
| 3.0  | Exp 16 | Exp 17 | Exp 18 | Exp 19 | Exp 20 | Exp 21 | Exp 22 |
| 3.25 | Exp 23 | Exp 24 | Exp 25 | Exp 26 | Exp 27 | Exp 28 | Exp 29 |
| 3.5  | Exp 30 | Exp 31 | Exp 32 | Exp 33 | Exp 34 | Exp 35 | Exp 36 |
| 3.75 | Exp 37 | Exp 38 | Exp 39 | Exp 40 | Exp 41 | Exp 42 | Exp 43 |
| 4.0  | Exp 44 | Exp 45 | Exp 46 | Exp 47 | Exp 48 | Exp 49 | Exp 50 |
| 4.25 | Exp 51 | Exp 52 | Exp 53 | Exp 54 | Exp 55 | Exp 56 | Exp 57 |
| 4.5  | Exp 58 | Exp 59 | Exp 60 | Exp 61 | Exp 62 | Exp 63 | Exp 64 |
| 5.0  | Exp 65 | Exp 66 | Exp 67 | Exp 68 | Exp 69 | Exp 70 | Exp 71 |

### Group C: Balance Point Runs (2 experiments)

| Exp | λ₁ | λ₂ | λ₃ | Purpose |
|-----|-----|-----|-----|---------|
| 72 | 2.5 | 0.5 | 2.5 | Equal weights (moderate) |
| 73 | 2.25 | 0.5 | 2.75 | Near-equal (efficiency-biased) |

**Total: 73 experiments**

---

## Research Questions

### RQ1: Trade-off Curve Shape
- Is the Pareto frontier smooth or jagged?
- Are there "knee points" where marginal returns diminish sharply?
- Does the trade-off relationship change at different λ₁ values?

### RQ2: Optimal Region Identification
- Where on the λ₁ × λ₃ grid are the best-balanced configurations?
- Can we validate λ₁=2.0, λ₃=1.0 from Exp 009 is near-optimal?
- Are there better configurations in the tested region?

### RQ3: Balance Point Hypothesis
- Do equal weights (λ₁=λ₃) produce balanced fairness-efficiency outcomes?
- Is there a "golden ratio" of fairness to utility?

### RQ4: Sensitivity Analysis
- How sensitive is system performance to small changes in λ₁?
- How sensitive is system performance to small changes in λ₃?
- Which weight has more impact on outcomes?

### RQ5: Component Dominance
- At what λ₁ values does fairness dominate the composite score?
- At what λ₃ values does utility dominate?
- Is there a "balanced zone" where all components contribute equally?

---

## Hypotheses

**H1**: The Pareto frontier will be smooth and concave
- Rationale: Diminishing returns as weights increase

**H2**: λ₁ ≈ 2.5-3.5, λ₃ ≈ 0.75-1.25 will form the optimal region
- Rationale: Based on Exp 009 results, near λ₁=2.0, λ₃=1.0

**H3**: Equal weights (λ₁=λ₃) will NOT be optimal
- Rationale: Fairness and utility have different scales and impacts

**H4**: λ₁ has greater impact than λ₃ on outcomes
- Rationale: Fairness is the key contribution, utility is well-understood

**H5**: Gini coefficient will decrease monotonically with λ₁
- Rationale: Higher fairness weight → more equitable distribution

**H6**: Wait time will increase monotonically with λ₁
- Rationale: Prioritizing fairness trades off efficiency

---

## Success Criteria

✅ **All 73 experiments complete successfully**  
✅ **TAR >85% for all composite configurations** (validates 4K worker choice)  
✅ **Clear Pareto frontier observable in plots**  
✅ **Optimal region identified with confidence intervals**  
✅ **All 78+ v2.0 metrics collected for each experiment**  
✅ **Completion by Sunday evening** (ready for Monday analysis)

---

## Data Collection

### Per-Experiment Metrics (78+ total)
All metrics from DATA_DICTIONARY.md v2.0:

**Core Performance**:
- completed_tasks, task_assignment_ratio, jains_fairness_index

**Task Wait Distribution** (Tier 1):
- mean, std, p90, p95, max, CV

**Worker Idle Time Distribution** (Tier 1):
- mean, std, p90, max, CV

**Worker Task Distribution** (Tier 2):
- mean, std, CV, Gini, p10, p50, p90
- pct_workers_zero_tasks, pct_workers_single_task

**Worker Utilization** (Tier 2):
- mean, std, p10, p90

**Pickup Distance Distribution** (Tier 2):
- mean, std, p90, max

**Task Deferrals** (Tier 2):
- total, pct_tasks_deferred, mean_per_task, max_per_task

**Assignment Timing** (Tier 2):
- mean, std, p90 (assignment delay in seconds)

**System Metrics**:
- total_travel_km, empty_km_ratio, peak_backlog, ewma_cv

---

## Expected Insights

1. **Precise Pareto Frontier**: High-resolution map of fairness-efficiency trade-off
2. **Optimal Parameter Recommendation**: Validated λ₁ and λ₃ values with confidence
3. **Sensitivity Analysis**: Robustness of performance to parameter changes
4. **Balance Point Analysis**: Whether equal weights are effective
5. **Component Interaction**: How fairness and utility weights interact

---

## Analysis Plan

### Primary Visualizations
1. **2D Heatmap**: λ₁ (x-axis) vs λ₃ (y-axis), color = JFI
2. **2D Heatmap**: λ₁ vs λ₃, color = Mean Wait Time
3. **2D Heatmap**: λ₁ vs λ₃, color = Gini Coefficient
4. **Pareto Frontier**: JFI vs Mean Wait Time, color = λ₁/λ₃ ratio
5. **Contour Plot**: λ₁ vs λ₃ with TAR contours
6. **3D Surface**: λ₁ (x), λ₃ (y), JFI (z)
7. **Sensitivity Analysis**: Marginal change plots

### Key Comparisons
- Exp 009 best config (λ₁=2.0, λ₃=1.0) vs Exp 013 optimal
- Equal weight configs vs optimal configs
- Greedy baseline vs best composite

---

## Connection to Other Experiments

**Builds on Exp 012**: 
- Uses validated 4K workers / 20K tasks configuration
- Uses stratified temporal sampling
- Validates θ=0.0 and λ₂=0.5

**Refines Exp 009**:
- Higher resolution grid (10×7 vs original sweep)
- Focused on promising region (λ₁ ≥ 2.5)
- Validates best configuration from Exp 009

**Informs Future Work**:
- Provides precise parameter recommendations
- Establishes confidence intervals for robustness
- Identifies sensitivity of outcomes to weight changes

---

## Technical Notes

### Implementation
- Uses stratified temporal sampling from Exp 012
- Deep copies tasks/workers for each experiment (bug fix from Exp 012)
- Saves individual JSON files + aggregate CSV
- Random seed: 42 (reproducibility)

### Performance
- No diagnostics enabled (fast path)
- Estimated 6-7 min per experiment (4K workers)
- Can run unattended overnight
- Total runtime: ~8 hours

### Reproducibility
- Same 20K task sample for all experiments (stratified)
- Same 4K worker sample for all experiments (stratified)
- Fixed random seed (42)
- All parameters logged in results

---

## Risk Mitigation

**Risk**: 8-hour runtime may be interrupted  
**Mitigation**: Can resume from last completed experiment, individual JSON files saved

**Risk**: Some configurations may have low TAR (<85%)  
**Mitigation**: 4K workers validated in Exp 012, θ=0.0 ensures assignments

**Risk**: High-fairness configs (λ₁=5.0) may cause extreme wait times  
**Mitigation**: Expected behavior, will be analyzed in trade-off context

---

**Experiment designed by**: Max  
**Sampling strategy**: Stratified temporal with validated 4K workers  
**Target completion**: Sunday, October 26, 2025, 18:00




