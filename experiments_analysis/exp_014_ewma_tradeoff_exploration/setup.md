# Experiment 014: EWMA & Trade-Off Exploration

## Overview

Comprehensive experiment to validate EWMA fairness metric and map the fairness-efficiency trade-off space under realistic constraints. This experiment addresses multiple research questions with efficient simulation design.

**Experiment Date**: October 23, 2025  
**Status**: Setup Complete

---

## Research Questions Addressed

- **RQ1**: What is the fairness-efficiency trade-off in spatial crowdsourcing?
- **RQ2.1**: Does EWMA provide better fairness tracking than simpler metrics?
- **RQ2.2**: How sensitive is the EWMA metric to gamma (γ) parameter?
- **RQ2.4**: Does EWMA converge within realistic task windows?

---

## Experimental Design

### Dataset Configuration
- **Workers**: 4,000 (validated optimal from Exp 012)
- **Tasks**: 20,000
- **Time Window**: 3-hour peak (Didi dataset)
- **Task Expiry**: **15 minutes** (realistic ride-sharing constraint)
- **Sampling**: Stratified temporal sampling (12 bins)

### Fixed Parameters (Validated Robust Baseline)
- **Soft Threshold (θ)**: 0.0 (disabled, validated by Exp 11 & 12)
- **Starvation Weight (λ₂)**: 0.5 (validated "safety net" value)
- **Normalization**: `normalize_scores=True`
- **K-Nearest**: 15 workers

---

## Simulation Plan (43 Total)

### Group 1: Baseline Strategies (3 simulations)

Establish reference points for pure strategy approaches.

| ID | Strategy | Description | Key Metrics |
|----|----------|-------------|-------------|
| 001 | Greedy | Pure distance-based (utility only) | Efficiency baseline |
| 002 | LAF | Least allocated first (simple fairness) | Fairness baseline |
| 003 | EWMA-Only | Time-weighted fairness (γ=0.5) | Advanced fairness baseline |

**Purpose**: Validate that composite strategy outperforms single-objective approaches.

---

### Group 2: Pareto Sweep - Fairness vs Utility (25 simulations)

Map the fairness-efficiency trade-off frontier under realistic constraints.

**Grid Design**: 5×5 parameter sweep

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| **λ₁ (Fairness)** | [2.5, 3.0, 3.5, 4.0, 4.5] | Exp 013 showed optimal ~2.5-4.0 |
| **λ₃ (Utility)** | [0.5, 1.0, 1.5, 2.0, 2.5] | Expanded range for complete mapping |

**Fixed**: λ₂=0.5, θ=0.0, γ=0.5

**Simulation IDs**: 004-028 (25 simulations)

**Expected Outcomes**:
- High λ₁, low λ₃: High fairness (JFI ~0.95), higher wait times
- Low λ₁, high λ₃: Lower fairness (JFI ~0.85), lower wait times
- Balanced λ₁≈λ₃: Optimal trade-off (JFI ~0.92, acceptable wait)

**Analysis Goals**:
1. Identify Pareto frontier
2. Quantify fairness-efficiency trade-off slope
3. Find "knee point" for optimal balance
4. Compare to baselines (Greedy, LAF, EWMA-Only)

---

### Group 3: EWMA Gamma (γ) Sensitivity (15 simulations)

Test whether EWMA smoothing parameter affects system performance under realistic constraints.

**Gamma Values**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Low γ (0.1)**: Responsive - emphasizes recent idle time
- **Medium γ (0.5)**: Balanced - default value
- **High γ (0.9)**: Smooth - more historical weight

**Test Configurations** (3 representative composite setups):

| Config | λ₁ (Fairness) | λ₃ (Utility) | Description |
|--------|---------------|--------------|-------------|
| **A (Balanced)** | 3.5 | 1.0 | Balanced fairness-utility |
| **B (High Fairness)** | 4.5 | 0.5 | Fairness-prioritized |
| **C (Efficiency Leaning)** | 2.5 | 2.0 | Utility-prioritized |

**Fixed**: λ₂=0.5, θ=0.0

**Simulation IDs**: 029-043 (15 simulations)

**Expected Outcomes**:
- γ should have **minimal impact** on TAR, JFI (based on Exp 007)
- EWMA metric is robust across γ values
- Final fairness distribution independent of γ choice

**Analysis Goals**:
1. Confirm EWMA robustness to γ parameter (RQ2.2)
2. Validate default γ=0.5 is reasonable
3. Show EWMA converges within 3-hour window (RQ2.4)

---

## Key Metrics to Collect

### Primary Metrics (All Simulations)
- Task Assignment Ratio (TAR)
- Jain's Fairness Index (JFI) - **final value**
- Gini Coefficient
- Mean Task Wait Time
- P95 Task Wait Time
- Mean Worker Utilization
- % Workers with 0 Tasks
- EWMA Fairness Value - **final value**

### Distribution Metrics (v2.0)
- Tasks per worker: P10, P50, P90, Mean, Std, CV
- Wait times: P10, P50, P95, Mean, Std
- Worker utilization: P10, Mean, P90, Std
- Idle times: P10, P50, P90, Mean, Std

### Optional (If Feasible)
- EWMA values over time (every 100 assignments) for convergence analysis (RQ2.4)
- Fairness metrics time series for trajectory analysis

---

## Expected Runtime

- **Simulations**: 43 total
- **Time per simulation**: ~7 minutes (4K workers, 20K tasks, 15-min expiry)
- **Total estimated time**: 43 × 7 ≈ **301 minutes** ≈ **5.0 hours**
- **Actual runtime**: May vary ±10% based on system load

---

## Success Criteria

### Baseline Validation
- ✅ Greedy achieves lowest wait times (efficiency)
- ✅ LAF achieves highest JFI (simple fairness)
- ✅ Composite outperforms all single-objective strategies

### Pareto Sweep
- ✅ Clear fairness-efficiency trade-off curve identified
- ✅ Optimal balance point found (high JFI + acceptable wait time)
- ✅ No crashes or 0% TAR runs (all simulations valid)

### EWMA Gamma Sensitivity
- ✅ Gamma has minimal impact on final JFI (<5% variation)
- ✅ All gamma values produce valid results (TAR > 90%)
- ✅ Convergence confirmed within 3-hour window

---

## Analysis Plan

### Phase 1: Baseline Comparison
1. Compare Greedy vs LAF vs EWMA-Only vs Best Composite
2. Quantify fairness improvement: ΔJFI, ΔGini
3. Quantify efficiency cost: Δwait time, ΔTAR

### Phase 2: Pareto Analysis
1. Plot λ₁ vs λ₃ heatmap (JFI, wait time)
2. Identify Pareto frontier (non-dominated solutions)
3. Find "knee point" (optimal trade-off)
4. Compare to Exp 013 (2-hour expiry) - validate 15-min impact

### Phase 3: EWMA Validation
1. Plot γ sensitivity: JFI vs γ for each config
2. Statistical test: ANOVA on γ effect
3. Convergence analysis: EWMA values over time (if collected)
4. Compare EWMA-Only vs LAF - validate EWMA metric value

### Phase 4: Research Question Answers
- **RQ1**: Present Pareto curve with trade-off quantification
- **RQ2.1**: Show EWMA-Only vs LAF comparison (already shows LAF wins!)
- **RQ2.2**: Report γ sensitivity results (likely minimal)
- **RQ2.4**: Show EWMA convergence within 3 hours

---

## Output Files

### Data Files
- `experiment_014_aggregate_results.csv` - All 43 runs, all metrics
- `exp_014_<timestamp>/` - Individual JSON summaries per run

### Analysis Notebooks
- `analysis.ipynb` - Comprehensive analysis with all plots
- `gamma_sensitivity_analysis.ipynb` - Detailed γ analysis (if needed)

### Plots (Planned)
1. Baseline comparison (bar charts)
2. Pareto heatmaps (λ₁ vs λ₃)
3. Pareto frontier curve (JFI vs wait time)
4. Gamma sensitivity plots (line plots per config)
5. EWMA convergence plots (if time-series data available)
6. Distribution comparison (violin plots)

---

## Comparison to Previous Experiments

| Experiment | Expiry | Workers | Tasks | λ₁ Range | λ₃ Range | γ Range | Purpose |
|------------|--------|---------|-------|----------|----------|---------|---------|
| **Exp 013** | 2 hours | 4K | 20K | 2.5-5.0 | 0.5-2.0 | 0.5 (fixed) | Initial trade-off mapping |
| **Exp 014** | **15 min** | 4K | 20K | 2.5-4.5 | 0.5-2.5 | **0.1-0.9** | **Realistic + γ sensitivity** |

**Key Differences**:
- More realistic 15-minute expiry (not 2 hours)
- Includes baseline strategies (Greedy, LAF, EWMA-Only)
- Tests γ sensitivity across 3 composite configs
- Focused parameter ranges based on Exp 013 learnings

---

## Implementation Notes

### Stratified Sampling
```python
sampled_tasks, worker_samples = stratified_temporal_sample(
    all_workers=all_workers,
    all_tasks=all_tasks,
    target_tasks=20000,
    worker_counts=[4000],
    num_bins=12,
    seed=42
)
```

### Deep Copy Critical
```python
# MUST deep copy for each simulation to avoid mutation
workers = copy.deepcopy(worker_samples[4000])
tasks = copy.deepcopy(sampled_tasks)
```

### Config Template
```python
config = create_composite_config(
    assignment_strategy="composite",
    fairness_weight=λ1,
    starvation_weight=0.5,  # Fixed
    utility_weight=λ3,
    soft_threshold=0.0,  # Fixed
    normalize_scores=True,
    gamma=γ,
    k=15
)
```

---

## Timeline

- **Setup**: October 23, 2025 (Complete)
- **Execution Start**: October 23, 2025
- **Expected Completion**: October 23, 2025 (5 hours runtime)
- **Analysis**: October 24, 2025
- **Documentation**: October 25, 2025

---

## Notes

- This experiment builds on validated findings from Exp 011, 012, 013
- 15-minute expiry is critical for realistic results
- Baseline strategies (LAF, EWMA-Only) freshly implemented and tested
- All code validated with no linter errors
- Sample tests show expected behavior (LAF outperforms EWMA-Only!)

---

## Experiment Status

**Current Status**: ✅ **READY TO EXECUTE**

**Prerequisites**:
- ✅ LAF strategy implemented and validated
- ✅ EWMA-Only strategy implemented and validated  
- ✅ 15-minute expiry implemented and tested
- ✅ Deep copy bug fixed
- ✅ Stratified sampling validated
- ✅ 4K workers / 20K tasks configuration validated

**Next Steps**:
1. Create `run_experiment.py` script
2. Execute all 43 simulations (~5 hours)
3. Create `analysis.ipynb` notebook
4. Generate publication-quality figures


