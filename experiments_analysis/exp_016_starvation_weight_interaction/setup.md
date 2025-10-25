# Experiment 016: Starvation Weight (λ₂) Interaction Analysis

**Date**: October 24, 2025  
**Status**: Ready to Run  
**Objective**: Validate if λ₂=0.5 remains optimal across different λ₁/λ₃ configurations (RQ3.4, RQ10.2)

---

## Overview

This experiment tests whether the optimal starvation weight (λ₂) varies depending on the primary fairness (λ₁) and utility (λ₃) weights. Previous experiments (Exp 009) found λ₂=0.5 to be optimal, but that was tested with limited λ₁/λ₃ configurations.

This experiment systematically validates λ₂=0.5 across 5 representative composite configurations that span the Pareto frontier (from high-fairness to high-efficiency).

**Key Question**: Does the optimal starvation weight depend on whether the system is tuned for fairness or efficiency?

---

## Research Questions Addressed

### Primary:
- **RQ3.4**: Does the optimal starvation weight (λ₂) vary with different fairness-utility trade-offs?
- **RQ10.2**: Are there interaction effects between λ₂ and (λ₁, λ₃) on system performance?

### Expected Findings:
- λ₂=0.5 should remain near-optimal across all configurations
- Starvation component provides consistent safety net regardless of λ₁/λ₃ balance
- No strong interaction effects (λ₂ impact is additive, not multiplicative)

---

## Dataset Configuration

**Fixed Parameters**:
- Workers: 4,000 (validated optimal from Exp 012)
- Tasks: 20,000
- Task Expiry: 15 minutes (realistic customer patience from Exp 014)
- Worker Window: 3 hours (morning peak period)
- Sampling: Stratified temporal sampling (validated from Exp 012)

**Robust Baseline Parameters** (from prior experiments):
- θ (Soft Threshold): 0.0 (disabled, validated in Exp 011)
- γ (EWMA Gamma): 0.5 (default, validated in Exp 014)
- Normalize Scores: True (validated in Exp 008)
- k (Nearest Workers): 15 (standard)

---

## Experiment Groups

### Group 1: Baseline Runs (3 simulations)

| # | Strategy | Description | Purpose |
|---|----------|-------------|---------|
| 1 | Greedy | Proximity-only | Efficiency reference |
| 2 | LAF | Least Allocated First | Simple fairness reference |
| 3 | EWMA-Only | EWMA-based fairness (γ=0.5) | Advanced fairness reference |

Purpose: Establish baseline performance for comparison

---

### Group 2: Starvation Weight Sweep (25 simulations)

**5 Representative Configurations** (identified from Exp 014/015):

| Config Name | λ₁ | λ₃ | Description | Rationale |
|-------------|----|----|-------------|-----------|
| Best_JFI | 4.5 | 0.5 | High fairness focus | Maximum JFI from Pareto frontier |
| Balanced | 3.5 | 1.0 | Balanced trade-off | Standard reference configuration |
| Mid_Range | 3.5 | 2.5 | Moderate efficiency | Near-optimal balance score |
| Efficiency_Leaning | 2.5 | 2.0 | Efficiency-focused | Leaning toward utility |
| Best_Efficiency | 2.5 | 2.5 | Maximum efficiency | Lowest wait time from Pareto frontier |

**5 λ₂ (Starvation Weight) Values**: [0.0, 0.5, 1.0, 1.5, 2.0]

- **λ₂=0.0**: No starvation mitigation (pure fairness + utility)
- **λ₂=0.5**: Current default (validated in Exp 009)
- **λ₂=1.0**: Moderate starvation emphasis
- **λ₂=1.5**: High starvation emphasis
- **λ₂=2.0**: Maximum starvation emphasis

**Total**: 5 configs × 5 λ₂ values = 25 simulations

---

## Total Simulations: 28

| Group | Simulations | Est. Time/Sim | Est. Total |
|-------|-------------|---------------|------------|
| Baselines | 3 | ~7 min | 21 min |
| Starvation Sweep | 25 | ~7 min | 175 min |
| **TOTAL** | **28** | | **~3.3 hours** |

---

## Expected Patterns

### Hypothesis 1: λ₂=0.5 is Robust
- Performance degradation should be minimal for λ₂ ∈ [0.3, 0.7]
- λ₂=0.0 should show increased task starvation (higher P95 wait times)
- λ₂ > 1.0 should show diminishing returns or slight efficiency loss

### Hypothesis 2: No Strong Interaction Effects
- Optimal λ₂ should not vary significantly across the 5 configurations
- If λ₂=0.5 is optimal for Best_JFI, it should also be optimal for Best_Efficiency
- This validates that starvation component is orthogonal to fairness-utility trade-off

### Hypothesis 3: Starvation Impact is Consistent
- λ₂ should reduce P95/P99 wait times across all configurations
- Impact magnitude should be similar regardless of λ₁/λ₃ values
- JFI should be minimally affected by λ₂ changes

---

## Analysis Plan (Post-Experiment)

### Primary Analysis:
1. **Interaction Heatmaps**:
   - JFI vs (Config, λ₂)
   - Mean Wait Time vs (Config, λ₂)
   - P95 Wait Time vs (Config, λ₂)

2. **Optimal λ₂ by Configuration**:
   - For each of 5 configs, identify best λ₂
   - Check if optimal λ₂ is consistent (~0.5) or varies

3. **Starvation Mitigation Effectiveness**:
   - Compare λ₂=0.0 vs λ₂=0.5 for each config
   - Quantify P95 wait time reduction

### Secondary Analysis:
4. **Sensitivity Analysis**:
   - How much does performance degrade for λ₂ ∈ [0.0, 2.0]?
   - Is there a "safe range" for λ₂?

5. **Baseline Comparison**:
   - How do best composite configs (with optimal λ₂) compare to baselines?

---

## Key Metrics to Track

### Fairness Metrics:
- Jain's Fairness Index (JFI)
- Gini Coefficient
- % Workers with Zero Tasks

### Efficiency Metrics:
- Mean Task Wait Time
- P95 Task Wait Time
- Task Assignment Ratio (TAR)

### Starvation Indicators:
- P95 Wait Time (primary indicator)
- P99 Wait Time (extreme cases)
- Max Wait Time (worst case)

---

## Output Data Structure

### Individual Simulation JSON
Standard format from Exp 015, including:
- All fairness metrics (JFI, Gini, etc.)
- All wait time statistics (mean, P90, P95, P99, max)
- Worker utilization metrics
- Temporal EWMA data (if logged)

### Aggregate CSV
One row per simulation (28 rows total):
- `exp_id`, `exp_name`, `strategy`
- `fairness_weight` (λ₁), `starvation_weight` (λ₂), `utility_weight` (λ₃)
- All performance metrics

**Naming Convention**:
```
Starvation_Balanced_L1_3.5_L3_1.0_L2_0.5
```

Format: `Starvation_{ConfigName}_L1_{λ₁}_L3_{λ₃}_L2_{λ₂}`

---

## Implementation Notes

### Critical: Deep Copy Safety
From Exp 014 lesson learned, workers must be deep copied before each simulation AND gamma must be explicitly updated:

```python
exp_workers = copy.deepcopy(workers)
exp_tasks = copy.deepcopy(sampled_tasks)

# CRITICAL: Update worker gamma if specified in config
if 'gamma' in exp['params']:
    for worker in exp_workers:
        worker.gamma = exp['params']['gamma']
```

### Data Loading
Use the same data loading as Exp 015:
```python
data_path = project_root / "data" / "didi"
all_workers, all_tasks = load_workers_tasks('didi', str(data_path))
```

### Progress Logging
Print progress every simulation with key metrics:
```
🎲 Experiment 005/028 - Starvation_Balanced_L1_3.5_L3_1.0_L2_0.5
   Strategy: composite
   Params: {'fairness_weight': 3.5, 'starvation_weight': 0.5, 'utility_weight': 1.0, ...}
   ✅ Completed: 18,686/20,000 tasks
   📊 JFI: 0.773, Wait: 2.5 min, TAR: 93.4%
```

---

## Validation Checklist

### Before Running:
- [ ] All 28 experiment configs defined
- [ ] Output directory exists: `data/`
- [ ] Didi 3-hour peak dataset available
- [ ] Dependencies installed (numpy, pandas, etc.)

### During Run:
- [ ] First 3 simulations (baselines) complete successfully
- [ ] λ₂=0.0 configs show expected higher wait times
- [ ] λ₂=0.5 configs show balanced performance
- [ ] No worker pool mutation between simulations

### After Completion:
- [ ] All 28 simulations successful (0 failures)
- [ ] Aggregate CSV contains 28 rows
- [ ] Individual JSON files created for each simulation
- [ ] Ready for analysis notebook creation

---

## Next Steps After Completion

1. **Create Analysis Notebook**
   - Interaction heatmaps (JFI, Wait Time vs Config × λ₂)
   - Optimal λ₂ identification for each config
   - Starvation mitigation quantification

2. **Update Research Questions Framework**
   - Mark RQ3.4 and RQ10.2 as validated
   - Document findings on λ₂ robustness

3. **Compare to Experiment 009**
   - Validate that findings are consistent
   - Document any new insights on interaction effects

---

**Experiment 016 is ready to run!** 🚀






