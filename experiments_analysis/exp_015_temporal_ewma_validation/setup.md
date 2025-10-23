# Experiment 015: EWMA Temporal & Baseline Validation

**Date**: October 23, 2025  
**Status**: Ready to Run  
**Objective**: Collect temporal EWMA data for RQ2.3 (Temporal Patterns) & RQ2.4 (Convergence), and validate Random baseline for RQ4.2

---

## Overview

This experiment extends Experiment 014 by adding:
1. **Temporal EWMA tracking** to analyze convergence patterns (RQ2.3, RQ2.4)
2. **Random assignment baseline** to test null hypothesis (RQ4.2)
3. **Reduced Gamma sensitivity tests** (4 runs vs 15) leveraging Exp 014 findings

The goal is to gather rich temporal data while validating the composite strategy against a true random baseline.

---

## Research Questions Addressed

### Primary:
- **RQ2.3**: How does EWMA fairness evolve temporally during simulation?
- **RQ2.4**: Does EWMA converge to a stable value? What is convergence rate?
- **RQ4.2**: How does the composite strategy compare to random assignment?

### Secondary:
- **RQ2.1** (Re-validation): Correlation between final EWMA and JFI
- **RQ2.2** (Confirmation): Gamma robustness with temporal data

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
- λ₂ (Starvation): 0.5 (validated safety net from Exp 009)
- Normalize Scores: True (validated in Exp 008)
- k (Nearest Workers): 15 (standard)

---

## Temporal EWMA Logging

**NEW FEATURE**: The simulation now logs EWMA fairness metrics every 50 completed tasks.

**Metrics Captured at Each Checkpoint**:
- Timestamp (ISO format)
- Completed task count
- EWMA Mean across all workers
- EWMA Std deviation
- EWMA Percentiles: P10, P50 (median), P90

**Expected Checkpoints**: ~400 snapshots per simulation (20,000 tasks / 50)

**Storage**: Included in simulation summary JSON as `ewma_temporal_history` array

---

## Experiment Groups

### Group 1: Baseline Runs (3 simulations)

| # | Strategy | Description | Purpose |
|---|----------|-------------|---------|
| 1 | Greedy | Proximity-only | Efficiency reference |
| 2 | LAF | Least Allocated First | Fairness reference |
| 3 | **Random** | Random from k=15 nearest | **RQ4.2: Null hypothesis** |

**New: Random Assignment Strategy**
- Behavior: Randomly selects one worker from k=15 nearest feasible candidates
- Purpose: Tests if spatial constraint alone (without optimization) provides value
- Expected: Worse than Greedy (no efficiency), worse than LAF (no fairness tracking)

---

### Group 2: EWMA-Only Run (1 simulation)

| # | Strategy | γ | Description |
|---|----------|---|-------------|
| 4 | EWMA-Only | 0.5 | Fairness-only baseline with default gamma |

Purpose: Isolate EWMA component effectiveness

---

### Group 3: Pareto Sweep (25 simulations)

**Composite Strategy Parameter Grid**:
- λ₁ (Fairness): [2.5, 3.0, 3.5, 4.0, 4.5]
- λ₃ (Utility): [0.5, 1.0, 1.5, 2.0, 2.5]
- γ (EWMA): 0.5 (fixed, default)
- λ₂: 0.5 (fixed)

**Total**: 5 × 5 = 25 configurations

**Purpose**: Map complete Pareto frontier with temporal data

---

### Group 4: Gamma Sensitivity (4 simulations)

**Configuration**: Balanced Composite (λ₁=3.5, λ₃=1.0, λ₂=0.5)

**Gamma Values**: [0.1, 0.3, 0.7, 0.9]

Note: γ=0.5 already covered in Pareto sweep

**Rationale**: Exp 014 showed minimal gamma impact (<0.5% JFI variation). These 4 runs re-validate this finding with temporal data to confirm EWMA convergence is gamma-invariant.

---

## Total Simulations: 33

| Group | Simulations | Est. Time/Sim | Est. Total |
|-------|-------------|---------------|------------|
| Baselines | 3 | ~7 min | 21 min |
| EWMA-Only | 1 | ~7 min | 7 min |
| Pareto Sweep | 25 | ~7 min | 175 min |
| Gamma Sensitivity | 4 | ~7 min | 28 min |
| **TOTAL** | **33** | | **~3.85 hours** |

---

## Output Data Structure

### Individual Simulation JSON

Standard fields from previous experiments, plus:

```json
{
  "ewma_temporal_history": [
    {
      "timestamp": "2016-11-13T05:18:17+00:00",
      "completed_tasks": 50,
      "ewma_mean": 0.4521,
      "ewma_std": 0.1234,
      "ewma_p10": 0.2100,
      "ewma_p50": 0.4500,
      "ewma_p90": 0.6800
    },
    // ... ~400 checkpoints
  ],
  "ewma_final_mean": 0.5123,
  // ... all other standard metrics
}
```

### Aggregate CSV

All standard metrics plus:
- `ewma_final_mean`: Final average EWMA across workers
- Temporal data stored separately in individual JSONs

---

## Analysis Plan

### Temporal Analysis (RQ2.3, RQ2.4):
1. Plot EWMA evolution curves for each strategy
2. Calculate convergence rates (time to reach 95% of final value)
3. Compare convergence patterns across strategies
4. Analyze variance reduction over time

### Random Baseline Analysis (RQ4.2):
1. Compare Random to Greedy, LAF, and Composite
2. Quantify improvement over pure randomness
3. Test significance of composite strategy gains

### EWMA-JFI Correlation (RQ2.1):
1. Scatter plot: Final EWMA vs Final JFI across all 33 runs
2. Calculate Pearson correlation coefficient
3. Validate EWMA as real-time fairness proxy

### Gamma Validation (RQ2.2):
1. Compare temporal curves for different gamma values
2. Confirm convergence endpoints are gamma-invariant
3. Document convergence speed differences (if any)

---

## Expected Findings

### Hypothesis 1: EWMA Converges Monotonically
- EWMA values should stabilize after ~30-40% of tasks completed
- Convergence should be smoother with higher gamma (more historical weight)

### Hypothesis 2: Random Performs Worse Than All Strategies
- JFI: Random < Greedy < Composite < LAF
- Wait Time: Random > LAF > Composite > Greedy

### Hypothesis 3: Gamma Affects Convergence Speed, Not Endpoint
- Low gamma (0.1): Faster convergence, more volatile
- High gamma (0.9): Slower convergence, smoother
- Final EWMA mean: Same across all gamma values (±2%)

### Hypothesis 4: EWMA-JFI Correlation Confirmed
- Pearson r > 0.7 between ewma_final_mean and JFI
- Validates EWMA as real-time fairness signal

---

## Key Differences from Experiment 014

| Aspect | Exp 014 | Exp 015 |
|--------|---------|---------|
| Temporal Logging | ❌ None | ✅ Every 50 tasks |
| Random Baseline | ❌ Not included | ✅ Included |
| Gamma Tests | 15 runs (3 configs × 5 γ) | 4 runs (1 config × 4 γ) |
| Total Sims | 43 | 33 |
| Runtime | ~5.8 hours | ~3.85 hours |
| Primary Focus | Pareto frontier mapping | Temporal validation |

---

## Implementation Notes

### Temporal Logging Overhead
- Per-checkpoint cost: ~0.2 ms (numpy percentile on 4K values)
- Total overhead: ~0.08 seconds per simulation (negligible)
- Storage: ~20 KB per simulation in JSON

### Random Strategy Implementation
- Uses same feasibility checks as other strategies
- Spatial constraint: k=15 nearest workers (fair comparison)
- Implemented as `random_assign` strategy in codebase

### Deep Copy Safety
- Workers and tasks deep copied before each simulation
- Prevents mutation bugs between runs
- Gamma parameter explicitly updated for each worker in sensitivity tests

---

## Validation Checklist

Before running:
- [ ] Temporal logging functional (check with single test sim)
- [ ] Random strategy registered in config.py
- [ ] All 33 experiment configs defined
- [ ] Output directory exists: `data/`
- [ ] Previous experiment data backed up

During run:
- [ ] First 3 simulations (baselines) complete successfully
- [ ] Temporal data present in JSON output
- [ ] Random baseline shows expected performance range
- [ ] No worker pool mutation between simulations

After completion:
- [ ] All 33 simulations successful (0 failures)
- [ ] Aggregate CSV contains 33 rows
- [ ] Temporal histories present in individual JSONs
- [ ] Ready for temporal analysis notebook creation

---

## Next Steps After Completion

1. **Create temporal analysis notebook** (`temporal_analysis.ipynb`)
   - EWMA convergence plots
   - Random baseline comparison
   - Gamma convergence analysis

2. **Update Research Questions Framework**
   - Mark RQ2.3, RQ2.4, RQ4.2 as validated
   - Document temporal findings

3. **Prepare for thesis writeup**
   - Export key temporal plots
   - Document convergence rates
   - Summarize Random baseline findings

---

**Experiment 015 is ready to run!** 🚀

