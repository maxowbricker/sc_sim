# Experiment 016: Starvation Weight Interaction Analysis

## Purpose

This experiment validates whether the optimal starvation weight (λ₂=0.5) remains consistent across different fairness-utility trade-off configurations.

## Quick Summary

- **Total Simulations**: 28 (3 baselines + 25 starvation sweep)
- **Dataset**: 4K workers / 20K tasks / 15-min expiry
- **Parameters Tested**: λ₂ ∈ {0.0, 0.5, 1.0, 1.5, 2.0} across 5 representative configs
- **Estimated Runtime**: ~3.3 hours
- **Status**: Ready to Run

## Research Questions

- **RQ3.4**: Does optimal λ₂ vary with different λ₁/λ₃ configurations?
- **RQ10.2**: Are there interaction effects between λ₂ and (λ₁, λ₃)?

## Data Directory Structure

```
exp_016_starvation_weight_interaction/
├── setup.md                    # Detailed experiment documentation
├── README.md                   # This file
├── run_experiment.py           # Execution script
├── READY_TO_RUN.md            # Validation checklist
└── data/
    ├── README.md              # Data catalog
    ├── exp_016_YYYYMMDD_HHMMSS/
    │   ├── exp_001_Greedy_Baseline_summary.json
    │   ├── exp_002_LAF_Baseline_summary.json
    │   ├── exp_003_EWMA_Only_Baseline_summary.json
    │   ├── exp_004_Starvation_Best_JFI_L1_4.5_L3_0.5_L2_0.0_summary.json
    │   ├── exp_005_Starvation_Best_JFI_L1_4.5_L3_0.5_L2_0.5_summary.json
    │   └── ... (23 more starvation sweep simulations)
    └── experiment_016_aggregate_results.csv
```

## File Naming Convention

### Baselines:
- `exp_00X_{Strategy}_Baseline_summary.json`
- Examples: `Greedy_Baseline`, `LAF_Baseline`, `EWMA_Only_Baseline`

### Starvation Sweep:
- `exp_00X_Starvation_{ConfigName}_L1_{λ₁}_L3_{λ₃}_L2_{λ₂}_summary.json`
- Example: `Starvation_Balanced_L1_3.5_L3_1.0_L2_0.5_summary.json`

## Configuration Summary

### Baseline Strategies (3 simulations)
1. **Greedy**: Proximity-only (efficiency reference)
2. **LAF**: Least Allocated First (fairness reference)
3. **EWMA-Only**: EWMA fairness (γ=0.5)

### Composite Configurations (5 × 5 = 25 simulations)

| Config | λ₁ | λ₃ | Description |
|--------|----|----|-------------|
| Best_JFI | 4.5 | 0.5 | Maximum fairness |
| Balanced | 3.5 | 1.0 | Standard balanced |
| Mid_Range | 3.5 | 2.5 | Near-optimal balance |
| Efficiency_Leaning | 2.5 | 2.0 | Efficiency-focused |
| Best_Efficiency | 2.5 | 2.5 | Maximum efficiency |

**λ₂ Values Tested**: 0.0, 0.5, 1.0, 1.5, 2.0

## Key Metrics

### Fairness Metrics
- Jain's Fairness Index (JFI)
- Gini Coefficient
- % Workers with Zero Tasks

### Efficiency Metrics
- Mean Task Wait Time
- P95 Task Wait Time
- Task Assignment Ratio (TAR)

### Starvation Indicators
- P95/P99 Wait Time
- Max Wait Time

## Expected Results

- λ₂=0.5 should be near-optimal across all 5 configurations
- Minimal interaction between λ₂ and (λ₁, λ₃)
- Consistent starvation mitigation regardless of fairness-utility balance

## Running the Experiment

```bash
# Navigate to experiment directory
cd experiments_analysis/exp_016_starvation_weight_interaction

# Run experiment
python run_experiment.py

# Monitor progress in real-time
tail -f experiment_016_run.log
```

## Post-Experiment Analysis

After completion:
1. Verify all 28 simulations completed successfully
2. Check aggregate CSV has 28 rows
3. Create analysis notebook for:
   - Interaction heatmaps
   - Optimal λ₂ identification
   - Starvation mitigation quantification

## Related Experiments

- **Exp 009**: Initial λ₂=0.5 validation (limited configs)
- **Exp 014**: Pareto frontier mapping (identified 5 representative configs)
- **Exp 015**: Temporal validation (confirmed robust parameters)

---

**For detailed methodology, see `setup.md`**





