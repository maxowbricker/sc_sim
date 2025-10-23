# Experiment 015: EWMA Temporal & Baseline Validation

**Status**: Ready to Run  
**Date**: October 23, 2025  
**Estimated Runtime**: ~3.85 hours (33 simulations)

## Quick Summary

This experiment collects **temporal EWMA fairness data** and validates the **Random assignment baseline** to address:
- **RQ2.3**: Temporal evolution of EWMA fairness
- **RQ2.4**: EWMA convergence patterns
- **RQ4.2**: Performance vs random assignment baseline

### What's New

1. **Temporal Logging**: EWMA metrics logged every 50 completed tasks (~400 snapshots per simulation)
2. **Random Baseline**: New `random_assign` strategy for null hypothesis testing
3. **Reduced Gamma Tests**: Only 4 runs (vs 15 in Exp 014) based on robustness findings

## Experiment Groups

| Group | Simulations | Description |
|-------|-------------|-------------|
| Baselines | 3 | Greedy, LAF, **Random** |
| EWMA-Only | 1 | γ=0.5 |
| Pareto Sweep | 25 | 5×5 grid (λ₁ × λ₃) |
| Gamma Sensitivity | 4 | Balanced config, γ=[0.1,0.3,0.7,0.9] |
| **Total** | **33** | |

## Quick Start

### Run Experiment

```bash
cd experiments_analysis/exp_015_temporal_ewma_validation
../../venv/bin/python run_experiment.py
```

### Monitor Progress

```bash
# In another terminal
tail -f nohup.out  # If running in background

# Or check progress
ls data/exp_015_*/exp_*.json | wc -l
```

### Run in Background

```bash
nohup ../../venv/bin/python -u run_experiment.py > experiment_015_run.log 2>&1 &
```

## Output Structure

```
data/
├── experiment_015_aggregate_results.csv  # All 33 runs
└── exp_015_YYYYMMDD_HHMMSS/
    ├── exp_001_Greedy_Baseline_summary.json
    ├── exp_002_LAF_Baseline_summary.json
    ├── exp_003_Random_Baseline_summary.json
    ├── exp_004_EWMA_Only_G_0.5_summary.json
    ├── exp_005_Pareto_L1_2.5_L3_0.5_summary.json
    ...
    └── exp_033_Gamma_Balanced_G_0.9_summary.json
```

### Temporal Data Format

Each JSON includes:
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
    // ~400 checkpoints total
  ],
  "ewma_final_mean": 0.5123,
  // ... standard metrics
}
```

## Key Parameters

**Fixed (Validated)**:
- Workers: 4,000
- Tasks: 20,000
- Expiry: 15 minutes
- θ: 0.0
- λ₂: 0.5
- Normalize: True

**Variable**:
- λ₁: [2.5, 3.0, 3.5, 4.0, 4.5]
- λ₃: [0.5, 1.0, 1.5, 2.0, 2.5]
- γ: [0.1, 0.3, 0.5, 0.7, 0.9]

## Expected Results

### Temporal Patterns (RQ2.3)
- EWMA should converge after ~30-40% of tasks
- Convergence smoothness depends on gamma

### Convergence (RQ2.4)
- Low γ (0.1): Faster, more volatile
- High γ (0.9): Slower, smoother
- Final values: Similar across all γ (±2%)

### Random Baseline (RQ4.2)
- Random < Greedy (efficiency)
- Random < LAF (fairness)
- Random < Composite (both)

## Analysis Steps

1. **Load temporal data**:
   ```python
   import json
   with open('data/exp_015_*/exp_005_*.json') as f:
       data = json.load(f)
       temporal = data['ewma_temporal_history']
   ```

2. **Plot convergence**:
   ```python
   tasks = [t['completed_tasks'] for t in temporal]
   ewma_mean = [t['ewma_mean'] for t in temporal]
   plt.plot(tasks, ewma_mean)
   ```

3. **Compare strategies**:
   - Load aggregate CSV
   - Compare Random to baselines
   - Analyze temporal curves

## References

- **Setup Details**: See `setup.md`
- **Predecessor**: Experiment 014 (Pareto frontier mapping)
- **Related**: Experiments 011 (theta), 012 (worker ratio), 009 (lambda2)

## Notes

- First experiment with temporal EWMA logging
- Introduces Random baseline strategy
- Validates gamma robustness with temporal data
- Runtime similar to Exp 014 despite fewer sims (33 vs 43)

---

**Ready to Run**: ✅  
**See**: `setup.md` for detailed methodology

