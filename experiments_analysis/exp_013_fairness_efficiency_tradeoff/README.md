# Experiment 013: High-Resolution Fairness-Efficiency Trade-off Mapping

**Status**: Ready to Run  
**Date**: October 24, 2025  
**Duration**: ~8 hours (73 experiments)

---

## Quick Reference

### Objective
Map the Pareto frontier at high resolution by systematically varying λ₁ (fairness) and λ₃ (utility) weights.

### Configuration
- **Workers**: 4,000 (validated optimal from Exp 012)
- **Tasks**: 20,000 (stratified temporal sampling)
- **Fixed λ₂**: 0.5 (validated safety net)
- **Fixed θ**: 0.0 (disabled, validated by Exp 011 & 012)
- **Normalize Scores**: True

### Variable Parameters
- **λ₁ (Fairness)**: [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0] (10 values)
- **λ₃ (Utility)**: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] (7 values)

### Experiments
1. **Greedy Baseline** (1): Efficiency reference
2. **λ₁ × λ₃ Grid** (70): 10 × 7 = 70 configurations
3. **Balance Points** (2): Equal/near-equal weight tests

**Total**: 73 experiments

---

## Key Innovation

**High-Resolution Parameter Sweep**: Unlike Exp 009's broad exploration, this experiment focuses on the **promising region** (λ₁ ≥ 2.5) with fine-grained resolution to precisely identify the optimal balance.

---

## Expected Outcomes

1. **Precise Pareto Frontier**: High-resolution map of fairness-efficiency trade-off
2. **Validated Optimal Parameters**: Confirm or refine λ₁=2.0, λ₃=1.0 from Exp 009
3. **Sensitivity Analysis**: Understand robustness to parameter changes
4. **Balance Point Analysis**: Test equal weighting hypothesis

---

## How to Run

```bash
cd experiments_analysis/exp_013_fairness_efficiency_tradeoff
../../venv/bin/python run_experiment.py
```

Or run in background:
```bash
nohup ../../venv/bin/python -u run_experiment.py > experiment_013_run.log 2>&1 &
```

Monitor progress:
```bash
tail -f experiment_013_run.log
```

---

## Output

- **Individual JSONs**: `data/exp_013_TIMESTAMP/exp_XXX_*.json` (73 files)
- **Aggregate CSV**: `data/experiment_013_aggregate_results.csv`
- **Metrics**: All 78+ v2.0 metrics for each experiment

---

## Analysis

After completion, analyze using `analysis.ipynb`:
- 2D heatmaps (JFI, Wait Time, Gini by λ₁ × λ₃)
- Pareto frontier plot
- Contour plots
- 3D surface plots
- Sensitivity analysis
- Comparison to Exp 009 optimal

---

## Links

- **Setup Details**: [setup.md](setup.md)
- **Execution Guide**: [READY_TO_RUN.md](READY_TO_RUN.md)
- **Exp 009**: Broad parameter sweep (parent experiment)
- **Exp 012**: Worker ratio analysis (validated 4K workers)

---

**Contact**: Max  
**Priority**: HIGH (core RQ1 validation)



