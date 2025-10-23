# Experiment 014: EWMA & Trade-Off Exploration

**Status**: Ready to Execute  
**Created**: October 23, 2025

## Quick Summary

Comprehensive 43-simulation experiment testing:
1. **Baseline strategies** (Greedy, LAF, EWMA-Only) - 3 sims
2. **Fairness-Efficiency trade-off** (λ₁ vs λ₃ sweep) - 25 sims  
3. **EWMA gamma sensitivity** (5 γ values × 3 configs) - 15 sims

**Key Innovation**: First experiment with realistic **15-minute task expiry** and all baseline strategies.

## Key Parameters

- **Dataset**: 4K workers / 20K tasks (3-hour peak)
- **Expiry**: 15 minutes (realistic)
- **Fixed**: θ=0.0, λ₂=0.5
- **Variable**: λ₁ (2.5-4.5), λ₃ (0.5-2.5), γ (0.1-0.9)

## Expected Runtime

~5 hours (43 sims × 7 min/sim)

## Research Questions

- **RQ1**: Fairness-efficiency trade-off quantification
- **RQ2.1**: EWMA vs simpler fairness metrics
- **RQ2.2**: Gamma parameter sensitivity
- **RQ2.4**: EWMA convergence within 3-hour window

## Files

- `setup.md` - Detailed experiment design
- `run_experiment.py` - Execution script (to be created)
- `analysis.ipynb` - Analysis notebook (to be created)
- `data/` - Results directory

## Quick Start

```bash
cd experiments_analysis/exp_014_ewma_tradeoff_exploration
python run_experiment.py
```

## Related Experiments

- **Exp 011**: Scalability analysis (worker-to-task ratios)
- **Exp 012**: Worker ratio validation
- **Exp 013**: Initial trade-off mapping (2-hour expiry)
- **Exp 014**: **This experiment** (realistic 15-min expiry + baselines)


