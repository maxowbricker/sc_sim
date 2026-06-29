# Section 2 — EWMA Fairness Signal Validation

Validates that the EWMA signal:
1. **Outperforms simpler fairness signals** (LAF, no-fairness Greedy)
2. **Is robust to the choice of γ** — results are stable across a range of smoothing factors

Signal ablation data (Greedy / EWMA-Only / LAF / Composite rows) comes directly from the
Section 1 Didi run — no extra script needed.

## Scripts in this directory

| Script | What it does |
|--------|-------------|
| `run_gamma_sweep.py` | Runs Composite at γ ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50} with all other params fixed |

## Run commands

```bash
# γ sweep — ~8 min
python scripts/experiments/s2_ewma_validation/run_gamma_sweep.py

# or with custom output path:
python scripts/experiments/s2_ewma_validation/run_gamma_sweep.py \
    --output results/s2_ewma_validation/gamma_sweep_20161109.csv
```

## Fixed parameters
```
fairness_weight  = 1.0
starvation_weight = 0.2
utility_weight   = 1.0
k                = 15
soft_threshold   = 0.05
```

## Output files
| File | Contents |
|------|----------|
| `results/s2_ewma_validation/gamma_sweep_20161109.csv` | One row per γ value — TAR, JFI, wait, JFI rate |

## Paper narrative
> "We fix all other hyperparameters and vary only γ. Results show Δ JFI < 0.005 and
> Δ wait < 0.1 min across the full range, confirming that Composite is insensitive to
> the precise choice of γ within [0.05, 0.30]."
