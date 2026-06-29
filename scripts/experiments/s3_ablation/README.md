# Section 3 — Ablation Study (Pareto Frontier)

Three sub-experiments:
1. **Pareto sweep** — map the fairness×starvation weight space
2. **Soft-threshold sensitivity** — confirm st=0.05 is a sensible default
3. **k sensitivity** — confirm k=15 is sufficient

Pareto and soft-threshold scripts live in the **root `scripts/` directory**.

## Scripts

| Location | Script | Purpose |
|----------|--------|---------|
| `scripts/` | `run_composite_pareto_sweep.py` | 110-config λ₁ × λ₂ grid |
| `scripts/` | `run_soft_threshold_test.py` | st ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 1.0} |
| `scripts/experiments/s3_ablation/` | `run_k_sweep.py` | k ∈ {5, 10, 15, 20, 30} |

## Run commands

```bash
# 1. Pareto sweep (fairness × starvation) — ~2.5 h
caffeinate python scripts/run_composite_pareto_sweep.py \
    --output results/s3_ablation/pareto_sweep_20161109.csv

# 2. Soft-threshold sensitivity — ~10 min
python scripts/run_soft_threshold_test.py \
    --output results/s3_ablation/soft_threshold_20161109.csv

# 3. k sensitivity — ~7 min
python scripts/experiments/s3_ablation/run_k_sweep.py
```

## Fixed parameters (across all three experiments)
```
utility_weight  = 1.0   (fixed to allow fairness/utility ratio interpretation)
gamma           = 0.1
```
Pareto sweep: `soft_threshold = 0.0`
k sweep: `fairness_weight=1.0  starvation_weight=0.2  soft_threshold=0.05`

## Output files
| File | Contents |
|------|----------|
| `results/s3_ablation/pareto_sweep_20161109.csv` | fw × sw grid — TAR, JFI, wait per config |
| `results/s3_ablation/soft_threshold_20161109.csv` | st ∈ 6 values — small monotonic JFI drop |
| `results/s3_ablation/k_sweep_20161109.csv` | k ∈ 5 values — should show saturation ≥ k=15 |

## Paper narrative
> "The Pareto frontier reveals that fw≈1.0 and sw≈0.2 (relative to uw=1.0) sit on the
> knee of the JFI–wait trade-off curve. We then fix these weights for all subsequent
> experiments. k=15 achieves indistinguishable performance from k=20 and k=30 at roughly
> half the per-event candidate scan cost."
