# Experiment Scripts — Section Map

Each subdirectory corresponds to one section of the paper's experimental results.
Scripts produce output CSVs to the matching `results/sX_*/` directory.

---

## Section 1 — Overall Strategy Performance
**Directory:** `scripts/experiments/s1_overall_performance/`
**Results:** `results/s1_overall_performance/`

| Script | Purpose | Est. time |
|---|---|---|
| `../../run_strategy_comparison.py --day 20161109 --output ../../results/s1_overall_performance/didi_20161109.csv` | Full 10-strategy comparison on Didi | ~4.5 h |
| `../../run_gowalla_comparison.py --compression compressed --output ../../results/s1_overall_performance/gowalla_austin_compressed.csv` | Full strategy comparison on Gowalla | ~70 min |

**Primary config (Didi):** 36,799 workers | 224,219 tasks | day = 20161109
**Primary config (Gowalla):** compressed, 1:7 ratio, Austin Sep 2010

---

## Section 2 — EWMA Fairness Signal Validation
**Directory:** `scripts/experiments/s2_ewma_validation/`
**Results:** `results/s2_ewma_validation/`

| Script | Purpose | Est. time |
|---|---|---|
| `run_gamma_sweep.py` | γ sensitivity — how smooth does EWMA need to be? | ~8 min |

**Signal ablation data (Greedy / EWMA-Only / LAF / Composite) comes from the Section 1 Didi run.**

---

## Section 3 — Ablation Study (Pareto Frontier)
**Directory:** `scripts/experiments/s3_ablation/`
**Results:** `results/s3_ablation/`

| Script | Purpose | Est. time |
|---|---|---|
| `../../run_composite_pareto_sweep.py --output ../../results/s3_ablation/pareto_sweep_20161109.csv` | 110-config fairness×starvation Pareto grid | ~2.5 h |
| `../../run_soft_threshold_test.py --output ../../results/s3_ablation/soft_threshold_20161109.csv` | Soft-threshold sensitivity (6 configs) | ~10 min |
| `run_k_sweep.py` | k (candidate pool size) sensitivity (5 configs) | ~7 min |

**Fixed params throughout:** `utility_weight=1.0, gamma=0.1, soft_threshold=0.0`

---

## Section 4 — Robustness to Spatiotemporal Density
**Directory:** `scripts/experiments/s4_robustness/`
**Results:** `results/s4_robustness/`

| Script | Purpose | Est. time |
|---|---|---|
| `run_didi_ratio_sweep.py` | Worker:task ratio sweep on Didi (3 ratios × 4 strategies) | ~20 min |
| `../../run_gowalla_comparison.py --compression compressed --output ../../results/s4_robustness/gowalla_ratio_sweep.csv` | Gowalla 1:4/1:5/1:7 (Greedy + Composite) | ~12 min |

---

## Output file naming convention

```
results/
  s1_overall_performance/
    didi_20161109.csv          — full strategy table (fixed strategies)
    gowalla_austin_compressed.csv — full strategy table on Gowalla
  s2_ewma_validation/
    gamma_sweep_20161109.csv   — γ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}
  s3_ablation/
    pareto_sweep_20161109.csv  — 110-config λ₁ × λ₂ grid
    soft_threshold_20161109.csv — st ∈ {0.0, 0.05, 0.1, 0.2, 0.3, 1.0}
    k_sweep_20161109.csv       — k ∈ {5, 10, 15, 20, 30}
  s4_robustness/
    didi_ratio_sweep.csv       — Didi at 1:4, 1:5, 1:7 ratios
    gowalla_ratio_sweep.csv    — Gowalla at 1:4, 1:5, 1:7 ratios
```
