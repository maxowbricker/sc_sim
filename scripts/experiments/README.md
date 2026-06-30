# Experiment Scripts — Section Map

Three experimental sections map directly to §5.2, §5.3, and §5.4 of the paper.
Scripts produce output CSVs to the matching `results/s5X_*/` directory.
`parameter_tuning/` holds dev and calibration scripts that are not paper sections.

---

## §5.2 — Main Results & Effectiveness
**Directory:** `scripts/experiments/s52_main_results/`
**Results:** `results/s1_overall_performance/`

| Script | Purpose | Est. time |
|---|---|---|
| `s52_main_results/run_strategy_comparison.py --day 20161109` | Full strategy comparison on Didi | ~4.5 h |
| `s52_main_results/run_gowalla_comparison.py --compression compressed --ratio 0.2` | Full strategy comparison on Gowalla | ~70 min |

See `s52_main_results/README.md` for locked hyperparameters, baseline decisions, and table design.

---

## §5.3 — Computational Efficiency & Scalability
**Directory:** `scripts/experiments/s53_scalability/`
**Results:** `results/s53_scalability/`

| Script | Purpose | Est. time |
|---|---|---|
| `s53_scalability/run_scalability_fleet.py` | Vary fleet size \|W\| 1k–40k, fixed tasks | ~3–4 h |
| `s53_scalability/run_scalability_tasks.py` | Vary task volume \|T\| 10k–200k, fixed fleet | ~1.5–2 h |

See `s53_scalability/README.md` for expected runtime growth patterns.

---

## §5.4 — Ablation & Sensitivity Analysis
**Directory:** `scripts/experiments/s54_ablation/`
**Results:** `results/s54_ablation/`

| Script | Section | Purpose | Est. time |
|---|---|---|---|
| `s54_ablation/run_knlf_k_sweep.py` | §5.4.1 | k sweep for k-NLF, Greedy & LAF anchors | ~10 min |
| `s54_ablation/run_signal_comparison.py` | §5.4.2 | Signal isolation: k-NLF vs k-NTF-EPH vs k-NTF-IR at k=15 | ~12 min |
| `s54_ablation/run_fairness_weight_sweep.py` | §5.4.3 | λ_f sweep for Composite, 13 weight values | ~22 min |

---

## Parameter Tuning (not paper sections)
**Directory:** `scripts/experiments/parameter_tuning/`

Dev and calibration scripts used to lock the paper-final hyperparameters.
Results are for reference only; re-running is not required for paper submission.

| Script | Purpose |
|---|---|
| `run_k_sweep.py` | Composite k sensitivity (used to confirm k=15) |
| `run_gamma_sweep.py` | EWMA γ sensitivity (used to confirm γ=0.1) |
| `run_didi_ratio_sweep.py` | Worker:task ratio robustness check on Didi |
| `sweep_review_period.py` | LP review-period calibration for Discrete Review LP |
| `verify_onrta_rt_scale.py` | ONRTA-RT scale verification |
| `verify_onrta_op_injection.py` | ONRTA-OP injection verification |

---

## Output directory layout

```
results/
  s1_overall_performance/        ← §5.2 main comparison outputs
    didi_20161109.csv
    gowalla_austin_compressed.csv
  s53_scalability/               ← §5.3 scalability sweep outputs
    scalability_fleet.csv
    scalability_tasks.csv
  s54_ablation/                  ← §5.4 ablation outputs
    knlf_k_sweep_20161109.csv
    signal_comparison_20161109.csv
    fairness_weight_sweep_20161109.csv
```
