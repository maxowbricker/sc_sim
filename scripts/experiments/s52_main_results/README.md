# §5.2 — Main Results & Effectiveness

## Run commands

```bash
# Didi 20161109 — full strategy comparison
caffeinate python scripts/experiments/s52_main_results/run_strategy_comparison.py \
    --day 20161109 \
    --output results/s1_overall_performance/didi_20161109.csv

# Gowalla Austin compressed, 1:5 ratio — full strategy comparison
caffeinate python scripts/experiments/s52_main_results/run_gowalla_comparison.py \
    --compression compressed --ratio 0.2 \
    --output results/s1_overall_performance/gowalla_austin_compressed.csv
```

## Output files
| File | Contents |
|------|----------|
| `results/s1_overall_performance/didi_20161109.csv` | One row per strategy — TAR, JFI, wait, revenue, time |
| `results/s1_overall_performance/gowalla_austin_compressed.csv` | Same columns, Gowalla dataset |

---

## Locked hyperparameters (as of Jun 30 2026)
| Strategy | Parameter | Value | Justification |
|----------|-----------|-------|---------------|
| Composite (static) | fw / sw / uw | 1.6 / 0.0 / 1.0 | Pareto sweep confirmed fw=1.6, sw=0.0 on Didi 20161109 (best JFI at sw=0 on frontier) |
| Composite (static) | soft_threshold | 0.0 | Sensitivity test: negligible effect; cleaner JFI |
| Composite (static) | gamma / k | 0.1 / 15 | Gamma sweep confirmed; k=15 standard pool size |
| FATP-ANN | mu | 1.5 | Calibrated as ln(2)/T_mean_hours; gives ~28% utility spread (vs 16% at old mu=0.5) |
| Discrete Review LP | review_period_seconds | 15.0 | Gowalla sweep confirmed JFI peaks at 15s > 5s/10s/30s; TAR flat across all periods |

> **Note:** Composite weights (fw, sw) and soft_threshold are identical for Didi and Gowalla —
> generalisation claim holds with updated values.

---

## Table column design (paper Tables 1 & 2)

Agreed column set:
```
Strategy | TAR ↑ | Wait ↓ | JFI (tasks) ↑ | JFI (earnings) ↑ | Rev. (k$) ↑ | Time (s) ↓
```

**Stakeholder grouping** (maps to §5.1.3):
- Platform: TAR, Rev.
- Customer: Wait
- Worker: JFI (tasks), JFI (earnings)
- System: Time (s)

**Dropped from table (discuss in narrative instead):**
- Backlog: near-zero for most strategies post-fix — not discriminating
- JFI rate: mention in text as supporting evidence
- Avg pickup distance: mention in text to explain revenue competitiveness

---

## Baseline inclusion decisions

### INCLUDE (confirmed for paper)

| Baseline | Reason |
|----------|--------|
| **Greedy** | Universal throughput anchor |
| **LTF** (formerly LAF) | Structural predecessor to k-NLF; ablates the k-NN contribution. Renamed to avoid collision with CR-10 "Learning to Assign with Fairness" (different algorithm, same acronym) |
| **FATP-ANN** | Published method (CR-08), O(k), directly comparable; mu now properly calibrated |
| **Discrete Review LP** | Canonical batch-assignment baseline; O(W×T) cost creates strong computational contrast |
| **ONRTA-RT** | Threshold-acceptance paradigm; different algorithmic family; fast runtime |
| **k-NLF** | Our contribution — O(k) local task-count fairness |
| **Composite** | Our contribution — O(k) local EWMA idle-time fairness |

### EXCLUDE (removed from paper table)

| Baseline | Reason |
|----------|--------|
| **TSGF** | Times out on Didi 20161109. `tsgf.py` is not a faithful reproduction (no LP, no PPDR, different objective). Including it would invite reviewer scrutiny. |
| **Cost-Balancing** | Times out on Didi 20161109 |
| **MMD-Batch** | Times out on Didi 20161109 |

### MAYBE (keep in scripts, decide after final results)

| Baseline | Lean | Condition for keeping |
|----------|------|-----------------------|
| **EWMA-Only** | Keep | Serves as Composite ablation (EWMA signal without utility term). Drop if JFI/wait indistinguishable from Composite |
| **BiRanking (BRK)** | Keep | Prior: JFI≈0.592 < Greedy (0.594) — negative finding strengthens k-NLF story. Drop if Gowalla shows same uninformative pattern |
| **ONRTA-OP** | Drop | Prior: JFI=0.594, TAR=0.863, wait=2.623m — indistinguishable from Greedy. ONRTA-RT alone sufficient |

---

## Gowalla dataset setup
Full setup documentation (compression rationale, exact config, first run results):
→ [`docs/datasets/gowalla_austin_setup.md`](../../../docs/datasets/gowalla_austin_setup.md)
