# Section 1 — Overall Strategy Performance

Scripts in this section live in the **root `scripts/` directory** (they predate section
organisation and are kept there to avoid breaking existing invocations).

## Run commands

```bash
# Didi 20161109 — full strategy comparison
caffeinate python scripts/run_strategy_comparison.py \
    --day 20161109 \
    --output results/s1_overall_performance/didi_20161109.csv

# Gowalla Austin compressed, 1:5 ratio — full strategy comparison
caffeinate python scripts/run_gowalla_comparison.py \
    --compression compressed --ratio 0.2 \
    --output results/s1_overall_performance/gowalla_austin_compressed.csv
```

## Output files
| File | Contents |
|------|----------|
| `results/s1_overall_performance/didi_20161109.csv` | One row per strategy — TAR, JFI, wait, revenue, time |
| `results/s1_overall_performance/gowalla_austin_compressed.csv` | Same columns, Gowalla dataset |

## Locked hyperparameters (as of Jun 30 2026)
| Strategy | Parameter | Value | Justification |
|----------|-----------|-------|---------------|
| Composite (static) | fw / sw / uw | 1.6 / 0.0 / 1.0 | Pareto sweep confirmed fw=1.6, sw=0.0 on Didi 20161109 (best JFI at sw=0 on frontier) |
| Composite (static) | soft_threshold | 0.0 | Sensitivity test: negligible effect; cleaner JFI |
| Composite (static) | gamma / k | 0.1 / 15 | gamma sweep confirmed; k=15 standard pool size |
| FATP-ANN | mu | 1.5 | Calibrated as ln(2)/T_mean_hours; gives ~28% utility spread (vs 16% at old mu=0.5) |
| Discrete Review LP | review_period_seconds | 15.0 | Gowalla sweep confirmed JFI peaks at 15s > 5s/10s/30s; TAR flat across all periods |

> **Note:** Composite weights (fw, sw) and soft_threshold are now identical for Didi and Gowalla —
> generalisation claim holds with updated values.

---

## Baseline inclusion decisions

### INCLUDE (confirmed for paper)

| Baseline | Reason |
|----------|--------|
| **Greedy** | Universal throughput anchor; every SC paper uses it |
| **LTF** (renamed from LAF) | Structural predecessor to k-NLF (k-NLF = LTF + spatial constraint); ablates the k-NN contribution. **Renamed** from LAF to avoid collision with CR-10's "Learning to Assign with Fairness" RL framework, which claims the same acronym for a completely different algorithm. |
| **FATP-ANN** | Published method (CR-08), O(k), directly comparable with proposed strategies; mu now properly calibrated |
| **Discrete Review LP** | Canonical batch-assignment baseline; O(W×T) cost at every review epoch creates a strong computational contrast to O(k) strategies |
| **ONRTA-RT** | Threshold-acceptance paradigm from cited paper; different algorithmic family; fast runtime |

### EXCLUDE (removed from paper table)

| Baseline | Reason |
|----------|--------|
| **TSGF** | (1) Times out on Didi 20161109 — no results on primary dataset. (2) `tsgf.py` docstring explicitly states it is "NOT a faithful reproduction of the paper algorithm": no LP solve, no PPDR, individual idle/wait proxies instead of group Rawlsian objectives, different utility functions, modified deferral mechanics. Including it (even as "TSGF-H") would invite reviewers to ask why the real algorithm was not implemented. (3) Nothing it provides that BiRanking does not already cover as a random-policy baseline. |
| **Cost-Balancing** | Times out on Didi 20161109 |
| **MMD-Batch** | Times out on Didi 20161109 |

### MAYBE (keep in scripts, decide after final results)

| Baseline | Lean | Condition for keeping |
|----------|------|-----------------------|
| **EWMA-Only** | Keep | Serves as Composite ablation (EWMA signal without utility term). Drop if JFI and wait are indistinguishable from Composite — row adds noise rather than insight. |
| **BiRanking (BRK)** | Keep | Core mechanism (permanent random priority) is preserved; adaptation note is disclosable. Prior results showed JFI ≈ 0.592 < Greedy (0.594) — a negative finding that strengthens the k-NLF story ("random reordering cannot accidentally achieve fairness"). Drop if Gowalla shows the same uninformative pattern. Two adaptation omissions: (1) stability constraints (require simultaneous neighbour knowledge — architecturally incompatible with sequential DES); (2) multi-capacity exhaustion loop (unit-capacity model). Both are disclosed in `biranking.py` docstring. |
| **ONRTA-OP** | Drop | Prior results: JFI=0.594, TAR=0.863, wait=2.623m — indistinguishable from Greedy on all metrics. ONRTA-RT alone is sufficient to represent the ONRTA family. Keep only if Gowalla results show meaningfully different behaviour. |

---

## Table column design (paper tables)

Agreed column set for Tables 1 (Didi) and 2 (Gowalla):

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
- JFI rate (workers with ≥1 task): mention in text as supporting evidence that Composite activates more of the workforce than Greedy despite lower task-count JFI
- Avg pickup distance: mention in text to explain why Composite revenue stays competitive despite fairness constraint

---

## Gowalla dataset setup
Full setup documentation (compression rationale, exact config, first run results):
→ [`docs/datasets/gowalla_austin_setup.md`](../../../docs/datasets/gowalla_austin_setup.md)
