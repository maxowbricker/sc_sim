# Gowalla Dataset — Austin September 2010 Setup

This document records the exact configuration used for the Gowalla dataset in the
paper's experiments, so it can be reproduced exactly in future runs.

---

## Why Gowalla?

A secondary real-world dataset to complement the Didi GAIA dataset and demonstrate
that Composite strategy results **generalise beyond ride-hailing**. Gowalla is a
location-based social network (LBSN) check-in dataset — fundamentally different from
ride-hailing in density, task structure, and spatial coverage.

---

## Source Data

| Property | Value |
|---|---|
| Dataset | Gowalla public LBSN check-in dataset |
| Region filter | **Austin, Texas** (densest spatial cluster in the dataset) |
| Bounding box | Set by `region="austin"` preset in `GOWALLA_CONFIG` |
| Date range | **2010-09-01 → 2010-09-30** (September 2010 — full month) |
| Raw check-ins in window | ~43,788 tasks after filtering |

---

## Key preprocessing: Temporal Compression

**This is the most important setup decision — do not omit it.**

Raw Gowalla check-in rates are ~150× lower than Didi trip rates. Without compression,
September 2010 Austin has only ~31 tasks active at any given moment — far too sparse
for meaningful strategy differentiation (all strategies converge to Greedy behaviour).

`compress_to_day=True` strips the calendar date from every check-in timestamp, keeping
only the time-of-day (HH:MM:SS). All 43,788 check-ins across the full month are
stacked onto a **single 24-hour reference window**.

After compression:
- **~912 tasks active concurrently** at peak
- Comparable density to a stratified-sampled Didi day
- Strategies show meaningful differentiation

```python
GOWALLA_CONFIG["compress_to_day"] = True   # MUST be True for paper experiments
```

---

## Task Generation

| Parameter | Value | Rationale |
|---|---|---|
| `task_mode` | `"checkin"` | Every check-in = 1 task. More realistic worker:task ratio than `location_pair`. |
| `task_window_hours` | `0.5` (30 min) | Task expiry window. Matches Didi's median trip duration (~19.5 min). Sits between Didi P50 and P90. |
| `dropoff_noise_km` | `2.0` | Std-dev radius (km) for synthetic dropoff displacement from pickup location. |

---

## Worker Configuration

Workers are synthetic — sampled from the check-in user pool and assigned random
starting locations drawn from the task spatial distribution.

| Parameter | Value | Rationale |
|---|---|---|
| `workers_per_task_ratio` | 0.20 (1:5) for primary run | Matches Didi natural ratio (~1:6). Also tested at 1:7 and 1:4. |
| `shift_hours` | 8.0 h | Worker shift duration (deadline = release_time + 8h × 3600s). |

**Resulting worker counts:**

| Ratio | Workers | Tasks |
|---|---|---|
| 1:5 (0.20) | 8,758 | 43,788 |
| 1:7 (0.143) | 6,255 | 43,788 |
| 1:4 (0.25) | 10,947 | 43,788 |

---

## Composite Strategy Weights (identical to Didi — generalisation claim)

```python
{
    "fairness_weight":   1.0,
    "starvation_weight": 0.2,
    "utility_weight":    1.0,
    "gamma":             0.1,
    "k":                 15,
    "soft_threshold":    0.05,
}
```

These weights were **tuned on Didi and left unchanged for Gowalla**. The paper uses
this as evidence of generalisation — no dataset-specific hyperparameter tuning.

---

## Exact run command (paper configuration)

```bash
caffeinate python3.11 scripts/run_gowalla_comparison.py \
    --region austin \
    --date-start 2010-09-01 \
    --date-end 2010-09-30 \
    --compression compressed \
    --output results/s52_main_results/gowalla_austin_compressed.csv
```

Or using the section-organised output path shorthand:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim
caffeinate python3.11 scripts/run_gowalla_comparison.py \
    --compression compressed \
    --output results/s52_main_results/gowalla_austin_compressed.csv
```

(`--region austin` and `--date-start/end` default to the paper values in the script.)

---

## Results from first paper run (Jun 28 2026)

Saved to: `gowalla_austin_sep2010.csv` (root dir) — **only Greedy + Composite at that time**.

| Config | Workers | TAR | JFI (tasks) | JFI (earnings) | JFI rate | Avg Wait (m) |
|---|---|---|---|---|---|---|
| 1:5 / Composite | 8,758 | 0.9984 | 0.6011 | 0.6438 | 0.9649 | 5.27m |
| 1:7 / Greedy | 6,255 | 0.9622 | 0.5886 | 0.6145 | 0.9898 | 3.69m |
| 1:7 / Composite | 6,255 | 0.9968 | 0.6319 | 0.6603 | 0.9805 | 6.63m |
| 1:4 / Greedy | 10,947 | 0.9841 | 0.5672 | 0.6107 | 0.9787 | 2.60m |
| 1:4 / Composite | 10,947 | 0.9988 | 0.6017 | 0.6543 | 0.9535 | 4.75m |

**Notable observations:**
- Composite achieves **higher TAR** than Greedy on Gowalla (unlike Didi where they're equal).
  Possible cause: temporal compression creates task clusters where Greedy's nearest-first
  logic leads to more task expirations.
- Composite **JFI advantage over Greedy is clearer on Gowalla** (~+0.043 at 1:7 ratio)
  compared to Didi (~−0.009). Consistent with the generalisation narrative.
- Both strategies show very high utilisation = 0.0% — a known artefact of the
  Gowalla adapter not tracking worker shift time correctly. **Utilisation metric
  is not reliable for Gowalla** and should be excluded from paper tables.

---

## TODOs for Section 1 Gowalla run

- [ ] Add full strategy list (EWMA-Only, LAF, BiRanking, FATP-ANN, ONRTA-OP, Discrete Review LP)
      to `run_gowalla_comparison.py` STRATEGIES list and rerun
- [ ] Confirm utilisation tracking fix or explicitly exclude from paper
- [ ] Add Gini + Utility Diff columns (same as Didi run) once `extract_metrics` is updated in
      `run_gowalla_comparison.py`
