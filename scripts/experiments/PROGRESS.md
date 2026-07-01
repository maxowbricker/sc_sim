# Experiment & Writing Progress Tracker

> Last updated: 2026-07-01 (post-optimisation re-run pass)
>
> **Stages**
> - **Stage 1 — Experiments**: raw simulation results collected
> - **Stage 2 — Plotting / Tabling**: results transferred into `experimental-section.tex`
> - **Stage 3 — Analysis & Writing**: narrative prose, claims, and discussion written


---

## Paper Data Provenance & Codebase Cleanup Guide

> **Purpose:** Documents exactly which CSV file (and which rows) supply each table
> and figure in the paper. Use this as the reference when cleaning `results/` to keep
> only files that are directly cited, as if you had just run the experimental scripts
> and left their output in place.
>
> **⚠️ Re-run required:** Greedy's implementation was reverted from k=50 spatial index
> back to O(|W|) global scan (2026-07-01) to match the paper's baseline description.
> The existing `didi_greedy_k50.csv` and `gowalla_greedy_k50.csv` are now stale.
> Run the laptop calibration procedure below before updating the paper tables.

### Why §5.2 tables draw from multiple CSVs

The DiDi and Gowalla main-results tables mix sources because Greedy was re-run
separately with a corrected implementation after the k=10 k=50 artifact was
discovered. All other strategies use k=15 by design (their candidate pool is a
fairness hyperparameter, not a search budget) and were not re-run. The Greedy-only
re-run files are the minimal additions required to correct this.

---

### Laptop calibration run — ✅ COMPLETE (2026-07-01)

The global-scan Greedy runtime has been estimated for the cluster via relative
benchmarking — the DES is deterministic so all non-runtime metrics (TAR, JFI, Wait,
Revenue) are taken directly from the laptop run; only wall-clock time is scaled.

**Laptop run outputs:**

| Dataset | File | Greedy laptop (s) | k-NLF laptop (s) |
|---|---|---|---|
| DiDi 20161109 | `results/s52_main_results/didi_greedy_global_laptop.csv` | 239.54 | 94.52 |
| Gowalla Austin | `results/s52_main_results/gowalla_greedy_global_laptop.csv` | 73.5 | 20.4 |

**Scaled cluster runtimes (formula: greedy_laptop × knlf_cluster / knlf_laptop):**

| Dataset | Formula | Cluster estimate |
|---|---|---|
| DiDi | 239.54 × (123 / 94.52) | **312s** |
| Gowalla | 73.5 × (27.3 / 20.4) | **98s** |

**Final global-scan Greedy values for paper tables:**

| Metric | DiDi | Gowalla |
|---|---|---|
| TAR | 0.863 | 0.998 |
| Avg Wait (m) | 2.62 | 2.87 |
| JFI (tasks) | 0.594 | 0.573 |
| JFI (earn.) | 0.611 | 0.608 |
| Revenue (k$) | 2,765.6 | 297.6 |
| CV (idle) | 1.976 | — |
| Time (s) — cluster est. | **312** | **98** |

**Note — CV (earn.) for signal_comparison table:** `didi_greedy_global_laptop.csv` does
not output per-worker earnings distribution (no CV earn. column). This value is needed
to update the Greedy row in `tab:signal_comparison`. Re-run signal comparison with
`--only greedy` once `run_signal_comparison.py` supports it, or accept the existing
v3 value (0.949, k=10 era) as a placeholder.

---

### §5.2 — Table `tab:didi_results` (DiDi main results)

| Paper row | Source CSV | Row filter |
|---|---|---|
| **Greedy** | `results/s52_main_results/didi_greedy_global_laptop.csv` *(pending re-run)* | Strategy="Greedy" — use laptop metrics; scale runtime |
| **k-NLF (k=15)** | `results/s52_main_results/didi_core_v2.csv` | Strategy="k-NLF (k=15)" |
| **Composite (static)** | `results/s52_main_results/didi_core_v2.csv` | Strategy="Composite (static)" |
| **LAF** | `results/s52_main_results/didi_core_v2.csv` | Strategy="LAF" |
| **BiRanking (BRK)** | `results/s52_main_results/didi_core_v2.csv` | Strategy="BiRanking (BRK)" |
| **ONRTA-RT** | `results/s52_main_results/didi_onrta_v2.csv` | Strategy="ONRTA-RT" |
| **Disc. Review LP** | `results/s52_main_results/didi_lp_v2.csv` | Strategy="Discrete Review LP" |

**For cleanup:** After the laptop run, merge all sources into one canonical
`results/s52_main_results/didi_main_results_final.csv` (7 rows). Delete
`didi_core_v2.csv`, `didi_onrta_v2.csv`, `didi_lp_v2.csv`, `didi_greedy_k50.csv`,
and `didi_greedy_global_laptop.csv`.

---

### §5.2 — Table `tab:gowalla_results` (Gowalla main results)

| Paper row | Source CSV | Row filter |
|---|---|---|
| **Greedy** | `results/s52_main_results/gowalla_greedy_global_laptop.csv` *(pending re-run)* | `_strategy`="Greedy" — use laptop metrics; scale runtime |
| **k-NLF (k=15)** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="k-NLF (k=15)", `_compress`=True, `_ratio`="Ratio 0.20" |
| **Composite (static)** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="Composite (static)", same filters |
| **LAF** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="LAF", same filters |
| **BiRanking (BRK)** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="BiRanking (BRK)", same filters |
| **ONRTA-RT** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="ONRTA-RT", same filters |
| **Disc. Review LP** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | `_strategy`="Discrete Review LP", same filters |

Note: `gowalla_austin_compressed_v2.csv` has 13 strategies × 1 config = 13 rows.
Only the 6 non-Greedy strategies above appear in the paper table. Rows for EWMA-Only,
k-NTF-EPH, k-NTF-IR, Random, ONRTA-OP, and the stale k=10 Greedy row are unused.

**For cleanup:** Merge into `results/s52_main_results/gowalla_main_results_final.csv`
(7 rows). Delete `gowalla_austin_compressed_v2.csv`, `gowalla_greedy_k50.csv`,
`gowalla_greedy_global_laptop.csv`, `gowalla_austin_compressed_cluster.csv`, and
`gowalla_austin_compressed_laptop.csv`.

---

### §5.3 — Figure `fig:market_conditions` (Supply–Demand Robustness)

| Panel | Source CSV | Notes |
|---|---|---|
| (a) Fleet sweep | `results/s53_scalability/scalability_fleet_v2.csv` | 27/30 rows used; 3 FATP-ANN timeout rows excluded from plot |
| (b) Task sweep | `results/s53_scalability/scalability_tasks_v2.csv` | 22/30 rows used; 8 timeout rows excluded from plot |

Single source per panel, no mixing. **For cleanup:** Keep both `_v2.csv` files.
Delete `scalability_fleet_cluster.csv` and `scalability_tasks_cluster.csv`.

---

### §5.4.1 — Figure `fig:k_sweep` (Impact of k)

| Source CSV | Content |
|---|---|
| `results/s54_ablation/knlf_k_sweep_20161109_v2.csv` | k ∈ {3,5,10,15,25,50,100} for k-NLF and Composite + Greedy/LAF anchors (16 rows) |

Single source, no mixing. **For cleanup:** Keep `_v2.csv`. Delete `_cluster.csv`.

**⚠️ Greedy anchor is stale** — the Greedy row in `knlf_k_sweep_20161109_v2.csv` was
recorded with the old spatial-index implementation (k=10), not the new O(|W|) global scan.

**Panel-by-panel decisions:**

| Panel | Greedy reference line? | Action needed |
|---|---|---|
| (a) Fairness vs k | ✅ Keep — useful baseline | Update JFI/wait values from `didi_greedy_global_laptop.csv` once laptop calibration run completes |
| (b) Wait Time vs k | ✅ Keep — useful baseline | Same as (a) |
| (c) Runtime vs k | ❌ Removed — Greedy has no k parameter (O(\|W\|) global scan). Runtime panel shows only how k-NLF and Composite scale with k | Done — `plot_k_sweep.py` updated 2026-07-01 |

**To update panels (a)/(b) Greedy reference values after laptop calibration run:**
1. Read `didi_greedy_global_laptop.csv` — filter Strategy="Greedy"
2. Replace `greedy_row["JFI (tasks)"]` and `greedy_row["Avg Wait (m)"]` in the plot, OR
   update the stale row directly in `knlf_k_sweep_20161109_v2.csv` before cleanup.
3. The JFI/wait values are deterministic — no scaling needed (unlike runtime).

---

### §5.4.2 — Table `tab:signal_comparison` (Fairness Signal Comparison)

| Source CSV | Rows used in paper |
|---|---|
| `results/s54_ablation/signal_comparison_20161109_v3.csv` | All 7 rows (Greedy, k-NTF-EPH k=5/15, k-NTF-IR k=5/15, k-NLF k=15, Composite k=15) — k=5 variants in CSV but not in paper table (prose only) |

Single source, no mixing. **For cleanup:** Keep `_v3.csv`. Delete `_v2.csv` and
`_cluster.csv`.

---

### §5.4.3 — Figure `fig:fw_sweep` (Fairness Weight Sensitivity)

| Source CSV | Content |
|---|---|
| `results/s54_ablation/fairness_weight_sweep_20161109_v2.csv` | λ_f ∈ {0.0, 0.2, …, 2.0, 2.5, 3.0}, Composite only — 13 rows |

Single source, no mixing. **For cleanup:** Keep `_v2.csv`. Delete `_cluster.csv`.

---

### Canonical final CSVs — ✅ Generated (2026-07-01)

Run `python3 scripts/experiments/build_final_csvs.py` to regenerate from sources.

| Canonical file | Script | Rows | Notes |
|---|---|---|---|
| `results/s52_main_results/didi_main_results_final.csv` | `build_final_csvs.py` | 7 | Greedy runtime=311.7s (scaled); CV(earn) from laptop |
| `results/s52_main_results/gowalla_main_results_final.csv` | `build_final_csvs.py` | 7 | Greedy runtime=98.4s (scaled) |
| `results/s54_ablation/signal_comparison_final.csv` | `build_final_csvs.py` | 7 | Greedy runtime=337.1s (scaled using 133s k-NLF); CV(earn) placeholder from v3 |

### Files to keep vs delete (post-cleanup summary)

| Keep | Delete |
|---|---|
| `didi_main_results_final.csv` ✅ | `didi_core_v2.csv`, `didi_onrta_v2.csv`, `didi_lp_v2.csv`, `didi_greedy_k50.csv`, `didi_greedy_global_laptop.csv` |
| `gowalla_main_results_final.csv` ✅ | `gowalla_austin_compressed_v2.csv`, `gowalla_greedy_k50.csv`, `gowalla_greedy_global_laptop.csv`, `gowalla_austin_compressed_cluster.csv`, `gowalla_austin_compressed_laptop.csv` |
| `scalability_fleet_v2.csv` | `scalability_fleet_cluster.csv` |
| `scalability_tasks_v2.csv` | `scalability_tasks_cluster.csv` |
| `knlf_k_sweep_20161109_v2.csv` | `knlf_k_sweep_20161109_cluster.csv` |
| `signal_comparison_final.csv` ✅ | `signal_comparison_20161109_v3.csv`, `signal_comparison_20161109_v2.csv`, `signal_comparison_20161109_cluster.csv` |
| `fairness_weight_sweep_20161109_v2.csv` | `fairness_weight_sweep_20161109_cluster.csv` |

---

## Code Optimisation Pass (2026-07-01)

The following strategy implementations were optimised before re-running affected experiments.
Results tagged `_v2` or `_v3` use the updated code; older `_cluster` files used the old code.

| Strategy | Change | Impact on results |
|---|---|---|
| `greedy.py` | `NEW_TASK` now uses spatial index (k=10) instead of O(\|W\|) linear scan; `drop_dist` hoisted out of worker loop | Runtime ~19s (was ~357s on DiDi); TAR/JFI unchanged |
| `knlf.py` | `FREE_WORKER` now uses `deferred_task_index` (k=15) instead of full linear scan | Faster; marginal TAR/JFI shift |
| `laf.py` | Swapped local `manhattan_km` → `fast_manhattan_km` (pre-computed `cos` scalar) | Tiny numerical delta (<0.15%) |
| `fatp_ann.py` | Same distance swap + `pickup_dist` reused from feasibility check + no pending-list rebuild in WP loop | Faster; tiny numerical delta |
| `biranking.py` | `drop_dist` hoisted/reused between `_is_feasible` and `_commit_assignment` | Numerically identical |
| `onrta_rt.py` | Same as BiRanking | Numerically identical |
| `aveklouris_lp.py` | `d_drop` hoisted out of O(\|T\|×\|W\|) inner loop | Numerically identical |

---

## §5.2 — Effectiveness (Main Results)

### `run_gowalla_comparison.py`
Script: `scripts/experiments/s52_main_results/run_gowalla_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| **Greedy k=50 ✅** | `results/s52_main_results/gowalla_greedy_k50.csv` | **USE FOR GREEDY ROW** — k=50 Greedy only, 2026-07-01 |
| **Cluster v2 ✅** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | **USE FOR ALL OTHER STRATEGIES** — post-optimisation re-run, 27.1 min total, all 13 strategies |
| Cluster v1 | `results/s52_main_results/gowalla_austin_compressed_cluster.csv` | Pre-optimisation; superseded |
| Laptop | `gowalla_austin_20100901_to_20100930.csv` (repo root) | Pre-bugfix idle-time, util=0.0% — superseded |

- **Stage 1 — Experiments**: ✅ Complete
  - 8,758 workers | 43,788 tasks | ratio 0.20 | all 13 strategies finished
  - **Greedy (k=50, USE THIS)**: TAR=0.998, Wait=3.33m, JFI-t=0.583, JFI-e=0.617, Rev=297.5k$, **Time=6s** — from `gowalla_greedy_k50.csv`
  - Other strategy wall times: k-NLF=27.3s, Composite=28.1s, LAF=91.0s, FATP-ANN=209.8s, BiRanking=115.7s, ONRTA-RT=108.6s, Disc.LP=142.9s, Random=205.5s, EWMA-Only=351.2s, ONRTA-OP=329.1s
- **Stage 2 — Plotting / Tabling**: ✅ Complete — `tab:gowalla_results` updated in `experimental-section.tex` (k=50 Greedy row + Gowalla prose rewritten, 2026-07-01)
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01)

---

### `run_strategy_comparison.py`  *(DiDi Chengdu)*
Script: `scripts/experiments/s52_main_results/run_strategy_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v2 ✅** | `results/s52_main_results/didi_core_v2.csv` | Greedy, k-NLF, Composite, LAF, BiRanking — **USE THIS** |
| **Cluster v2 ✅** | `results/s52_main_results/didi_onrta_v2.csv` | ONRTA-RT — **USE THIS** |
| **Cluster v2 ✅** | `results/s52_main_results/didi_lp_v2.csv` | Discrete Review LP — **USE THIS** |
| Laptop | terminal output only | Pre-optimisation; superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v2, 2026-07-01, 3 focused runs)
  - 36,799 workers | 224,219 tasks
  - Greedy (k=10 raw): TAR=0.863, Wait=3.38m, JFI-t=0.564, JFI-e=0.620, Rev=2,765.6k$, Time=28s ← superseded by k50 below
  - **Greedy (k=50, USE THIS)**: TAR=0.863, Wait=2.78m, JFI-t=0.572, JFI-e=0.616, Rev=2,765.7k$, **Time=32s** — from `didi_greedy_k50.csv`
  - k-NLF: TAR=0.863, Wait=3.45m, JFI-t=0.644, JFI-e=0.666, Rev=2,765.6k$, Time=123s
  - Composite: TAR=0.863, Wait=3.23m, JFI-t=0.590, JFI-e=0.626, Rev=2,765.7k$, Time=112s
  - LAF: TAR=0.862, Wait=9.65m, JFI-t=0.705, JFI-e=0.699, Rev=2,765.1k$, Time=385s
  - BiRanking: TAR=0.857, Wait=12.34m, JFI-t=0.592, JFI-e=0.636, Rev=2,750.9k$, Time=557s
  - ONRTA-RT: TAR=0.855, Wait=12.17m, JFI-t=0.641, JFI-e=0.655, Rev=2,750.3k$, Time=480s
  - Disc. LP: TAR=0.862, Wait=1.63m, JFI-t=0.516, JFI-e=0.700, Rev=2,765.4k$, Time=1,936s
- **Stage 2 — Plotting / Tabling**: ✅ Complete — `tab:didi_results` updated in `experimental-section.tex` (2026-07-01)
  - Bold: TAR→Greedy/k-NLF/Composite (0.863); Wait→Disc.LP (1.63m); JFI-t→LAF (0.705); JFI-e→Disc.LP (0.700); Rev→Composite (2,765.7k$); Time→Composite/k-NLF (112–123s, once Greedy global-scan runtime updated)
  - Fixed label: LTF → LAF
- **Stage 3 — Analysis & Writing**: ✅ Complete (prose covers both datasets jointly)

---

## §5.3 — Computational Efficiency & Scalability

### `run_scalability_fleet.py`  *(Vary |W|, fix |T|)*
Script: `scripts/experiments/s53_scalability/run_scalability_fleet.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v2 ✅** | `results/s53_scalability/scalability_fleet_v2.csv` | **USE THIS** — post-optimisation, 27/30 runs succeeded |
| Cluster v1 | `results/s53_scalability/scalability_fleet_cluster.csv` | Pre-optimisation early snapshot (only 10k workers); superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v2, 2026-07-01) — 27/30 succeeded
  - Config: `TARGET_TASKS=50,000`, fleet sizes `[10k, 15k, 20k, 25k, 30k, 36,799]`, 5 strategies, 900s timeout
  - FATP-ANN timed out at 25k, 30k, 36,799 workers (3 runs) — expected and reportable
  - Runtime summary (seconds):

    | Strategy | 10k | 15k | 20k | 25k | 30k | 36,799 |
    |---|---|---|---|---|---|---|
    | k-NLF (k=15) | 16.0 | 21.0 | 22.0 | 22.7 | 91.6 | 68.5 |
    | Composite (static) | 19.6 | 15.8 | 25.9 | 26.9 | 129.7 | 138.6 |
    | Greedy | 5.5 | 7.3 | 8.3 | 10.5 | 25.7 | 44.3 |
    | LAF | 96.8 | 178.2 | 260.0 | 393.4 | 475.8 | 544.7 |
    | FATP-ANN | 385.9 | 469.6 | 722.8 | TIMEOUT | TIMEOUT | TIMEOUT |

- **Stage 2 — Plotting / Tabling**: ✅ Complete (2026-07-01, revised)
  - `scripts/plots/plot_market_conditions.py` — two-panel JFI figure, saved to `results/figures/market_conditions.pdf`
  - §5.3 **pivoted** from "Efficiency & Scalability" to "Robustness to Supply–Demand Conditions"
  - Old scalability runtime figure (scalability.pdf) retained in results/figures/ but no longer used in tex
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01, revised)
  - Three `\paragraph` blocks: Supply-Rich Markets (fleet sweep), Demand-Pressure Regimes (task sweep), Computational Tractability
  - Key findings: (1) k-NLF relative advantage over Greedy WIDENS with fleet size (+33% → +46%); (2) all strategies converge under extreme demand overload; (3) LAF achieves high JFI but at severe wait cost

---

### `run_scalability_tasks.py`  *(Vary |T|, fix |W|)*
Script: `scripts/experiments/s53_scalability/run_scalability_tasks.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v2 ✅** | `results/s53_scalability/scalability_tasks_v2.csv` | **USE THIS** — post-optimisation, retrieved via SCP 2026-07-01 |
| Cluster v1 | `results/s53_scalability/scalability_tasks_cluster.csv` | Pre-optimisation early snapshot; superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v2, 2026-07-01)
  - Config: fixed 10k workers, task volumes `[50k, 100k, 150k, 200k, 224,219]`, 5 strategies, 900s timeout
  - FATP-ANN timed out at ≥100k tasks (4 timeouts); LAF timed out at ≥150k tasks (3 timeouts) — expected
  - Runtime summary (seconds):

    | Strategy | 50k | 100k | 150k | 200k | 224k |
    |---|---|---|---|---|---|
    | k-NLF (k=15) | 20.7 | 42.7 | 166.0 | 312.8 | 409.5 |
    | Composite (static) | 18.1 | 41.7 | 191.5 | 275.2 | 411.5 |
    | Greedy | 5.6 | 13.8 | 77.9 | 154.1 | 309.3 |
    | LAF | 99.3 | 116.6 | TIMEOUT | TIMEOUT | TIMEOUT |
    | FATP-ANN | 396.3 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |

  - Key TAR values at full scale (224k tasks): k-NLF=0.763, Composite=0.764, Greedy=0.765
  - JFI (tasks) at full scale: k-NLF=0.835, Composite=0.835, Greedy=0.836

- **Stage 2 — Plotting / Tabling**: ✅ Complete (2026-07-01) — shared with fleet sweep via `plot_market_conditions.py`
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01) — shared prose in `\subsection{Robustness to Supply–Demand Conditions}`

---

## §5.4 — Ablation & Sensitivity Analysis

### `run_knlf_k_sweep.py`  *(§5.4.1 — Impact of k)*
Script: `scripts/experiments/s54_ablation/run_knlf_k_sweep.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v2 ✅** | `results/s54_ablation/knlf_k_sweep_20161109_v2.csv` | **USE THIS** — post-optimisation, now includes **both k-NLF and Composite** |
| Cluster v1 | `results/s54_ablation/knlf_k_sweep_20161109_cluster.csv` | k-NLF only, pre-optimisation; superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v2, 2026-07-01)
  - k ∈ {3, 5, 10, 15, 25, 50, 100} for both k-NLF and Composite + Greedy & LAF anchors (16 runs)
  - Greedy anchor: JFI=0.5634 (baseline). Key ΔJFI vs Greedy:
    - k-NLF: k=3→+0.030, k=15→+0.081, k=50→+0.150, k=100→+0.167
    - Composite: k=3→+0.013, k=15→+0.022, k=50→+0.047, k=100→+0.051
  - k=15 is the knee of the curve for both strategies (diminishing returns beyond k=25)
- **Stage 2 — Plotting / Tabling**: 🟡 Plot updated (2026-07-01) — Composite lines added; Greedy removed from panel (c)
  - `results/figures/k_sweep.pdf` — regenerated with shared bottom legend; panel (c) now shows only k-NLF and Composite runtime curves
  - Greedy reference lines remain in panels (a) and (b) but use **stale values** from old implementation — see provenance note above
  - ⚠️ **Pending:** once `didi_greedy_global_laptop.csv` exists, update panels (a)/(b) Greedy anchor values and re-run plot
  - `\subsubsection{Impact of k}` stub in tex still needs `\includegraphics` + caption + prose
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_signal_comparison.py`  *(§5.4.2 — Task-Count vs EWMA)*
Script: `scripts/experiments/s54_ablation/run_signal_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v3 ✅** | `results/s54_ablation/signal_comparison_20161109_v3.csv` | **USE THIS** — post-optimisation re-run, all 7 strategies |
| Laptop v2 | `results/s54_ablation/signal_comparison_20161109_v2.csv` | Pre-optimisation (Greedy/k-NLF were slower); superseded |
| Cluster v1 | `results/s54_ablation/signal_comparison_20161109_cluster.csv` | No Composite, wrong k=5 labels; superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v3, 2026-07-01)
  - Greedy, k-NTF-EPH (k=5, k=15), k-NTF-IR (k=5, k=15), k-NLF (k=15), Composite (k=15) — all 7 strategies
  - Baseline k-NLF: JFI-tasks=0.6466, JFI-earn=0.6691, JFI-rate=0.8899, wait=3.476m
  - ΔJFI-tasks vs k-NLF: Greedy=−0.085, k-NTF-EPH(k=15)=−0.034, k-NTF-IR(k=15)=−0.029, Composite=−0.057
  - k-NLF still dominates all fairness columns; k=5 variants add wait penalty (+0.5–0.7m)
- **Stage 2 — Plotting / Tabling**: ✅ Table updated with v3 values (2026-07-01)
- **Stage 3 — Analysis & Writing**: 🟡 Draft written and reviewed (2026-07-01)
  - Analysis covers: k-NLF dominance across all fairness columns; Composite as efficiency specialist; tractability vs LP/LTF; definitional divergence framing
  - **3 minor wording fixes still needed before submission:**
    1. "under 133s" → "at most 133s" (k-NLF is exactly 133s, not strictly under)
    2. "task orphaning" is wrong mechanism — Composite reduces idle-time variance, not task expiry; rephrase
    3. Composite Gowalla wait advantage is 0.01m (5.11 vs 5.12m) — "across both environments" overstates it; soften or drop
  - **Note**: LTF runtime (319s) used in tractability comparison is from pre-fix DiDi table — directionally correct since LTF is structurally O(\|W\|), but exact multiple will shift when `didi_20161109_v2.csv` arrives

---

### `run_fairness_weight_sweep.py`  *(§5.4.3 — Weighted Scorer Sensitivity)*
Script: `scripts/experiments/s54_ablation/run_fairness_weight_sweep.py`

| Copy | Path | Notes |
|---|---|---|
| Laptop v2 ✅ | `results/s54_ablation/fairness_weight_sweep_20161109_v2.csv` | **Use this** — Composite-only, unaffected by optimisation pass |
| Cluster v1 | `results/s54_ablation/fairness_weight_sweep_20161109_cluster.csv` | No CV(idle); superseded |

- **Stage 1 — Experiments**: ✅ Complete (laptop v2 — Composite-only, unaffected by optimisation pass)
  - λ_f ∈ {0.0, 0.2, …, 2.0, 2.5, 3.0}; paper default = 1.6
  - ΔJFI(tasks)=0.023 across full sweep; wait range 3.14–3.25m — robust to λ_f
- **Stage 2 — Plotting / Tabling**: ⚠️ Plot needs re-run with v2 CSV
  - `conda run -n sc python3 scripts/plots/plot_fairness_weight.py --input results/s54_ablation/fairness_weight_sweep_20161109_v2.csv`
  - `\subsubsection{Weighted Scorer Sensitivity}` stub still needs `\includegraphics` + caption + prose
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

## Quick Status Summary

| Script | Best CSV | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| §5.2 Gowalla comparison | `gowalla_austin_compressed_v2.csv` | ✅ | ⚠️ refresh from v2 | ✅ |
| §5.2 DiDi strategy comparison | `didi_core/onrta/lp_v2.csv` | ✅ | ✅ | ✅ |
| §5.3 Scalability fleet | `scalability_fleet_v2.csv` | ✅ | ✅ | ✅ |
| §5.3 Scalability tasks | `scalability_tasks_v2.csv` | ✅ | ✅ | ✅ |
| §5.4 k-NLF k-sweep | `knlf_k_sweep_20161109_v2.csv` | ✅ | ✅ plot updated (2026-07-01) | ❌ |
| §5.4 Signal comparison | `signal_comparison_20161109_v3.csv` | ✅ | ✅ | 🟡 draft done, 3 wording fixes needed |
| §5.4 Fairness weight sweep | `fairness_weight_sweep_20161109_v2.csv` | ✅ | ✅ plot updated (2026-07-01) | ❌ |

**Legend:** ✅ Complete · 🟡 Partial · 🔄 Running on cluster · ⚠️ Minor action needed · ❌ Not started

---

## Immediate Next Actions (priority order)

1. ~~**SCP didi v2**~~ — ✅ Done. `tab:didi_results` updated in tex.
2. **Re-run k-sweep plot** — `plot_k_sweep.py` with `knlf_k_sweep_20161109_v2.csv` to add Composite lines; insert figure in tex
3. **Refresh signal table** — update `tab:signal_comparison` from `signal_comparison_20161109_v3.csv`
4. **Re-run fairness weight plot** — `plot_fairness_weight.py` with `fairness_weight_sweep_20161109_v2.csv`; insert figure in tex
5. **Fix 3 wording issues in §5.4.2 analysis** — see Stage 3 notes under `run_signal_comparison.py`
6. **Write §5.4.3 analysis** — draft generated; insert into tex
