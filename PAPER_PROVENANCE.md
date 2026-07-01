# Paper Data Provenance & Codebase Cleanup Guide

> Last updated: 2026-07-01
>
> This document maps every paper table and figure to its exact source CSV(s),
> explains version history and the Greedy k=10→k=50 correction, and provides a
> canonical "keep vs. delete" guide for repository cleanup before submission.

---

## 1. Why Some Tables Have Mixed-CSV Provenance

All experiments were originally run on an AWS EC2 cluster using `config.py`'s
default `STRATEGY_PARAMS["greedy"]["k"] = 10`. This means Greedy's `NEW_TASK`
handler queried only the 10 nearest workers via spatial index — a value that is
adequate for timing benchmarks but too restrictive for correctness on the full
DiDi dataset (36,799 workers, thousands simultaneously available).

With k=10, Greedy defers tasks more frequently (the true nearest feasible worker
is often ranked 11th or beyond), inflating mean wait time from the expected ~2.62m
to ~3.38m and depressing JFI (tasks) from ~0.594 to ~0.564. This created a false
impression that Composite achieved lower wait time than Greedy.

**Fix applied (2026-07-01):** `config.py` greedy `k` changed from 10 → 50.
Greedy was re-run in isolation producing `didi_greedy_k50.csv` (local laptop,
31.5s). All other strategy rows remain from the cluster runs. The k=50 runtime
(32s) is still a 10× speedup over the original O(W) scan (357s on the same day),
which is the paper's primary computational efficiency claim.

**Gowalla note:** The same k=10 artifact exists for the Gowalla table
(`gowalla_greedy_k50.csv` is already present and shows wait=3.33m vs 5.12m in
the current table). The Gowalla table has not yet been updated — see §5 (Pending
Actions) below.

---

## 2. Per-Table / Figure Provenance Maps

### `tab:didi_results` — DiDi Strategy Comparison (7 paper rows, 4 source CSVs)

| Paper row | Strategy= filter | Source CSV | Run env | Notes |
|---|---|---|---|---|
| Greedy (baseline) | `Greedy` | `results/s52_main_results/didi_greedy_k50.csv` | Laptop k=50 | k=50 corrected run; 32s |
| k-NLF (k=15) | `k-NLF (k=15)` | `results/s52_main_results/didi_core_v2.csv` | EC2 cluster | post-optimisation |
| Composite (static) | `Composite (static)` | `results/s52_main_results/didi_core_v2.csv` | EC2 cluster | post-optimisation |
| LAF | `LAF` | `results/s52_main_results/didi_core_v2.csv` | EC2 cluster | fast\_manhattan\_km fix |
| BiRanking (BRK) | `BiRanking (BRK)` | `results/s52_main_results/didi_core_v2.csv` | EC2 cluster | numerically identical |
| ONRTA-RT | `ONRTA-RT` | `results/s52_main_results/didi_onrta_v2.csv` | EC2 cluster | numerically identical |
| Disc. Review LP | `Discrete Review LP` | `results/s52_main_results/didi_lp_v2.csv` | EC2 cluster | d\_drop hoist fix; 1936s |

---

### `tab:gowalla_results` — Gowalla Strategy Comparison (7 paper rows, 1 source CSV)

All rows sourced from: `results/s52_main_results/gowalla_austin_compressed_v2.csv`

Filter columns: `_compress=True`, `_ratio=0.20`, `_region=austin`

| Paper row | `_strategy` filter value |
|---|---|
| Greedy (baseline) | `Greedy` |
| k-NLF (k=15) | `k-NLF (k=15)` |
| Composite (static) | `Composite (static)` |
| LAF | `LAF` |
| BiRanking (BRK) | `BiRanking (BRK)` |
| ONRTA-RT | `ONRTA-RT` |
| Disc. Review LP | `Discrete Review LP` |

⚠️ **Pending:** Gowalla Greedy row uses k=10 values (wait=5.12m). Corrected k=50
result exists in `gowalla_greedy_k50.csv` (wait=3.33m, JFI=0.583). Gowalla table
and prose need updating before final submission — see §5.

---

### `fig:market_conditions` — Supply/Demand Scalability (2-panel figure)

| Panel | Source CSV | Filter |
|---|---|---|
| (a) Fleet sweep — vary \|W\|, fix \|T\|=50k | `results/s53_scalability/scalability_fleet_v2.csv` | All rows |
| (b) Task sweep — fix \|W\|=10k, vary \|T\| | `results/s53_scalability/scalability_tasks_v2.csv` | All rows |

Plot script: `scripts/plots/plot_scalability.py`
Output: `results/figures/market_conditions.pdf`

Strategies shown: k-NLF (k=15), Composite (static), Greedy.
LAF and FATP-ANN are in the CSVs but excluded from the figure (timed out at larger
scales — timeout rows have `TAR=None`).

---

### `fig:k_sweep` — Candidate Pool Size (k) Sweep (3-panel figure)

Source CSV: `results/s54_ablation/knlf_k_sweep_20161109_v2.csv` (single source, 16 rows)

| Rows used | Description |
|---|---|
| `label` contains `k-NLF k=` | k-NLF sweep, k ∈ {3, 5, 10, 15, 25, 50, 100} |
| `label` contains `Composite k=` | Composite sweep, k ∈ {3, 5, 10, 15, 25, 50, 100} |
| `label = "Greedy (k=∞, dist)"` | Greedy anchor (dashed reference line) |
| `label = "LAF (k=W, count)"` | LAF anchor (dashed reference line) |

Plot script: `scripts/plots/plot_k_sweep.py`
Output: `results/figures/k_sweep.pdf`

**Note:** The Greedy anchor in this CSV used k=10 from config (JFI=0.563, wait=3.41m).
The corrected k=50 anchor gives JFI≈0.572 — a shift of +0.009 in the dashed reference
line, visually negligible. The figure does not need to be regenerated for submission.

---

### `tab:signal_comparison` — Fairness Signal Comparison (5 paper rows, 1 source CSV)

Source CSV: `results/s54_ablation/signal_comparison_20161109_v4.csv` (7 rows total)

| Paper row | `strategy` column value | In paper? |
|---|---|---|
| Greedy (baseline) | `Greedy` | ✅ Yes |
| k-NTF-EPH (k=15) | `k-NTF-EPH (k=15)` | ✅ Yes |
| k-NTF-IR (k=15) | `k-NTF-IR  (k=15)` | ✅ Yes |
| k-NLF (k=15) | `k-NLF (k=15)` | ✅ Yes |
| Composite (k=15) | `Composite (k=15)` | ✅ Yes |
| k-NTF-EPH (k=5) | `k-NTF-EPH (k=5)` | ❌ Not in paper table (CSV only) |
| k-NTF-IR (k=5) | `k-NTF-IR  (k=5)` | ❌ Not in paper table (CSV only) |

---

### `fig:fw_sweep` — Fairness Weight Sensitivity (2-panel figure)

Source CSV: `results/s54_ablation/fairness_weight_sweep_20161109_v2.csv` (single source)

Plot script: `scripts/plots/plot_fairness_weight.py`
Output: `results/figures/fairness_weight_sweep.pdf`

Sweep: λ_f ∈ {0.0, 0.2, 0.4, …, 2.0, 2.5, 3.0}; all rows used.
Note: figure was generated on laptop. Composite-only; unaffected by any Greedy k fix.

---

## 3. Cleanup Instructions Per Section

### `results/s52_main_results/`

**Keep:**
- `didi_greedy_k50.csv` — Greedy row in `tab:didi_results`
- `didi_core_v2.csv` — k-NLF, Composite, LAF, BiRanking rows
- `didi_onrta_v2.csv` — ONRTA-RT row
- `didi_lp_v2.csv` — Disc. Review LP row
- `gowalla_austin_compressed_v2.csv` — all Gowalla rows (current)
- `gowalla_greedy_k50.csv` — Gowalla Greedy corrected (pending update)

**Delete (superseded):**
- `gowalla_austin_compressed_cluster.csv` — pre-optimisation, superseded by v2
- `gowalla_austin_compressed_laptop.csv` — pre-bugfix idle-time, util=0.0%
- `log_gowalla_laptop.log` — log file, not a result

---

### `results/s53_scalability/`

**Keep:**
- `scalability_fleet_v2.csv` — feeds `fig:market_conditions` panel (a)
- `scalability_tasks_v2.csv` — feeds `fig:market_conditions` panel (b)

**Delete (superseded):**
- `scalability_fleet_cluster.csv` — early snapshot (only 10k workers), pre-optimisation
- `scalability_tasks_cluster.csv` — early snapshot (50k + 100k tasks only), pre-optimisation

---

### `results/s54_ablation/`

**Keep:**
- `signal_comparison_20161109_v4.csv` — feeds `tab:signal_comparison` (k=50 Greedy)
- `knlf_k_sweep_20161109_v2.csv` — feeds `fig:k_sweep`
- `fairness_weight_sweep_20161109_v2.csv` — feeds `fig:fw_sweep`

**Delete (superseded):**
- `signal_comparison_20161109_cluster.csv` — no Composite, wrong k=5 labels
- `signal_comparison_20161109_v2.csv` — k=10 Greedy, superseded by v4
- `signal_comparison_20161109_v3.csv` — k=10 Greedy, superseded by v4
- `knlf_k_sweep_20161109_cluster.csv` — k-NLF only, no Composite; superseded by v2
- `fairness_weight_sweep_20161109_cluster.csv` — no CV(idle); superseded by v2
- `log_fw_sweep_v2.log` — log file, not a result
- `log_signal_v2.log` — log file, not a result

---

### `results/figures/`

**Keep all** — these are the paper figures. All are currently up to date except:
- `fairness_weight_sweep.pdf/.png` — generated from v2 CSV; numerically correct but
  may be from an earlier plot run; regenerate with:
  `conda run -n sc python3 scripts/plots/plot_fairness_weight.py --input results/s54_ablation/fairness_weight_sweep_20161109_v2.csv`
- `k_sweep.pdf/.png` — Greedy dashed line uses k=10 anchor (JFI=0.563); corrected
  anchor (JFI=0.572) shift is +0.009, visually negligible; no regeneration required.

---

### Old result directories (stale section numbering)

The following directories use old section numbering (`s1_`, `s2_`, `s3_`, `s4_`,
`s2_ewma_validation`, `s3_ablation`) and pre-date the current paper structure.
Their contents are **not referenced** by any current paper table or figure.

**Safe to delete entire directories:**
- `results/s1_overall_performance/` (`.gitkeep` only)
- `results/s2_ewma_validation/`
- `results/s3_ablation/`
- `results/s4_robustness/` (`.gitkeep` only)
- `results/starvation_ablation/`
- `results/supervisor/`
- `results/parameter_tuning/`

---

## 4. Keep vs. Delete Summary

| File | Keep? | Reason |
|---|---|---|
| `s52_main_results/didi_greedy_k50.csv` | ✅ Keep | Paper source (Greedy row, DiDi) |
| `s52_main_results/didi_core_v2.csv` | ✅ Keep | Paper source (k-NLF, Composite, LAF, BRK) |
| `s52_main_results/didi_onrta_v2.csv` | ✅ Keep | Paper source (ONRTA-RT) |
| `s52_main_results/didi_lp_v2.csv` | ✅ Keep | Paper source (Disc. LP) |
| `s52_main_results/gowalla_austin_compressed_v2.csv` | ✅ Keep | Paper source (all Gowalla rows) |
| `s52_main_results/gowalla_greedy_k50.csv` | ✅ Keep | Pending Gowalla table update |
| `s52_main_results/gowalla_austin_compressed_cluster.csv` | ❌ Delete | Superseded by v2 |
| `s52_main_results/gowalla_austin_compressed_laptop.csv` | ❌ Delete | Pre-bugfix data |
| `s52_main_results/didi_greedy_global_laptop.csv` | ✅ Keep | Historical k=∞ reference |
| `s52_main_results/gowalla_greedy_global_laptop.csv` | ✅ Keep | Historical k=∞ reference |
| `s52_main_results/log_gowalla_laptop.log` | ❌ Delete | Log file |
| `s53_scalability/scalability_fleet_v2.csv` | ✅ Keep | Paper source (fig:market_conditions a) |
| `s53_scalability/scalability_tasks_v2.csv` | ✅ Keep | Paper source (fig:market_conditions b) |
| `s53_scalability/scalability_fleet_cluster.csv` | ❌ Delete | Incomplete early snapshot |
| `s53_scalability/scalability_tasks_cluster.csv` | ❌ Delete | Incomplete early snapshot |
| `s54_ablation/signal_comparison_20161109_v4.csv` | ✅ Keep | Paper source (tab:signal_comparison) |
| `s54_ablation/knlf_k_sweep_20161109_v2.csv` | ✅ Keep | Paper source (fig:k_sweep) |
| `s54_ablation/fairness_weight_sweep_20161109_v2.csv` | ✅ Keep | Paper source (fig:fw_sweep) |
| `s54_ablation/signal_comparison_20161109_cluster.csv` | ❌ Delete | Superseded |
| `s54_ablation/signal_comparison_20161109_v2.csv` | ❌ Delete | k=10 Greedy; superseded by v4 |
| `s54_ablation/signal_comparison_20161109_v3.csv` | ❌ Delete | k=10 Greedy; superseded by v4 |
| `s54_ablation/knlf_k_sweep_20161109_cluster.csv` | ❌ Delete | k-NLF only; superseded by v2 |
| `s54_ablation/fairness_weight_sweep_20161109_cluster.csv` | ❌ Delete | No CV(idle); superseded by v2 |
| `s54_ablation/log_fw_sweep_v2.log` | ❌ Delete | Log file |
| `s54_ablation/log_signal_v2.log` | ❌ Delete | Log file |
| `results/s1_overall_performance/` | ❌ Delete | Stale section numbering; not referenced |
| `results/s2_ewma_validation/` | ❌ Delete | Stale section numbering; not referenced |
| `results/s3_ablation/` | ❌ Delete | Stale section numbering; not referenced |
| `results/s4_robustness/` | ❌ Delete | Stale section numbering; not referenced |
| `results/starvation_ablation/` | ❌ Delete | Not referenced in paper |
| `results/supervisor/` | ❌ Delete | Not referenced in paper |
| `results/parameter_tuning/` | ❌ Delete | Not referenced in paper |

---

## 5. Pending Actions Before Submission

1. **Gowalla Greedy table update** — `gowalla_greedy_k50.csv` exists (wait=3.33m,
   JFI=0.583) but the table still shows k=10 values (wait=5.12m, JFI=0.591).
   Update `tab:gowalla_results` Greedy row and revise §5.2 Gowalla prose accordingly.
   The k-NLF % fairness improvement claim over Greedy will shift slightly.

2. **Fairness weight figure** — regenerate from v2 CSV to confirm figure matches
   the current data file (cosmetic; story unchanged).

3. **k-sweep plot Greedy anchor** — optional cosmetic fix: update the dashed
   Greedy reference line from JFI=0.563 (k=10) to JFI=0.572 (k=50); shift is
   +0.009, visually negligible, so this can be skipped.

4. **Delete stale files** — run the cleanup from §3 to reduce repo size and
   avoid reviewer confusion about which CSV is authoritative.
