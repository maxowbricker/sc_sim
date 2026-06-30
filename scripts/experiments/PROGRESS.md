# Experiment & Writing Progress Tracker

> Last updated: 2026-07-01 (post-optimisation re-run pass)
>
> **Stages**
> - **Stage 1 — Experiments**: raw simulation results collected
> - **Stage 2 — Plotting / Tabling**: results transferred into `experimental-section.tex`
> - **Stage 3 — Analysis & Writing**: narrative prose, claims, and discussion written

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
| **Cluster v2 ✅** | `results/s52_main_results/gowalla_austin_compressed_v2.csv` | **USE THIS** — post-optimisation re-run, 27.1 min total, all 13 strategies |
| Cluster v1 | `results/s52_main_results/gowalla_austin_compressed_cluster.csv` | Pre-optimisation; superseded |
| Laptop | `gowalla_austin_20100901_to_20100930.csv` (repo root) | Pre-bugfix idle-time, util=0.0% — superseded |

- **Stage 1 — Experiments**: ✅ Complete (cluster v2, 2026-07-01)
  - 8,758 workers | 43,788 tasks | ratio 0.20 | all 13 strategies finished
  - Wall times: Greedy=4.4s, k-NLF=27.3s, Composite=28.1s, LAF=91.0s, FATP-ANN=209.8s, BiRanking=115.7s, ONRTA-RT=108.6s, Disc.LP=142.9s, Random=205.5s, EWMA-Only=351.2s, ONRTA-OP=329.1s
- **Stage 2 — Plotting / Tabling**: ✅ Complete (values from cluster v1; **update table from v2**)
  - `tab:gowalla_results` in `experimental-section.tex` — needs refresh from `gowalla_austin_compressed_v2.csv`
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01)

---

### `run_strategy_comparison.py`  *(DiDi Chengdu)*
Script: `scripts/experiments/s52_main_results/run_strategy_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| **Cluster v2 🔄** | `results/s52_main_results/didi_20161109_v2.csv` | **In progress** — post-optimisation re-run still running (window 0 on cluster) |
| Laptop | terminal output only | Pre-optimisation; superseded |

- **Stage 1 — Experiments**: 🔄 Running on cluster (window 0)
  - 36,799 workers | 224,219 tasks | all 13 strategies
  - SCP when `=== didi DONE ===` appears: `scp -i '...macbook-m1-key.pem' ec2-user@ec2-3-26-204-128.ap-southeast-2.compute.amazonaws.com:/home/ec2-user/sc_sim/results/s52_main_results/didi_20161109_v2.csv results/s52_main_results/didi_20161109_v2.csv`
- **Stage 2 — Plotting / Tabling**: ⚠️ Table drafted from laptop values — update from v2 CSV when it arrives
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

- **Stage 2 — Plotting / Tabling**: ❌ Not started
- **Stage 3 — Analysis & Writing**: ❌ Not started

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

- **Stage 2 — Plotting / Tabling**: ❌ Not started
- **Stage 3 — Analysis & Writing**: ❌ Not started

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
- **Stage 2 — Plotting / Tabling**: 🟡 Figure exists for k-NLF only — re-run plot script with v2 CSV to add Composite lines
  - `results/figures/k_sweep.pdf` — needs update: `conda run -n sc python3 scripts/plots/plot_k_sweep.py --input results/s54_ablation/knlf_k_sweep_20161109_v2.csv`
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
| §5.2 DiDi strategy comparison | `didi_20161109_v2.csv` (🔄 running) | 🔄 | ⚠️ update when v2 arrives | ✅ |
| §5.3 Scalability fleet | `scalability_fleet_v2.csv` | ✅ | ❌ | ❌ |
| §5.3 Scalability tasks | `scalability_tasks_v2.csv` | ✅ | ❌ | ❌ |
| §5.4 k-NLF k-sweep | `knlf_k_sweep_20161109_v2.csv` | ✅ | ✅ plot updated (2026-07-01) | ❌ |
| §5.4 Signal comparison | `signal_comparison_20161109_v3.csv` | ✅ | ✅ | 🟡 draft done, 3 wording fixes needed |
| §5.4 Fairness weight sweep | `fairness_weight_sweep_20161109_v2.csv` | ✅ | ✅ plot updated (2026-07-01) | ❌ |

**Legend:** ✅ Complete · 🟡 Partial · 🔄 Running on cluster · ⚠️ Minor action needed · ❌ Not started

---

## Immediate Next Actions (priority order)

1. **SCP didi v2** — 3 focused cluster windows running (`didi_core_v2.csv`, `didi_onrta_v2.csv`, `didi_lp_v2.csv`); SCP each as it finishes and update `tab:didi_results` in tex
2. **Stage 2 §5.3** — both scalability CSVs now local; produce scalability plots and populate tex
3. **Re-run k-sweep plot** — `plot_k_sweep.py` with `knlf_k_sweep_20161109_v2.csv` to add Composite lines; insert figure in tex
4. **Refresh signal table** — update `tab:signal_comparison` from `signal_comparison_20161109_v3.csv`
5. **Re-run fairness weight plot** — `plot_fairness_weight.py` with `fairness_weight_sweep_20161109_v2.csv`; insert figure in tex
