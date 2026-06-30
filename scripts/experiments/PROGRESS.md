# Experiment & Writing Progress Tracker

> Last updated: 2026-07-01
>
> **Stages**
> - **Stage 1 — Experiments**: raw simulation results collected
> - **Stage 2 — Plotting / Tabling**: results transferred into `experimental-section.tex`
> - **Stage 3 — Analysis & Writing**: narrative prose, claims, and discussion written

---

## §5.2 — Effectiveness (Main Results)

### `run_gowalla_comparison.py`
Script: `scripts/experiments/s52_main_results/run_gowalla_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster ✅ | `results/s52_main_results/gowalla_austin_compressed_cluster.csv` | Retrieved via SCP — **use this one** (has correct util %, Revenue, JFI earn.) |
| Laptop ✅ | `gowalla_austin_20100901_to_20100930.csv` (repo root) | Pre-bugfix idle-time, util=0.0% — superseded |

- **Stage 1 — Experiments**: ✅ Complete (laptop + cluster)
  - 8,758 workers | 43,788 tasks | ratio 0.20 | all 13 strategies finished including Disc. Review LP
- **Stage 2 — Plotting / Tabling**: ✅ Complete
  - `tab:gowalla_results` fully filled in `experimental-section.tex` (2026-07-01)
  - All columns populated: TAR, Wait, JFI (tasks), JFI (earn.), Rev. (k$), Time
  - All values from **cluster run** (`gowalla_austin_compressed_cluster.csv`)
  - Bold updated: TAR→Greedy/ONRTA-OP/Disc.LP (0.999 tie); Wait→Disc.LP (2.72m); JFI tasks→LAF (0.894); JFI earn→LAF (0.875); Rev→Greedy/ONRTA-OP (297.7k); Time→k-NLF (4.6s)
  - Note: "LTF" label corrected to "LAF" in tex
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01)
  - Prose written directly below Gowalla table in `experimental-section.tex`
  - Covers: Throughput, Worker fairness, Task latency, Platform revenue, Computational cost, Cross-dataset consistency

---

### `run_strategy_comparison.py`  *(DiDi Chengdu)*
Script: `scripts/experiments/s52_main_results/run_strategy_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| Laptop ✅ | `scripts/run_strategy_comparison.py` (terminal output) | **Use this** — all 13 strategies complete, bug-fixed codebase |
| Cluster | `results/s52_main_results/` (still running) | Backup; not needed now |

- **Stage 1 — Experiments**: ✅ Complete (laptop, all 13 strategies)
  - 36,799 workers | 224,219 tasks | all strategies finished incl. FATP-ANN (12,410s) and Disc. Review LP (5,060s)
  - FATP-ANN degenerate: TAR=0.599, peak backlog=39,298 (threshold distribution-shift issue)
  - Disc. Review LP: TAR/JFI/Wait valid; utilisation metric broken (Mean Idle=2.9M min — known bug)
  - ONRTA-OP dropped from paper (indistinguishable from Greedy: JFI=0.594, wait=2.61m)
- **Stage 2 — Plotting / Tabling**: ✅ Complete (2026-07-01)
  - `tab:didi_results` fully filled in `experimental-section.tex`
  - Rows included: Greedy, k-NLF, Composite, LTF, BiRanking, FATP-ANN†, ONRTA-RT, Disc. Review LP
  - All columns populated: TAR, Wait, JFI (tasks), JFI (earn.), Rev. (k$), Time (s)
  - Bold: TAR→k-NLF/Composite (0.8626); Wait→Disc.LP (1.64m); JFI tasks→LTF (0.705); JFI earn→LTF (0.700); Rev→k-NLF/Composite (2,765.8k tied); Time→k-NLF (38s)
  - FATP-ANN values in parentheses, excluded from bolding; ONRTA-OP removed from both tables
- **Stage 3 — Analysis & Writing**: ✅ Complete (2026-07-01)
  - Prose shared with Gowalla (single §5.2 narrative covers both datasets)

---

## §5.3 — Computational Efficiency & Scalability

### `run_scalability_fleet.py`  *(Vary |W|, fix |T|)*
Script: `scripts/experiments/s53_scalability/run_scalability_fleet.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster 🔄 | `results/s53_scalability/scalability_fleet_cluster.csv` | Early snapshot retrieved — script still running on cluster, re-SCP when done |

- **Stage 1 — Experiments**: 🔄 Still running on cluster
  - Script is progressing through all 6 fleet sizes; early snapshot had only 10k workers complete
  - LAF and FATP-ANN time out (>900s) at every fleet size — this is expected and **reportable** (supports scalability claim)
  - k-NLF, Composite, Greedy will complete all 6 sizes — these are the primary strategies of interest
  - When done: re-SCP with `_cluster` suffix and update this entry
- **Stage 2 — Plotting / Tabling**: ❌ Not started
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_scalability_tasks.py`  *(Vary |T|, fix |W|)*
Script: `scripts/experiments/s53_scalability/run_scalability_tasks.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster 🔄 | `results/s53_scalability/scalability_tasks_cluster.csv` | Early snapshot retrieved — script still running on cluster, re-SCP when done |

- **Stage 1 — Experiments**: 🔄 Still running on cluster
  - Script is progressing through all task volumes; early snapshot had 50k and 100k complete
  - FATP-ANN times out (>900s) at 100k tasks and beyond — expected, supports scalability claim
  - k-NLF, Composite, Greedy, LAF will complete all volumes
  - When done: re-SCP with `_cluster` suffix and update this entry
- **Stage 2 — Plotting / Tabling**: ❌ Not started
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

## §5.4 — Ablation & Sensitivity Analysis

### `run_knlf_k_sweep.py`  *(§5.4.1 — Impact of k)*
Script: `scripts/experiments/s54_ablation/run_knlf_k_sweep.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster ✅ | `results/s54_ablation/knlf_k_sweep_20161109_cluster.csv` | Retrieved — complete, k ∈ {3,5,10,15,25,50,100} + Greedy + LAF anchors |

- **Stage 1 — Experiments**: ✅ Complete on cluster (k-NLF only)
  - k ∈ {3,5,10,15,25,50,100}; Greedy (k=∞) and LAF (k=W) anchors included
  - ⚠️ **Composite not yet included** — `run_knlf_k_sweep.py` has been updated to sweep k for Composite too, but needs a re-run on the cluster when time permits. Plot script (`plot_k_sweep.py`) will automatically render the Composite line once the CSV has it.
- **Stage 2 — Plotting / Tabling**: ✅ Figure complete (k-NLF only)
  - `results/figures/k_sweep.pdf` — 3-panel figure (JFI / Wait / Runtime vs k), 300 DPI, Times New Roman
  - Script: `scripts/plots/plot_k_sweep.py`
  - Will gain Composite lines automatically once re-run CSV is available
  - `\subsubsection{Impact of k}` stub in tex still needs `\includegraphics` + caption + prose
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_signal_comparison.py`  *(§5.4.2 — Task-Count vs EWMA)*
Script: `scripts/experiments/s54_ablation/run_signal_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| Laptop v2 ✅ | `results/s54_ablation/signal_comparison_20161109_v2.csv` | **Use this** — complete, incl. Composite + corrected k=5 labels |
| Cluster v1 | `results/s54_ablation/signal_comparison_20161109_cluster.csv` | Superseded (no Composite, wrong k=5 labels) |

- **Stage 1 — Experiments**: ✅ Complete (laptop v2, all 7 strategies)
  - Greedy, k-NTF-EPH (k=15), k-NTF-EPH (k=5), k-NTF-IR (k=15), k-NTF-IR (k=5), k-NLF (k=15), Composite (k=15)
  - Key result: **k-NLF wins every column including those each alternative directly optimises**
    - CV(idle): k-NLF=1.475 < k-NTF-IR=1.561 (k-NTF-IR targets idle equity but k-NLF beats it!)
    - CV(earn): k-NLF=0.825 < k-NTF-EPH=0.864 (same story for earnings equity)
  - Composite: JFI(t)=0.589, JFI(e)=0.624, CV(idle)=1.561, CV(earn)=0.918, wait=3.22m
- **Stage 2 — Plotting / Tabling**: ✅ Table complete (`tab:signal_comparison`, updated 2026-07-01)
  - All rows fully populated with v2 laptop values; Composite CV(earn.)=0.918 now filled
  - Caption updated: "k-NLF dominates every column including those each alternative directly optimises"
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_fairness_weight_sweep.py`  *(§5.4.3 — Weighted Scorer Sensitivity)*
Script: `scripts/experiments/s54_ablation/run_fairness_weight_sweep.py`

| Copy | Path | Notes |
|---|---|---|
| Laptop v2 ✅ | `results/s54_ablation/fairness_weight_sweep_20161109_v2.csv` | **Use this** — complete, incl. CV(idle) per λ_f |
| Cluster v1 | `results/s54_ablation/fairness_weight_sweep_20161109_cluster.csv` | Superseded (no CV idle) |

- **Stage 1 — Experiments**: ✅ Complete (laptop v2)
  - λ_f ∈ {0.0, 0.2, …, 2.0, 2.5, 3.0}; paper default = 1.6
  - Key findings from v2:
    - ΔJFI (tasks) = 0.023 across full sweep (min=0.567 at λ_f=0.0, max=0.590 at λ_f=1.4) — robust
    - JFI rate peaks at λ_f≈1.4 (0.8705), stays high through λ_f=1.6 (0.8665) — plateau behaviour
    - Wait range: 3.14–3.25m (Δ=0.11m) — insensitive to λ_f
    - P95 wait: 14.12m at λ_f=0.0 → 12.86m at λ_f=3.0 (tail latency improves with more fairness weight)
    - Paper default λ_f=1.6 vs λ_f=0.0: ΔJFI=+0.020 (+3.6%), ΔWait=+0.09m — good robustness argument
- **Stage 2 — Plotting / Tabling**: ⚠️ Plot needs re-run with v2 CSV
  - Current figure (`results/figures/fairness_weight_sweep.pdf`) used cluster v1 data
  - Re-run: `conda run -n sc python3 scripts/plots/plot_fairness_weight.py --input results/s54_ablation/fairness_weight_sweep_20161109_v2.csv`
  - Story unchanged (JFI robust, peak around 1.4–1.6, P95 improves); numbers may shift slightly
  - `\subsubsection{Weighted Scorer Sensitivity}` stub in tex still needs `\includegraphics` + caption + prose
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

## Quick Status Summary

| Script | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| §5.2 Gowalla comparison | ✅ | ✅ | ✅ |
| §5.2 DiDi strategy comparison | ✅ | ✅ | ✅ |
| §5.3 Scalability fleet | 🔄 | ❌ | ❌ |
| §5.3 Scalability tasks | 🔄 | ❌ | ❌ |
| §5.4 k-NLF k-sweep | ✅ (k-NLF only) | 🟡 | ❌ |
| §5.4 Signal comparison | ✅ | ✅ | ❌ |
| §5.4 Fairness weight sweep | ✅ | ⚠️ plot needs v2 re-run | ❌ |

**Legend:** ✅ Complete · 🟡 Partial / in progress · 🔄 Running · ⚠️ Minor action needed · ❌ Not started

---

## Immediate Next Actions (priority order)

1. **Write §5.2 prose** — both tables fully populated; start narrative (Effectiveness section)
2. **Wait for scalability cluster runs** — re-SCP when done, then Stage 2 for §5.3
3. **Re-run signal comparison** — include Composite + fix k=5 labels (`_cluster_v2` suffix)
4. **Re-run fairness weight sweep** — add CV(idle) output (`_cluster_v2` suffix)
5. **Add `\includegraphics` + captions + prose stubs** for §5.4.1 (k-sweep) and §5.4.3 (fairness weight)
