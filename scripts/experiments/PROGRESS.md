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
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_strategy_comparison.py`  *(DiDi Chengdu)*
Script: `scripts/experiments/s52_main_results/run_strategy_comparison.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster | `results/s52_main_results/` (TBD — not yet finished) | Still running |
| Laptop | `results/s52_main_results/` (TBD — not yet finished) | Still running |

- **Stage 1 — Experiments**: 🟡 In progress (laptop + cluster)
  - Laptop: Greedy, k-NLF, Composite, EWMA-Only, k-NTF-EPH, k-NTF-IR, Random, LAF, BiRanking done; FATP-ANN, ONRTA-RT, ONRTA-OP, Disc. Review LP still running
  - Cluster: running in parallel
- **Stage 2 — Plotting / Tabling**: 🟡 Partially complete
  - `tab:didi_results` exists in `experimental-section.tex`
  - Greedy, k-NLF, Composite rows have TAR / Wait / JFI (tasks) filled (new run)
  - `JFI (earn.)`, `Rev. (k$)` still `---` for all rows — need completed CSV
  - LAF, BiRanking, FATP-ANN, ONRTA-RT, ONRTA-OP, Disc. Review LP rows still `---` or `$^*$` (pre-bugfix)
- **Stage 3 — Analysis & Writing**: ❌ Not started

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
| Cluster ✅ | `results/s54_ablation/signal_comparison_20161109_cluster.csv` | Retrieved — complete |

- **Stage 1 — Experiments**: ✅ Complete on cluster (k-NLF + NTF variants)
  - ⚠️ **Composite not included + k=5 labels were wrong** — `run_signal_comparison.py` updated to fix both; needs re-run on cluster
  - Re-run command: `python3 -u scripts/experiments/s54_ablation/run_signal_comparison.py 2>&1 | tee results/s54_ablation/log_signal.log`
  - Output will overwrite `signal_comparison_20161109.csv` (re-SCP with `_cluster_v2` suffix)
- **Stage 2 — Plotting / Tabling**: ✅ Table in tex (`tab:signal_comparison`, 2026-07-01)
  - Columns: JFI (tasks), JFI (earn.), CV (idle), CV (earn.), Wait, Time — k=15 rows only
  - Composite row currently `---`; will be filled after re-run
  - Key result: k-NLF bold in every column, incl. EPH's own metric and IR's own metric
- **Stage 3 — Analysis & Writing**: ❌ Not started

---

### `run_fairness_weight_sweep.py`  *(§5.4.3 — Weighted Scorer Sensitivity)*
Script: `scripts/experiments/s54_ablation/run_fairness_weight_sweep.py`

| Copy | Path | Notes |
|---|---|---|
| Cluster ✅ | `results/s54_ablation/fairness_weight_sweep_20161109_cluster.csv` | Retrieved — complete, λ_f ∈ {0.0…3.0} |

- **Stage 1 — Experiments**: ✅ Complete on cluster
  - Swept λ_f ∈ {0.0, 0.2, …, 2.0, 2.5, 3.0}; paper default = 1.6
- **Stage 2 — Plotting / Tabling**: ✅ Figure complete (2026-07-01)
  - `results/figures/fairness_weight_sweep.pdf` — 2-panel (Worker Fairness / Task Tail Latency), 300 DPI, Times New Roman
  - Script: `scripts/plots/plot_fairness_weight.py`
  - Key findings: ΔJFI=0.024 (robust); JFI rate peaks at λ_f=1.6; P95 wait drops 14.0→12.8m; TAR flat (reported in prose only)
  - **JFI (tasks) trend for discussion** (not plotted — superseded by JFI rate as hero metric):
    - λ_f=0.0: JFI=0.566 — starts *below* Greedy (0.593), because k=15 spatial restriction limits reassignment options
    - λ_f=0.2–0.4: JFI rises sharply to ~0.587, crossing the Greedy baseline
    - λ_f=0.4–2.0: JFI plateaus in 0.585–0.590 range — robust, but never reaches k-NLF (0.645)
    - λ_f=2.5–3.0: slight decline back to ~0.587
    - Range across full sweep: Δ=0.024 (min=0.566 at λ_f=0, max=0.590 at λ_f=1.2)
    - Interpretation: raw task-count fairness is bounded by the k=15 spatial pool; the EWMA signal improves participation equity (JFI rate) more than raw task distribution, confirming JFI rate is the more appropriate metric for a time-aware system
  - Greedy and k-NLF reference lines on both panels (grey dashed / indigo dash-dot); shared bottom legend; guide label at ¼ height
  - `\subsubsection{Weighted Scorer Sensitivity}` stub in tex still needs `\includegraphics` + caption + prose
- **Stage 3 — Analysis & Writing**: ❌ Not started
- **Pending re-run** (low priority, add if time permits):
  - `run_fairness_weight_sweep.py` updated to output `CV (idle)` — coefficient of variation of worker idle time
  - This directly validates the core claim: Composite (starvation=0) = EWMA inter-task wait time fairness → as λ_f ↑, idle time should become more uniform (CV(idle) ↓)
  - Without CV(idle) the claim is supported by proxy (JFI rate ↑, P95 ↓) but not directly proven
  - Re-run command: `python3 -u scripts/experiments/s54_ablation/run_fairness_weight_sweep.py 2>&1 | tee results/s54_ablation/log_fw_sweep_v2.log`
  - Re-SCP with suffix `_cluster_v2` when done

---

## Quick Status Summary

| Script | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| §5.2 Gowalla comparison | ✅ | ✅ | ❌ |
| §5.2 DiDi strategy comparison | 🟡 | 🟡 | ❌ |
| §5.3 Scalability fleet | 🔄 | ❌ | ❌ |
| §5.3 Scalability tasks | 🔄 | ❌ | ❌ |
| §5.4 k-NLF k-sweep | ✅ (k-NLF only) | 🟡 | ❌ |
| §5.4 Signal comparison | ✅ | ✅* | ❌ |
| §5.4 Fairness weight sweep | ✅ | ✅* | ❌ |

**Legend:** ✅ Complete · 🟡 Partial / in progress · ⚠️ Retrieved but incomplete · ❌ Not started

---

## Immediate Next Actions (priority order)

1. **Re-run scalability scripts** with higher timeout (`--timeout 1800`) or drop FATP-ANN / LAF from scalability — both CSVs cut off early
2. **Wait for DiDi strategy comparison to finish** — then fill in `tab:didi_results` completely
3. **Fill Gowalla table gaps now** — cluster CSV is in hand; copy JFI (earn.), Rev., and Disc. Review LP row into tex
4. **Start Stage 2 for §5.4** — all three ablation CSVs are in hand; fairness weight sweep and k-sweep are the quickest wins
5. **Stage 2 for §5.3** — once scalability re-runs complete
