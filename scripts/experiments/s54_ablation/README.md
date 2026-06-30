# §5.4 — Ablation & Sensitivity Analysis

Three independent ablation experiments, each isolating one design dimension.
All run on Didi 20161109. All complete in under 25 minutes each.

**Results output:** `results/s54_ablation/`

---

## Run commands (can be run concurrently)

```bash
# §5.4.1 — k sweep: how does candidate pool size affect the fairness–efficiency tradeoff?
python scripts/experiments/s54_ablation/run_knlf_k_sweep.py \
    > results/s54_ablation/log_knlf_k_sweep.log 2>&1 &

# §5.4.2 — Signal comparison: which O(k) fairness signal wins?
python scripts/experiments/s54_ablation/run_signal_comparison.py \
    > results/s54_ablation/log_signal_comparison.log 2>&1 &

# §5.4.3 — Weight sweep: how sensitive is Composite to λ_f?
python scripts/experiments/s54_ablation/run_fairness_weight_sweep.py \
    > results/s54_ablation/log_fairness_weight_sweep.log 2>&1 &

wait
echo "All ablation experiments complete"
```

---

## §5.4.1 — Impact of k (Candidate Pool Radius)
**Script:** `run_knlf_k_sweep.py`
**Output:** `results/s54_ablation/knlf_k_sweep_20161109.csv`
**Est. time:** ~10 min

Sweeps k ∈ {3, 5, 10, 15} for k-NLF, with Greedy (k=1 equivalent) and LAF (k=|W|)
as end-anchors. Shows the strategy smoothly transitions from Greedy to LAF as k grows,
with JFI gain saturating around k=15.

**Plot:** k (x-axis) vs JFI and Avg Wait (dual y-axis). Highlight k=15 as paper default.

---

## §5.4.2 — Signal Isolation: Task-Counts vs Duration-Aware Signals
**Script:** `run_signal_comparison.py`
**Output:** `results/s54_ablation/signal_comparison_20161109.csv`
**Est. time:** ~12 min

Holds the spatial structure constant (k=15 nearest workers) and varies only the
fairness signal used to rank the candidate pool:

| Signal | Strategy | Duration-aware? |
|--------|----------|-----------------|
| None (nearest wins) | Greedy | — |
| Raw task count | k-NLF | No |
| Earnings per hour | k-NTF-EPH | Yes |
| Idle ratio | k-NTF-IR | Yes |
| Unconstrained count | LAF (k=W) | No |

**Narrative:** "Billy vs John" — a worker online for 6 hours is treated identically to one
who started 30 minutes ago under raw task-count signals. Duration-aware signals correct this.

**Plot:** Grouped bar chart or scatter — JFI (tasks) vs Avg Wait, one point per signal.

---

## §5.4.3 — Weighted Scorer Sensitivity (λ_f sweep)
**Script:** `run_fairness_weight_sweep.py`
**Output:** `results/s54_ablation/fairness_weight_sweep_20161109.csv`
**Est. time:** ~22 min

Sweeps `fairness_weight` (λ_f) ∈ {0.0, 0.2, 0.4, …, 2.0, 2.5, 3.0} in the Composite
Scorer, holding all other params fixed (sw=0.0, uw=1.0, γ=0.1, k=15, st=0.0).

Paper default λ_f=1.6 is highlighted in output.

**Expected finding (two possible outcomes):**
- **Flat plateau (likely on dense Didi):** JFI is stable across a wide λ_f range → the
  structural choice (k) is a stronger fairness lever than fine-tuning λ_f. Robust to misconfiguration.
- **Steep curve:** λ_f is a meaningful tuning knob → operators can dial fairness vs throughput.

**Plot:** λ_f (x-axis) vs JFI and Avg Wait (dual y-axis). Mark λ_f=1.6 with a dashed line.
