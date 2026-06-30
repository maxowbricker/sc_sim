# Section 4/5.3 — Robustness & Computational Efficiency

Two sets of experiments:
1. **Robustness** (§5.4): strategy rankings hold across varying supply/demand ratios
2. **Scalability** (§5.3): O(k log W) strategies stay flat as fleet/task volume grows

---

## Scripts in this directory

| Script | Purpose | Est. runtime |
|--------|---------|-------------|
| `run_didi_ratio_sweep.py` | Didi at 1:4 / 1:5 / 1:7 ratios × 4 strategies | ~20 min |
| `run_scalability_fleet.py` | **§5.3 Exp A**: Vary \|W\| (10k→36.8k), fix \|T\| at 224k | ~3–4 hours |
| `run_scalability_tasks.py` | **§5.3 Exp B**: Vary \|T\| (50k→224k), fix \|W\| at 10k | ~1.5–2 hours |

Gowalla ratio sweep uses **`scripts/run_gowalla_comparison.py`** with its built-in
`--ratio` flag (see run commands below).

---

## §5.3 Scalability experiments

### Experiment A — Fleet Scaling (primary figure)
**Claim being proved:** k-NLF and Composite runtime is independent of fleet size (O(k log W));
Greedy and LAF grow linearly (O(W)).

```bash
caffeinate python scripts/experiments/s4_robustness/run_scalability_fleet.py
# Output: results/s4_robustness/scalability_fleet.csv
```

**Design:**
- Tasks: full 224,219 (no subsampling)
- Workers: [10,000 · 15,000 · 20,000 · 25,000 · 30,000 · 36,799]
- Stratified temporal sampling (288 bins, seed=42) — preserves intra-day worker density
- Strategies: k-NLF, Composite, Greedy, LAF, FATP-ANN
- Disc. Review LP and ONRTA-OP **excluded** (O(W×T); mention in paper text)

**Expected result:** k-NLF / Composite → nearly flat lines; Greedy / LAF → linear slope.
Plot on log-log axes to clearly show the O(1) vs O(W) curve shapes.

### Experiment B — Task Volume Scaling (supplementary)
**Claim being proved:** Both O(k log W) and O(W) strategies scale linearly with |T|, but
O(k log W) strategies have a significantly shallower slope (lower per-event constant factor).

```bash
caffeinate python scripts/experiments/s4_robustness/run_scalability_tasks.py
# Output: results/s4_robustness/scalability_tasks.csv
```

**Design:**
- Workers: fixed 10,000 (stratified sample)
- Tasks: [50,000 · 100,000 · 150,000 · 200,000 · 224,219] (stratified, 288 bins)
- Same 5 strategies as Experiment A

**Expected result:** Parallel linear curves; k-NLF / Composite slopes much shallower than
Greedy / LAF. FATP-ANN has O(k log W) asymptotics but a high constant factor (visible here).

---

## §5.4 Robustness experiments (ratio sweep)

```bash
# Didi ratio sweep — ~20 min
caffeinate python scripts/experiments/s4_robustness/run_didi_ratio_sweep.py

# Gowalla ratio sweep: 1:7 (default), 1:5, 1:4
python scripts/run_gowalla_comparison.py --compression compressed --ratio 0.143 \
    --output results/s4_robustness/gowalla_ratio_0143.csv
python scripts/run_gowalla_comparison.py --compression compressed --ratio 0.2 \
    --output results/s4_robustness/gowalla_ratio_020.csv
python scripts/run_gowalla_comparison.py --compression compressed --ratio 0.25 \
    --output results/s4_robustness/gowalla_ratio_025.csv
```

---

## Output files

| File | Contents |
|------|----------|
| `results/s4_robustness/scalability_fleet.csv` | 6 fleet sizes × 5 strategies — n_workers, elapsed_s, TAR, JFI |
| `results/s4_robustness/scalability_tasks.csv` | 5 task volumes × 5 strategies — n_tasks, elapsed_s, TAR, JFI |
| `results/s4_robustness/didi_ratio_sweep.csv` | 3 ratios × 4 strategies — TAR, JFI, wait, elapsed_s |
| `results/s4_robustness/gowalla_ratio_*.csv` | Gowalla at each density — full strategy table |

---

## Complexity reference

| Strategy | Complexity per event | Notes |
|---|---|---|
| Greedy | O(W) | scans all workers for nearest |
| k-NLF | O(k log W) | k-NN query + O(k) task-count sort |
| Composite | O(k log W) | k-NN query + O(k) EWMA scoring |
| LAF | O(W) | global least-allocated scan |
| FATP-ANN | O(k log W) | k-NN query + cap tracking; high constant |
| ONRTA-RT | O(W) | threshold scan over all workers |
| ONRTA-OP | O(W×T) | Hungarian match at midpoint |
| Disc. Review LP | O(W×T) per epoch | Hungarian at each review epoch |

> **Note:** The O(k log W) complexity is split: the k-d tree query is O(k log W);
> the per-candidate scoring step is O(k). Since k=15 is fixed, both terms are
> effectively constant w.r.t. W — the key differentiator vs O(W) strategies.
