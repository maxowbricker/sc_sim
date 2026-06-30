# Future Experiment Improvements

Low-priority additions identified during paper sprint (Jun 2026).
These are deferred due to time constraints but are worth implementing post-submission.

---

## 1. Surface deferral stats in main comparison output

**What:** Add `Rescue Rate`, `Direct Assign %`, and `Avg Deferral Wait (s)` to the
`run_strategy_comparison.py` CSV output.

**Why:** The `DeferralTracker` in `metrics/deferral_tracker.py` already computes
(at O(1) cost per event):
- `unique_tasks_deferred` — tasks that found no available worker on arrival
- `tasks_assigned_after_deferral` — rescued from backlog by a FREE_WORKER event
- `rescue_rate` — fraction of deferred tasks eventually served

Direct assignments (NEW_TASK path) = `completed - tasks_assigned_after_deferral`.

This directly answers "where does the fairness benefit come from?" — fairness
logic applied at FREE_WORKER time (backlog pickup) vs NEW_TASK time (immediate dispatch).

**Blocker:** `enable_deferral_tracking` is nested under `strategy_params` in the config
but `MetricsManager.__init__` reads it from the top-level config dict:
```python
# simulation.py (line 64 and 182):
self.metrics = MetricsManager({'strategy_params': self.strategy_params})

# MetricsManager.__init__ (line 36):
self.enable_deferral_tracking = config.get('enable_deferral_tracking', False)
# BUG: key is nested under 'strategy_params', not at config top level → always False
```

**Fix required (2 files):**
1. `simulator/simulation.py` — flatten the flag when constructing MetricsManager:
```python
self.metrics = MetricsManager({
    'strategy_params': self.strategy_params,
    'enable_deferral_tracking': self.strategy_params.get('enable_deferral_tracking', False),
    'enable_diagnostics': self.strategy_params.get('enable_diagnostics', False),
})
```
2. `scripts/experiments/s52_main_results/run_strategy_comparison.py` — inject flag for all strategies and extract from `stats.get('deferral_stats', {})` in `extract_metrics()`.

---

## 3. JFI stability at scale in scalability scripts

**What:** Add JFI (and optionally wait time) as output columns in
`s53_scalability/run_scalability_fleet.py` and `run_scalability_tasks.py`.

**Why:** Currently both scripts only record wall-clock runtime vs fleet/task size.
A stronger scalability claim is: "not only does runtime stay flat as |W| grows,
but JFI also remains stable." k-NLF is scale-invariant by construction (local k-NN
scope is independent of |W|), so showing JFI holds at 1k/5k/10k/20k/40k workers
is direct empirical evidence.

**Effort:** Both scripts already call `get_final_results()` — just add JFI extraction
and include in the CSV FIELDNAMES.

---

## 4. k=1 as explicit k-NLF anchor in k-sweep

**What:** Add `k=1` to `K_VALUES` in `s54_ablation/run_knlf_k_sweep.py`.

**Why:** The current sweep uses Greedy as the k=1 proxy, but doesn't explicitly run
`knlf` at `k=1`. Running k-NLF with k=1 should produce results indistinguishable
from Greedy (it picks the nearest worker regardless of task count since there is
only one candidate). This proves the graceful degradation claim directly rather
than by proxy, and only costs one extra ~40s simulation run.

**Effort:** One line change: `K_VALUES = [1, 3, 5, 10, 15]`

---

## 5. Multi-day generalization check

**What:** Run `run_strategy_comparison.py` on a second Didi day (e.g., `20161128`)
and compare the JFI and wait-time rankings.

**Why:** All experiments use Didi day `20161109`. Running on one more day tests
whether the strategy rankings are stable across different demand patterns
(day-of-week effects, density variation). This would allow the paper to say
"results hold across demand conditions" rather than relying on a single day.

**Effort:** `caffeinate python scripts/experiments/s52_main_results/run_strategy_comparison.py --day 20161128 --output results/s1_overall_performance/didi_20161128.csv`

~4.5 hours on laptop, ~4.5 hours on cluster. Requires a second Didi day folder
to exist under `data/didi/full_didi_gaia/`.
