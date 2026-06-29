# Assignment Strategies

This document describes each built-in assignment strategy in the spatial crowdsourcing simulator. Strategies plug into the event-driven `EventSimulator` via `NEW_TASK` and `FREE_WORKER` handlers (and optionally `REVIEW_BATCH`), deciding how arriving tasks and idle workers are matched under shared feasibility and travel-time rules.

Strategies are grouped below by role in experiments: **baselines** establish reference behavior, **composite and ablations** isolate the research contribution, **batch and deferral policies** accumulate market thickness before dispatch, and **literature online policies** reproduce published algorithms. Within each group, entries progress from simplest to most elaborate.

| Group | Registry name | Matching style |
|---|---|---|
| Baselines | `greedy`, `random_assign`, `biranking` | Immediate |
| Composite & ablations | `composite`, `laf`, `ewma_only` | Immediate (composite defers below threshold) |
| Batch & deferral | `cost_balancing`, `mmd_batch`, `discrete_review_lp` | Deferred / periodic |
| Literature online | `fatp_ann`, `onrta_op`, `onrta_rt`, `tsgf` | Mixed |

---

## Baselines

Simple reference policies used to anchor efficiency, fairness, and randomness comparisons. Start here when interpreting experiment tables.

### Greedy (`greedy`)

The greedy strategy is the pure efficiency baseline: whenever a task arrives or a worker becomes free, it selects the nearest feasible counterpart by Manhattan pickup distance among all currently available workers or active tasks. Feasibility requires that the worker can reach the pickup before the task expires and complete the dropoff before the worker's shift deadline, assuming 30 km/h travel speed. On task release, it scans all available workers; on worker release, it scans all unassigned active tasks. Optionally, worker acceptance can be enabled so the strategy cascades through distance-ranked candidates until one accepts the offer. If no feasible match exists, the task is deferred into the active pool for later matching.

### Random Assign (`random_assign`)

Random assignment is a spatially constrained null baseline for ablation studies. On task arrival, it collects the *k* nearest available workers (default 15), filters to feasible ones, and randomly selects one with uniform probability—no optimization beyond spatial locality. On worker release, it applies the same logic over the *k* nearest active tasks. Unlike greedy, distance affects only candidate pool membership, not the final choice, making it useful for testing whether observed fairness or efficiency gains come from deliberate scoring rather than spatial filtering alone.

### Bipartite Ranking (`biranking`)

Bipartite ranking (BRK) assigns each worker and task a permanent uniform random rank in `[0, 1)` at first appearance, using a seeded RNG for reproducibility. On arrival of either side, it scans feasible counterparts (available workers or deferred tasks) and matches to the one with the *lowest* rank—not the nearest spatially. This KVV-style random-priority policy spreads assignments across the map rather than clustering on distance, providing a two-sided online baseline with incomplete lists. Unmatched tasks are deferred. The simulator injects a persistent `rank_tracker` dictionary at reset.

---

## Composite and Ablations

The project's primary strategy and controlled variants that isolate individual design choices. Composite is the target policy; LAF and EWMA-Only remove components to test what drives fairness gains.

### Composite (`composite`)

Composite is the project's primary fairness-aware strategy. For each new task, it considers the *k* nearest available workers (default 15) via a spatial index and scores each feasible pair with a weighted sum of three components: an EWMA fairness signal based on worker idle time (`(1−γ)·T_idle + γ·previous_EWMA`), a starvation term `log(1 + task_age)`, and a spatial utility `1/(1 + pickup_distance)`. The highest-scoring worker is assigned only if the composite score meets a configurable soft threshold; otherwise the task is deferred. When a worker becomes free, matching is restricted to the deferred-task pool (not freshly released active tasks): nearby deferred tasks are ranked by starvation and utility, with the worker's fairness EWMA added as a fixed contribution, again subject to the soft threshold. Optional score normalization (min–max across candidates) and worker-acceptance cascading are supported. This design balances equitable worker service, task wait-time protection, and spatial efficiency in a single online heuristic.

### LAF (`laf`)

Least Allocated Worker First (LAF) is a simple fairness-only baseline. When a task arrives, it considers all available workers, filters to those who can feasibly serve the task, and assigns to the worker with the fewest completed tasks, breaking ties by nearest pickup distance. It does not optimize spatially beyond tie-breaking. When a worker becomes free, it falls back to greedy nearest-task matching among active tasks—a deliberate asymmetry so fairness is enforced on the task-initiated side while worker-initiated matching stays efficient.

### EWMA-Only (`ewma_only`)

EWMA-Only uses the same exponentially weighted idle-time fairness signal as composite, but without utility or starvation components. On task arrival, among all feasible available workers, it selects the one with the highest EWMA fairness score (most under-served), using pickup distance only as a tie-breaker. Like LAF, worker-side matching is greedy nearest-task among active tasks. This baseline isolates whether a sophisticated fairness metric alone improves equity without explicit spatial or starvation terms.

---

## Batch and Deferral Policies

These strategies hold tasks (and sometimes workers) before dispatching. They trade responsiveness for thicker markets or globally optimized assignments. Ordered by how dispatch is triggered: adaptive condition, every event, fixed clock.

### Cost Balancing (`cost_balancing`)

Cost balancing (CB) is an online batching policy inspired by cost-balancing delivery literature. All newly arriving tasks are immediately deferred rather than matched. Dispatch occurs only when a balance condition holds: the average nearest-neighbor pickup distance *M* across deferred tasks is at most `α × W`, where *W* is the total accumulated wait time (seconds) of deferred tasks. When triggered, it runs a greedy longest-wait-first batch match: deferred tasks are processed in release-time order, each assigned to the nearest feasible worker among its *k* nearest candidates. Worker-free events re-evaluate the same condition. This contrasts with immediate heuristics (greedy, composite) and fixed-window batching (MMD batch, discrete review) by tying dispatch timing to market thickness versus waiting cost.

### MMD Batch (`mmd_batch`)

MMD batch approximates min-max delay matching (MMD-SC style) using periodic global re-optimization. On every task release or worker-free event, it builds a cost matrix over all available workers and all pending tasks (both active and deferred), where each entry is total end-to-end delay from task release through pickup and dropoff travel, or infinite cost if infeasible. It raises feasible delays to a power (default 3) and runs the Hungarian algorithm (SciPy `linear_sum_assignment`) to minimize the sum of powered delays—a standard bottleneck heuristic. All matched pairs are committed in one batch. Unlike composite, it does not defer tasks; infeasible pairs simply receive infinite cost and remain unmatched.

### Discrete Review LP (`discrete_review_lp`)

Discrete review LP (Aveklouris et al.) defers all task arrivals and free workers until fixed review epochs (default every 60 seconds). At each `REVIEW_BATCH` event, it solves a bipartite assignment maximizing total spatial utility `1/(1 + pickup_distance)` over all available workers and all pending tasks (deferred plus active), using the Hungarian algorithm on negated utilities. Infeasible pairs receive a large negative utility. Arrivals and worker releases only enqueue deferrals and schedule the next review; no immediate matching occurs. This models impatient-agent reneging under periodic batch clearing rather than continuous online assignment.

---

## Literature Online Policies

Published algorithms from spatial crowdsourcing and delivery fairness literature. These mix immediate assignment, deferral, and global optimization in strategy-specific ways.

### FATP-ANN (`fatp_ann`)

FATP-ANN implements the fairness-aware task planning baseline from the crowdsourced delivery literature. It maintains a dynamic fairness cap *ĉ* = Σ(countᵢ²) / Σ(countᵢ) over worker completion counts; workers at or above the cap are ineligible for new assignments. On task arrival (Task-Process), it assigns each new task to the nearest eligible available worker who can feasibly complete it. On worker release (Worker-Process), it greedily bundles multiple tasks while the worker remains under the cap: using a shadow location/time state, it repeatedly picks the unassigned task with highest exponentially decayed utility `α·exp(−μ·wait_hours)` until no valid tasks remain or the cap binds. The simulator injects a `FairnessCapTracker` at reset to maintain *ĉ* in O(1) per assignment. An optional *k*-nearest worker filter can limit the candidate pool for performance.

### ONRTA-OP (`onrta_op`)

ONRTA-OP (Online Non-rejection Task Assignment — Optimal) is a two-stage policy from the non-rejection spatial crowdsourcing literature. It counts total arrivals (tasks and workers); while arrivals remain below half the expected market size `(expected_a + expected_b) / 2`, it uses greedy immediate matching (highest utility `1/(1 + d_pick)` among feasible pairs). After crossing that threshold (Stage 2), each arrival triggers a global utility-maximizing bipartite match over the full pending pool; the new entity is assigned only if it appears in the optimal matching, otherwise the policy falls back to greedy. Unmatched tasks are deferred. The simulator injects an `onrta_tracker` and defaults `expected_a`/`expected_b` to the dataset's task and worker counts.

### ONRTA-RT (`onrta_rt`)

ONRTA-RT (Randomized Threshold) draws a single random utility threshold θ once per episode from `{e⁰, e¹, …, e^(λ−1)}` where `λ = ceil(ln(U_max + 1))` and utilities are scaled by 100. On each task or worker arrival, it evaluates all feasible counterparts (available workers for tasks, deferred tasks for workers), computes scaled utility `100/(1 + d_pick)`, and if any candidate meets or exceeds θ, randomly selects among those; otherwise it falls back to the highest-utility feasible match. Unmatched tasks are deferred. The threshold creates a randomized quality floor that encourages waiting for better matches early in the episode while guaranteeing a fallback assignment.

### TSGF (`tsgf`)

Two-Sided Group Fairness (TSGF) implements a runtime sampling adaptation of the TSGF-KIID framework. All arriving tasks are deferred; on each dispatch opportunity (task arrival or worker free), a deterministic random roll selects one of four actions: with probability `α`, run pure greedy (minimize pickup distance across the deferred pool); with probability `β`, serve the most idle worker (max-min worker fairness); with probability `γ`, serve the longest-waiting deferred task (max-min task fairness); or with remaining probability, explicitly defer again to accumulate market thickness. Each sub-heuristic finds the best feasible *k*-nearest match for its chosen entity. This randomized policy explores trade-offs between operator profit and two-sided fairness without scalarizing them into a single score.
