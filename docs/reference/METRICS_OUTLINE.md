# Detailed Metrics Collection Outline

This document provides a comprehensive overview of all metrics collected throughout the spatial crowdsourcing simulation, organised by collection layer.

**Last Updated**: June 2026  
**Version**: 3.0

**Architecture**: All metrics are coordinated by `MetricsManager` (`metrics/manager.py`). The design follows an **O(1) event-driven pattern** — metrics are accumulated in lightweight counters at each event, then summarised once per simulation step in `snapshot_step()`. Heavy historical tracking is locked behind `enable_diagnostics=True`.

---

## Table of Contents
1. [MetricsManager — Central Hub](#1-metricsmanager--central-hub)
2. [Step-Level Snapshot (`current_step_stats`)](#2-step-level-snapshot-current_step_stats)
3. [Global Accumulators (persist across steps)](#3-global-accumulators-persist-across-steps)
4. [Fairness Metrics (`fairness.py`)](#4-fairness-metrics-fairnesspy)
5. [Revenue & Earnings Tracking](#5-revenue--earnings-tracking)
6. [Worker Acceptance Tracking (`state.py`)](#6-worker-acceptance-tracking-statepy)
7. [RL Interface](#7-rl-interface)
8. [Deferral Tracking (`deferral_tracker.py`)](#8-deferral-tracking-deferral_trackerpy)
9. [Diagnostic Tracking (`tracker.py`)](#9-diagnostic-tracking-trackerpy)
10. [Final Results (`get_final_results()`)](#10-final-results-get_final_results)
11. [Metric Collection Points](#11-metric-collection-points)
12. [Research Questions Addressed](#12-research-questions-addressed)

---

## 1. MetricsManager — Central Hub

**File**: `metrics/manager.py`

`MetricsManager` is instantiated once per simulation run. It owns all sub-trackers and exposes a clean event-driven API to the simulator. All downstream consumers (RL gym, baseline scripts) call only `MetricsManager` methods.

### Configuration flags (from `sim_config`)

| Flag | Default | Effect |
|------|---------|--------|
| `enable_deferral_tracking` | `False` | Instantiates `DeferralTracker`; adds deferral event detail |
| `enable_diagnostics` | `False` | Activates heavy per-tick historical records in `MetricTracker` |

---

## 2. Step-Level Snapshot (`current_step_stats`)

Computed once per step by `snapshot_step()` and cached as `self.current_step_stats`. Used directly by the RL reward and observation functions.

### Task-count fairness

| Key | Description |
|-----|-------------|
| `jfi` | Jain's Fairness Index on completed-task counts across active workers |
| `gini` | Gini coefficient on completed-task counts |
| `utility_diff` | Utility Difference (max − min task counts) |

### Earnings fairness (v3.0)

| Key | Description |
|-----|-------------|
| `jfi_earnings` | Jain's index on `worker.total_earnings` |
| `jfi_earnings_opportunity` | Jain's index on earnings/opportunity rate (Basık LAR) |
| `gini_earnings` | Gini coefficient on `worker.total_earnings` |

### Backlog & flow

| Key | Description |
|-----|-------------|
| `backlog` | Total unassigned tasks (active + deferred) |
| `deferred_ratio` | `deferred_backlog / total_tasks_released` |
| `completed_in_step` | Tasks completed in this step |
| `task_worker_ratio` | Task release rate (per min) / active workers |

### Latency

| Key | Description |
|-----|-------------|
| `avg_wait` | Mean wait time (min) for tasks completed this step; carries forward last value if no completions (prevents RL reward exploit) |
| `step_avg_assignment_delay` | Mean seconds from release to assignment for tasks completed this step |

### Worker state

| Key | Description |
|-----|-------------|
| `worker_availability_ratio` | `available_workers / total_workers` |
| `mean_worker_idle_min` | Mean idle time (minutes) across all workers |
| `cv_worker_idle` | Coefficient of variation of worker idle times |

### Deferral breakdown (per step)

| Key | Description |
|-----|-------------|
| `pct_deferrals_below_threshold` | Fraction of this step's deferrals caused by score below threshold |
| `pct_deferrals_no_candidates` | Fraction caused by no eligible workers |

### Step accumulators (reset every step)

These feed `current_step_stats` and are zeroed after `snapshot_step()`:

- `step_completed_tasks_count` — task completions this step
- `step_wait_times` — wait time list for completions this step
- `step_travel_dist` — travel distance sum this step
- `step_tasks_released` — tasks released this step
- `step_total_deferrals`, `step_deferrals_below_threshold`, `step_deferrals_no_candidates`

---

## 3. Global Accumulators (persist across steps)

These accumulate across the full simulation and feed `get_final_results()`.

| Accumulator | Description |
|-------------|-------------|
| `_completed_tasks` | Total tasks completed |
| `_total_platform_revenue` | Sum of `task.revenue` for all completed tasks ($) |
| `_total_travel_km` | Total distance (pickup + drop) across all workers |
| `_empty_km` | Total empty (pickup) travel km |
| `_passenger_km` | Total loaded (drop) travel km |
| `_total_wait_min` | Sum of all task wait times (minutes) |
| `_wait_times` | List of per-task wait times (minutes) |
| `_backlog_peak` | Maximum backlog observed at any step |
| `_assignment_delays` | List of per-task assignment delays (seconds) |
| `total_tasks_released` | Cumulative tasks released |
| `_summary_minimal['pickup_distances']` | Per-task pickup distances (km) |
| `_summary_minimal['service_times']` | Per-task estimated service durations (min) |
| `_summary_minimal['expired_tasks']` | List of `(task_id, expiration_timestamp)` tuples |

---

## 4. Fairness Metrics (`fairness.py`)

**File**: `metrics/fairness.py`

### Core functions (called in `snapshot_step()`)

| Function | Description |
|----------|-------------|
| `jains_fairness_index(task_counts)` | Jain's index on task counts — formula: `(Σxᵢ)² / (n·Σxᵢ²)` |
| `gini_coefficient(task_counts)` | Gini coefficient on task counts |
| `utility_difference(task_counts)` | Max − min task count |

### Earnings fairness functions (v3.0, called in `snapshot_step()`)

| Function | Input | Description |
|----------|-------|-------------|
| `jfi_earnings(workers)` | `worker.total_earnings` | Jain's index on absolute earnings |
| `jfi_earnings_opportunity(workers)` | earnings/opportunity rate per worker | Jain's index on Basık LAR |
| `gini_earnings(workers)` | `worker.total_earnings` | Gini on absolute earnings |
| `gini_earnings_opportunity(workers)` | earnings/opportunity rate | Gini on LAR (available but not in final results) |
| `worker_earnings_opportunity_rates(workers)` | — | Returns list of `total_earnings / opportunity_revenue` per worker |
| `worker_feasible_for_task(worker, task, now)` | — | True if worker is available and on-shift at task release time (used for opportunity crediting) |

### Other fairness functions (available, not called in hot path)

| Function | Description |
|----------|-------------|
| `fairness_loss(actual, ideal)` | Deviation from ideal fair distribution: `Σ|actual−ideal| / Σideal` |
| `calculate_input_output_ratio(worker_stats)` | IOR per worker (reachable-area fairness concept) |
| `fairness_loss_supervisor_definition(worker_stats)` | FL using IOR-based ideal share |

These are implemented but not called by `MetricsManager` in the hot path. Available for offline analysis via `FairnessMetricsTracker`.

### `FairnessMetricsTracker`

A heavier tracker owned by `MetricsManager` (`self.fairness_tracker`). Maintains per-worker task eligibility logs and EWMA fairness histories. Called via:

- `fairness_tracker.record_task_release(task, available_workers, time)` — logs eligible workers
- `fairness_tracker.record_task_assignment(task, worker, time)` — records actual assignment
- `fairness_tracker.update_worker_stats(workers)` — updates EWMA values each step
- `fairness_tracker.get_fairness_summary()` — returns final aggregate fairness dict (included in `get_final_results()`)

---

## 5. Revenue & Earnings Tracking

### Task revenue (computed at construction)

Every `Task` object has:
- `core_movement_cost_km` — Manhattan distance pickup→dropoff (α)
- `revenue` — `base_fare + per_km_rate × α` (from `PLATFORM_REVENUE` config)

### Per-worker earnings (accumulated on `Worker` objects)

| Field | Accumulated by | Description |
|-------|---------------|-------------|
| `worker.total_earnings` | `worker.record_completion(time, revenue)` called in `on_task_completed()` | Sum of `task.revenue` for completed tasks |
| `worker.opportunity_revenue` | `on_task_released()` (k-NN neighbours) and `on_task_assigned()` | Sum of `task.revenue` for tasks the worker was a feasible k-NN candidate for. De-duplicated via `task._opp_credited_workers` set. |

### Platform revenue

- `MetricsManager._total_platform_revenue` — incremented in `on_task_completed()` by `task.revenue`
- Exposed as `total_platform_revenue` in `get_final_results()`

### Opportunity crediting logic

At task release (`on_task_released()`), the k nearest workers are queried via `spatial_index`. Each feasible worker (available, on-shift) has `task.revenue` added to `opportunity_revenue` if not already credited. The same deduplication runs again at assignment (`on_task_assigned()`), covering workers who became available after release.

---

## 6. Worker Acceptance Tracking (`state.py`)

Only non-zero when `WORKER_ACCEPTANCE.enabled = True` (stochastic cascade dispatch).

| Field | Description |
|-------|-------------|
| `state.offers_made` | Incremented each time a worker is offered a task in the cascade |
| `state.offers_rejected` | Incremented when `P(accept) = exp(−d_pick) × c_willingness` roll fails |

These are read in `simulation.py::get_final_results()` and exposed as:
- `total_offers`
- `total_rejections`
- `offer_acceptance_rate` — `(total_offers − total_rejections) / total_offers`

---

## 7. RL Interface

Three methods form the RL-facing surface of `MetricsManager`:

### `get_reward_stats(current_time) → dict`

Returns the three signals used by `_calculate_reward()` in `gym_environment.py`:

| Key | Description |
|-----|-------------|
| `fairness` | `current_step_stats['jfi']` |
| `latency` | `current_step_stats['avg_wait']` (minutes) |
| `recent_expirations` | Count of tasks expired in the trailing 30 minutes (from `_summary_minimal['expired_tasks']`) |

### `get_observation_data(state, current_time) → dict`

Returns raw features for the 15-dimensional observation vector:

| Key | Description |
|-----|-------------|
| `deferred_ratio` | From `current_step_stats` |
| `worker_availability_ratio` | From `current_step_stats` |
| `total_workers` | `len(state.all_workers_map)` |
| `jfi` | From `current_step_stats` |
| `step_avg_wait` | From `current_step_stats['avg_wait']` |
| `step_avg_assignment_delay` | From `current_step_stats` |
| `backlog_peak` | `self._backlog_peak` (global max) |
| `task_arrival_rate` | `step_tasks_released / 5.0` (tasks per minute, assumes 5 min step) |
| `is_midweek`, `is_mon_fri`, `is_weekend` | Day-of-week flags |
| `time_sin`, `time_cos` | Hour-of-day encoded cyclically |
| `revenue_density` | `Σ(deferred task revenue) / available_workers` — computed live from `state.deferred_tasks` |

Note: `revenue_density` is computed here but is only included in the observation vector when `include_revenue_density=True` in `RL_REWARD` config. Current committed gym is 15-dim (excludes it).

### `get_recent_expirations(current_time, window_minutes=30) → int`

Counts tasks in `_summary_minimal['expired_tasks']` whose expiration timestamp falls within the trailing window. Uses a reverse-iterate early-exit for O(1) typical performance. Used as the starvation signal in the RL reward.

---

## 8. Deferral Tracking (`deferral_tracker.py`)

Only active when `enable_deferral_tracking=True`. `DeferralTracker` is instantiated by `MetricsManager` and populated via `on_task_deferred()` and `on_task_assigned()`.

### Deferral events (per deferral)

| Field | Description |
|-------|-------------|
| `task_id` | Task identifier |
| `timestamp` | When the deferral occurred |
| `score` | Best composite score at time of deferral |
| `reason` | `'below_threshold'` or `'no_candidates'` |

### Assignment events (per assignment)

| Field | Description |
|-------|-------------|
| `task_id` | Task identifier |
| `timestamp` | Assignment time |
| `was_deferred` | Whether previously deferred |
| `deferral_count` | Total deferral events for this task |
| `deferral_duration_sec` | Time from first deferral to assignment (if deferred) |

### Summary statistics (`deferral_tracker.get_summary()`)

| Key | Description |
|-----|-------------|
| `immediate_assignments` | Tasks assigned without deferral |
| `deferred_assignments` | Tasks assigned after ≥1 deferral |
| `total_assignments` | Total |
| `mean_deferral_duration_sec` | Average deferred wait |
| `median_deferral_duration_sec` | Median deferred wait |
| `std_deferral_duration_sec`, `p95_deferral_duration_sec`, `max_deferral_duration_sec` | Distribution |
| `mean_deferral_count`, `max_deferral_count` | Repeat-deferral statistics |
| `total_deferral_events`, `unique_deferred_tasks` | Volume |
| `pct_below_threshold`, `pct_no_candidates` | Deferral reason breakdown |

---

## 9. Diagnostic Tracking (`tracker.py`)

`MetricTracker` is owned by `MetricsManager` (`self.metric_tracker`). Called via `metric_tracker.snapshot(state, time)` at the end of every `snapshot_step()`.

### Normal mode (`enable_diagnostics=False`, default)

- Only `_wait_time_samples`, `_travel_distance_samples`, `_completion_time_samples` lists are populated
- No per-tick record is saved (saves RAM)
- `export_to_dataframe()` returns an empty DataFrame
- `get_temporal_summary()` returns aggregate averages over the full run

### Diagnostics mode (`enable_diagnostics=True`)

Additionally populates:

- `_records` — list of per-tick dicts with: `time`, `jfi`, `ud`, `backlog`, `avg_wait_sec`, `ewma_fairness_mean`, `active_workers`, `unassigned_ratio`
- `_worker_fairness_history` — per-worker list of `{time, ewma, completed}` snapshots

Export methods:
- `export_to_dataframe()` → `pd.DataFrame` of per-tick records
- `export_worker_fairness_history()` → dict of per-worker DataFrames
- `save_all_metrics(output_dir)` → writes CSVs + JSON summary to disk

---

## 10. Final Results (`get_final_results()`)

Called once at simulation end by `simulation.py`. Returns a flat dict merging all accumulators.

### Key fields

| Key | Source |
|-----|--------|
| `completed_tasks` | `_completed_tasks` |
| `wait_times` | `_wait_times` (full list) |
| `assignment_delays` | `_assignment_delays` (full list) |
| `service_times`, `pickup_distances` | `_summary_minimal` |
| `expired_tasks` | `_summary_minimal` (list of `(id, timestamp)`) |
| `total_travel_km`, `empty_km`, `passenger_km` | global accumulators |
| `backlog_peak` | `_backlog_peak` |
| `total_platform_revenue` | `_total_platform_revenue` ($) |
| `final_jains_fairness_index` | `current_step_stats['jfi']` |
| `final_gini_coefficient` | `current_step_stats['gini']` |
| `final_jfi_earnings` | `current_step_stats['jfi_earnings']` |
| `final_jfi_earnings_opportunity` | `current_step_stats['jfi_earnings_opportunity']` |
| `final_gini_earnings` | `current_step_stats['gini_earnings']` |
| `final_utility_difference_tasks` | `current_step_stats['utility_diff']` |
| `deferral_stats` | `deferral_tracker.get_summary()` (if enabled) |
| `metric_tracker` | `MetricTracker` object reference |
| `+ fairness_summary` | All keys from `fairness_tracker.get_fairness_summary()` |

`simulation.py` further enriches this with distribution statistics (std, p90, etc.) and worker acceptance counts (`total_offers`, `total_rejections`, `offer_acceptance_rate`).

---

## 11. Metric Collection Points

| Event | Method | What is updated |
|-------|--------|-----------------|
| Task released | `on_task_released()` | `total_tasks_released`, `step_tasks_released`; `fairness_tracker.record_task_release()`; opportunity revenue credited to k nearest feasible workers |
| Task assigned | `on_task_assigned()` | `_assignment_delays`; opportunity revenue credit (deduplication); `deferral_tracker.record_assignment()` if enabled |
| Task completed | `on_task_completed()` | `_completed_tasks`, `_total_platform_revenue`, travel km, `_wait_times`, step accumulators; `worker.record_completion()`; `fairness_tracker.record_task_assignment()` |
| Task deferred | `on_task_deferred()` | `step_total_deferrals`, reason counters; `deferral_tracker.record_deferral()` if enabled |
| Task expired | `on_task_expired()` | `_summary_minimal['expired_tasks']` |
| End of step | `snapshot_step()` | Idle time update for all available workers; `current_step_stats` recomputed; `metric_tracker.snapshot()`; step accumulators reset |

---

## 12. Research Questions Addressed

### Fairness
- **Task-count fairness**: `jfi`, `gini`, `utility_diff` (step-level and final)
- **Earnings fairness** (Basık et al.): `jfi_earnings`, `jfi_earnings_opportunity`, `gini_earnings`
- **Worker experience equity**: `cv_worker_idle`, `mean_worker_idle_min`

### Efficiency
- **Latency**: `avg_wait`, `_wait_times` distribution (mean, std, p90, p95, max)
- **Travel**: `total_travel_km`, `empty_km_ratio`, pickup distance distribution
- **Throughput**: `completed_tasks`, `task_assignment_ratio`

### Starvation prevention
- **Deferral volume**: `step_total_deferrals`, `pct_deferrals_below_threshold/no_candidates`
- **Detailed deferral analysis**: `deferral_stats` (if `enable_deferral_tracking=True`)
- **Expiration-based starvation**: `get_recent_expirations()` (30-min trailing window, used in RL reward)

### Platform economics
- **Revenue**: `total_platform_revenue`
- **Earnings equity**: earnings JFI/Gini
- **Acceptance behaviour**: `total_offers`, `total_rejections`, `offer_acceptance_rate`

### RL training signal
- **Reward components**: `get_reward_stats()` → fairness (ΔJFI), latency (avg wait), starvation (recent expirations)
- **Observation vector**: `get_observation_data()` → 15 features covering backlog, fairness, latency, worker state, time context

---

*For metric definitions and interpretation guidance, see `docs/reference/DATA_DICTIONARY.md`. For RL reward formula details, see `rl/gym_environment.py::_calculate_reward()`.*
