# Spatial Crowdsourcing Simulation: Data Dictionary

## Overview
Complete reference for all metrics, flags, and data structures collected by the simulation.

**Last Updated**: October 21, 2025  
**Version**: 2.0 (Post-statistics enhancement)

---

## Table of Contents
1. [Simulation Configuration Flags](#simulation-configuration-flags)
2. [Core Metrics](#core-metrics)
3. [Task Wait Time Metrics](#task-wait-time-metrics)
4. [Worker Metrics](#worker-metrics)
5. [Fairness Metrics](#fairness-metrics)
6. [System Performance Metrics](#system-performance-metrics)
7. [Data Formats](#data-formats)
8. [Diagnostic Mode Data](#diagnostic-mode-data)
9. [Computational Cost Summary](#computational-cost-summary)
10. [Version History](#version-history)

---

## 1. Simulation Configuration Flags

### `enable_diagnostics` (boolean)
- **Default**: `false`
- **Impact**: Enables detailed component-level tracking
- **Performance Cost**: ~40% slower (enables "slow path")
- **Additional Data Collected**:
  - Score component dominance per assignment
  - Per-assignment breakdown (fairness/starvation/utility)
  - Assignment decision history
- **When to Enable**: Only for mechanism analysis (Exp 008-style studies)

### `normalize_scores` (boolean)
- **Default**: `false` (changed to `true` post-Exp 008)
- **Impact**: Applies min-max normalization to composite score components
- **Purpose**: Prevents component dominance (resolved worker idle time paradox)
- **Performance Cost**: Negligible (<0.1%)

### `gamma` (float, 0.0-1.0)
- **Default**: `0.5`
- **Description**: EWMA decay parameter for fairness tracking
- **Impact**: Higher = more weight on recent history

### `soft_threshold` (float, 0.0-1.0)
- **Default**: `0.5`
- **Description**: Minimum score required for task assignment
- **Impact**: Defers low-quality assignments

### `fairness_weight` (λ₁, float)
- **Default**: varies by experiment
- **Description**: Weight for EWMA fairness component in composite score
- **Range**: Typically 0.0-5.0

### `starvation_weight` (λ₂, float)
- **Default**: varies by experiment
- **Description**: Weight for starvation prevention component
- **Range**: Typically 0.0-2.0

### `utility_weight` (λ₃, float)
- **Default**: varies by experiment
- **Description**: Weight for distance/utility component
- **Range**: Typically 0.1-2.0

---

## 2. Core Metrics

### Simulation Completion

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `completed_tasks` | int | count | Number of tasks successfully completed |
| `task_assignment_ratio` | float | ratio (0-1) | Proportion of tasks assigned (TAR) |
| `duration_seconds` | float | seconds | Wall-clock time for simulation |

### Travel & Distance

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `total_travel_km` | float | kilometers | Total distance traveled by all workers |
| `mean_pickup_distance_km` | float | kilometers | Average empty-travel distance to pickup |
| `empty_km_ratio` | float | ratio (0-1) | Proportion of travel that was empty (no passenger) |

---

## 3. Task Wait Time Metrics

**Data Source**: List of wait times for all completed tasks

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `mean_task_wait_time_min` | float | minutes | Average time from task release to pickup | v1.0 |
| `std_task_wait_time_min` | float | minutes | Standard deviation of wait times | v2.0 |
| `p90_task_wait_time_min` | float | minutes | 90th percentile wait time | v2.0 |
| `p95_task_wait_time_min` | float | minutes | 95th percentile wait time | v2.0 |
| `max_task_wait_time_min` | float | minutes | Maximum wait time observed | v2.0 |
| `cv_task_wait_time` | float | ratio | Coefficient of variation (std/mean) | v2.0 |

**Interpretation**:
- Higher `std` or `cv` = less predictable wait times
- `p90` and `p95` show tail behavior (worst-case scenarios)
- Lower CV indicates more uniform/predictable service

---

## 4. Worker Metrics

### Worker Idle Time Distribution

**Data Source**: `Worker.total_idle_time` for all 15K workers

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `mean_worker_idle_time_min` | float | minutes | Average idle time across all workers | v2.0 |
| `std_worker_idle_time_min` | float | minutes | Standard deviation of idle times | v2.0 |
| `p90_worker_idle_time_min` | float | minutes | 90th percentile (most starved workers) | v2.0 |
| `max_worker_idle_time_min` | float | minutes | Maximum idle time (worst worker) | v2.0 |
| `cv_worker_idle_time` | float | ratio | Coefficient of variation (equity measure) | v2.0 |

**Interpretation**:
- Lower `cv_worker_idle_time` = more equitable idle time distribution
- Higher `p90` or `max` = some workers severely starved
- Fairness strategies should reduce CV

### Worker Task Distribution

**Data Source**: `Worker.completed_tasks` for all workers

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `tasks_per_worker_mean` | float | count | Average tasks per worker | v2.0 |
| `tasks_per_worker_std` | float | count | Standard deviation of task counts | v2.0 |
| `tasks_per_worker_cv` | float | ratio | Coefficient of variation | v2.0 |
| `tasks_per_worker_gini` | float | ratio (0-1) | Gini coefficient (0=perfect equality) | v2.0 |
| `tasks_per_worker_p10` | float | count | 10th percentile (least-served workers) | v2.0 |
| `tasks_per_worker_p50` | float | count | Median tasks per worker | v2.0 |
| `tasks_per_worker_p90` | float | count | 90th percentile (most-served workers) | v2.0 |
| `pct_workers_zero_tasks` | float | ratio (0-1) | Fraction of workers with 0 tasks | v2.0 |
| `pct_workers_single_task` | float | ratio (0-1) | Fraction with only 1 task | v2.0 |

**Interpretation**:
- `gini` is gold standard for inequality (0 = perfect equality, 1 = max inequality)
- High `pct_workers_zero_tasks` indicates starvation problem
- Lower Gini and CV indicate more equitable task distribution

### Worker Utilization

**Data Source**: Computed as (available_time - idle_time) / available_time

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `mean_worker_utilization` | float | ratio (0-1) | Average utilization rate | v2.0 |
| `std_worker_utilization` | float | ratio | Standard deviation of utilization | v2.0 |
| `p10_worker_utilization` | float | ratio | 10th percentile (underutilized) | v2.0 |
| `p90_worker_utilization` | float | ratio | 90th percentile (overutilized) | v2.0 |

**Interpretation**:
- Utilization = 1.0 means worker was always busy
- Utilization = 0.0 means worker never got a task
- Lower std = more uniform workload distribution

---

## 5. Fairness Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `jains_fairness_index` (JFI) | float | ratio (0-1) | Task-based fairness (1=perfect) |
| `ewma_cv` | float | ratio | Coefficient of variation of EWMA values |

**EWMA (Exponentially Weighted Moving Average)**: Tracks cumulative fairness experience for each worker over time, decayed by parameter γ.

---

## 6. System Performance Metrics

### Pickup Distance Distribution

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `mean_pickup_distance_km` | float | km | Average pickup distance | v1.0 |
| `std_pickup_distance_km` | float | km | Standard deviation | v2.0 |
| `p90_pickup_distance_km` | float | km | 90th percentile | v2.0 |
| `max_pickup_distance_km` | float | km | Maximum pickup distance | v2.0 |

**Interpretation**:
- Higher std/CV indicates some workers travel much farther
- Fairness strategies may increase pickup distance variance

### Assignment Timing

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `mean_assignment_delay_sec` | float | seconds | Avg time from release to assignment | v2.0 |
| `std_assignment_delay_sec` | float | seconds | Standard deviation | v2.0 |
| `p90_assignment_delay_sec` | float | seconds | 90th percentile | v2.0 |

**Note**: Assignment delay is different from wait time. Assignment delay is when the decision is made; wait time includes travel to pickup.

### Task Deferral Tracking

| Metric | Type | Unit | Description | Added |
|--------|------|------|-------------|-------|
| `total_deferrals` | int | count | Total times tasks were deferred | v2.0 |
| `pct_tasks_deferred` | float | ratio (0-1) | Fraction of tasks deferred at least once | v2.0 |
| `mean_deferrals_per_task` | float | count | Average deferrals per task | v2.0 |
| `max_deferrals_per_task` | int | count | Max times any task was deferred | v2.0 |

**Interpretation**:
- High deferral metrics indicate `soft_threshold` is actively filtering assignments
- If `total_deferrals` = 0, threshold has no effect
- Deferral only applies to composite strategy

### Backlog

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `peak_backlog` | int | count | Maximum simultaneous unassigned tasks |

---

## 7. Data Formats

### Aggregate Results CSV

**Location**: `experiments_analysis/exp_XXX/data/experiment_XXX_aggregate_results.csv`

**Structure**: One row per experiment

**Columns** (78+ total as of v2.0):

**Experiment Metadata (6 columns)**:
- `experiment_id` - Unique experiment number
- `group` - Experimental group (A, B, C, etc.)
- `name` - Descriptive name
- `description` - Full description
- `strategy` - "greedy" or "composite"
- `timestamp` - ISO 8601 timestamp

**Configuration Parameters (6 columns)**:
- `fairness_weight` (λ₁)
- `starvation_weight` (λ₂)
- `utility_weight` (λ₃)
- `soft_threshold` (θ)
- `normalize_scores` (boolean)
- `gamma` (γ)

**Core Metrics (3 columns)**:
- `completed_tasks`
- `task_assignment_ratio`
- `duration_seconds`

**Task Wait Time Distribution (6 columns)**:
- `mean_task_wait_time_min`
- `std_task_wait_time_min`
- `p90_task_wait_time_min`
- `p95_task_wait_time_min`
- `max_task_wait_time_min`
- `cv_task_wait_time`

**Worker Idle Time Distribution (5 columns)**:
- `mean_worker_idle_time_min`
- `std_worker_idle_time_min`
- `p90_worker_idle_time_min`
- `max_worker_idle_time_min`
- `cv_worker_idle_time`

**Worker Task Distribution (9 columns)**:
- `tasks_per_worker_mean`
- `tasks_per_worker_std`
- `tasks_per_worker_cv`
- `tasks_per_worker_gini`
- `tasks_per_worker_p10`
- `tasks_per_worker_p50`
- `tasks_per_worker_p90`
- `pct_workers_zero_tasks`
- `pct_workers_single_task`

**Worker Utilization (4 columns)**:
- `mean_worker_utilization`
- `std_worker_utilization`
- `p10_worker_utilization`
- `p90_worker_utilization`

**Pickup Distance Distribution (4 columns)**:
- `mean_pickup_distance_km`
- `std_pickup_distance_km`
- `p90_pickup_distance_km`
- `max_pickup_distance_km`

**Assignment Timing (3 columns)**:
- `mean_assignment_delay_sec`
- `std_assignment_delay_sec`
- `p90_assignment_delay_sec`

**Task Deferral Tracking (4 columns)**:
- `total_deferrals`
- `pct_tasks_deferred`
- `mean_deferrals_per_task`
- `max_deferrals_per_task`

**Other Metrics (4 columns)**:
- `total_travel_km`
- `empty_km_ratio`
- `peak_backlog`
- `jains_fairness_index`
- `ewma_cv`

### Individual Experiment JSON

**Location**: `experiments_analysis/exp_XXX/data/exp_YYY_name_summary.json`

**Structure**: Complete summary dictionary including all metrics above plus:
- `full_summary`: Raw simulation output
- Lists (not saved in CSV): 
  - `wait_times`: Individual task wait times
  - `service_times`: Individual service durations
  - `pickup_distances`: Individual pickup distances
  - `assignment_delays`: Individual assignment delays

---

## 8. Diagnostic Mode Data

When `enable_diagnostics=True`:

### Additional Fields in Summary

| Field | Type | Description |
|-------|------|-------------|
| `diagnostic_summary` | dict | Component dominance statistics |
| `diagnostic_tracker` | object | Full diagnostic tracker (not serialized) |

### Diagnostic Summary Structure

```json
{
  "total_assignments": 17248,
  "fairness_dominant": 2103,
  "starvation_dominant": 1891,
  "utility_dominant": 13254,
  "fairness_pct": 12.2,
  "starvation_pct": 11.0,
  "utility_pct": 76.8
}
```

**Use Case**: Understanding which composite score component drives assignments

**Performance Note**: Diagnostic mode adds ~40% overhead. Only enable for mechanism analysis.

---

## 9. Computational Cost Summary

| Category | Metrics | Cost per Experiment | Version |
|----------|---------|---------------------|---------|
| Core metrics | 10 | baseline | v1.0 |
| Wait time stats | 6 | <0.02ms | v2.0 |
| Worker idle stats | 5 | <0.02ms | v2.0 |
| Worker task distribution | 9 | ~0.12ms | v2.0 |
| Utilization rates | 4 | ~0.04ms | v2.0 |
| Pickup distance | 4 | <0.03ms | v2.0 |
| Assignment timing | 3 | <0.02ms | v2.0 |
| Deferral tracking | 4 | <0.01ms | v2.0 |
| **TOTAL** | **~55 metrics** | **~0.5ms** | |

**Overhead**: ~0.00033% of a 25-minute (1,500,000ms) experiment

**Why So Fast?**:
- Data already collected during simulation
- Computed only once at end, not per-event
- Uses optimized NumPy vectorized operations
- Pure in-memory computation (no I/O)

---

## 10. Version History

### v1.0 (Initial Release)
- Basic metrics: completed tasks, TAR, mean wait time, mean pickup distance
- JFI and basic fairness metrics
- Peak backlog, travel distances

### v2.0 (October 2025 - Statistics Enhancement)
**Added Distribution Metrics**:
- Task wait time distribution (std, percentiles, CV)
- Worker idle time distribution (std, percentiles, CV)
- Worker task distribution (Gini, percentiles, zero/single task %)
- Worker utilization rates (mean, std, percentiles)
- Pickup distance distribution (std, percentiles, max)
- Assignment timing metrics (mean, std, p90 delay)
- Task deferral tracking (total, %, mean, max)

**Key Improvements**:
- Can now measure inequality (Gini coefficient)
- Can assess predictability (CV, std dev)
- Can identify tail behavior (percentiles)
- Can track mechanism behavior (deferrals)
- Negligible performance cost (~0.5ms per experiment)

---

## Quick Reference: Key Metrics by Research Question

### "Is the system fair?"
- `jains_fairness_index` - Overall fairness score
- `tasks_per_worker_gini` - Task distribution inequality
- `cv_worker_idle_time` - Worker experience equity
- `ewma_cv` - EWMA fairness spread

### "Is the system efficient?"
- `task_assignment_ratio` - % tasks successfully assigned
- `mean_task_wait_time_min` - Average responsiveness
- `mean_pickup_distance_km` - Travel efficiency
- `empty_km_ratio` - Wasted travel

### "Is the system predictable?"
- `cv_task_wait_time` - Wait time consistency
- `std_task_wait_time_min` - Wait time spread
- `p90_task_wait_time_min` - Worst-case wait time

### "Are there starved workers?"
- `pct_workers_zero_tasks` - % completely starved
- `pct_workers_single_task` - % severely underutilized
- `tasks_per_worker_p10` - Worst 10% experience
- `max_worker_idle_time_min` - Longest idle time

### "Does soft threshold work?"
- `total_deferrals` - How often it activates
- `pct_tasks_deferred` - % of tasks affected
- `mean_deferrals_per_task` - Repeated deferrals

---

## Usage Examples

### Loading and Analyzing Data

```python
import pandas as pd

# Load aggregate results
df = pd.read_csv('experiments_analysis/exp_010/data/experiment_010_aggregate_results.csv')

# Filter for composite strategies
composite = df[df['strategy'] == 'composite']

# Plot fairness vs efficiency
import matplotlib.pyplot as plt
plt.scatter(composite['jains_fairness_index'], 
            composite['mean_task_wait_time_min'],
            c=composite['fairness_weight'])
plt.xlabel('Jain\'s Fairness Index')
plt.ylabel('Mean Task Wait Time (min)')
plt.colorbar(label='λ₁ (Fairness Weight)')
plt.show()

# Find configuration with best Gini coefficient
best_gini = composite.loc[composite['tasks_per_worker_gini'].idxmin()]
print(f"Best inequality: Gini={best_gini['tasks_per_worker_gini']:.3f}")
print(f"  λ₁={best_gini['fairness_weight']}, λ₂={best_gini['starvation_weight']}, λ₃={best_gini['utility_weight']}")
```

---

**For questions or clarifications, refer to experiment-specific README files or contact the research team.**




