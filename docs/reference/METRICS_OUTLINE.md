# Detailed Metrics Collection Outline

This document provides a comprehensive overview of all metrics collected throughout the spatial crowdsourcing simulation.

**Single Source of Truth**: All metrics are aggregated at the end of each simulation step by `MetricsManager` (`metrics/manager.py`). The old `Simulation.summary` dictionary has been deprecated.

---

## 1. Core Time-Series Metrics (`tracker.py` - `MetricTracker`)

### 1.1 Task State Metrics
- **`backlog`**: Number of active (unassigned) tasks at current timestep
- **`assigned`**: Number of tasks currently assigned to workers
- **`completed_total`**: Total number of tasks completed so far

### 1.2 Fairness Metrics (Per-Timestep)

#### Jain's Fairness Index (JFI)
- **`jfi`**: Jain's Fairness Index calculated from per-worker completion counts
  - Formula: `(sum(x_i))² / (n * sum(x_i²))`
  - Range: 0 (completely unfair) to 1 (perfectly fair)

#### Utility Difference (UD)
- **`ud`**: Mean absolute deviation of completions per worker
  - Lower values indicate better fairness
  - Measures disparity in task allocation

#### EWMA-Based Fairness Distribution
- **`fairness_mean`**: Mean EWMA fairness value across all active workers
- **`fairness_p90`**: 90th percentile of EWMA fairness values
- **`fairness_max`**: Maximum EWMA fairness value
- **`fairness_min`**: Minimum EWMA fairness value
- **`fairness_std`**: Standard deviation of EWMA fairness values

### 1.3 Task Age Metrics
- **`avg_task_age_min`**: Average age (in minutes) of active + assigned tasks
  - Age = time since task release
- **`max_task_age_min`**: Maximum age of any active/assigned task

### 1.4 Wait Time Metrics

#### Current Timestep Wait Times
- **`current_avg_wait_min`**: Average wait time for tasks completed in this timestep
- **`current_max_wait_min`**: Maximum wait time in this timestep
- **`current_min_wait_min`**: Minimum wait time in this timestep
- **`current_wait_samples`**: Number of wait time samples in this timestep

#### Overall Simulation Wait Times
- **`overall_avg_wait_min`**: Cumulative average wait time across all completed tasks
- **`overall_max_wait_min`**: Maximum wait time across entire simulation
- **`overall_min_wait_min`**: Minimum wait time across entire simulation
- **`total_wait_samples`**: Total number of wait time samples collected

### 1.5 Travel Distance Metrics

#### Current Timestep Travel Distances
- **`current_avg_travel_km`**: Average travel distance for assigned tasks in this timestep
- **`current_max_travel_km`**: Maximum travel distance in this timestep
- **`current_travel_samples`**: Number of travel distance samples in this timestep

#### Overall Simulation Travel Distances
- **`overall_avg_travel_km`**: Cumulative average travel distance across all assigned tasks
- **`total_travel_samples`**: Total number of travel distance samples collected

### 1.6 Task Completion Time Metrics

#### Current Timestep Completion Times
- **`current_avg_completion_min`**: Average completion time for tasks finished in this timestep
  - Completion time = time from assignment (start_time) to finish_time
- **`current_completion_samples`**: Number of completion time samples in this timestep

#### Overall Simulation Completion Times
- **`total_completion_samples`**: Total number of completion time samples collected

### 1.7 Per-Worker Fairness Tracking
- **`_worker_fairness_history`**: Time-series of EWMA fairness per worker
  - Fields: `time`, `worker_id`, `ewma_fairness`, `completed_tasks`, `available`
  - Exported via `get_worker_fairness_dataframe()`

### 1.8 Distribution Data
- **`_wait_time_samples`**: All wait times for distribution analysis
- **`_travel_distance_samples`**: All travel distances for distribution analysis
- **`_completion_time_samples`**: All completion times for distribution analysis

---

## 2. Deferral Tracking (`deferral_tracker.py` - `DeferralTracker`)

### 2.1 Deferral Events
Tracks when tasks are deferred (not immediately assigned):
- **`task_id`**: Unique task identifier
- **`timestamp`**: When the deferral occurred
- **`score`**: Composite score at time of deferral
- **`reason`**: Why deferred (`'below_threshold'` or `'no_candidates'`)

### 2.2 Assignment Events
Tracks when tasks are assigned to workers:
- **`task_id`**: Unique task identifier
- **`timestamp`**: When the assignment occurred
- **`was_deferred`**: Whether this task was previously deferred
- **`deferral_count`**: Number of times this task was deferred
- **`deferral_duration_sec`**: Time from first deferral to assignment (if deferred)

### 2.3 Summary Statistics
- **`immediate_assignments`**: Tasks assigned without deferral
- **`deferred_assignments`**: Tasks assigned after deferral
- **`total_assignments`**: Total number of assignments
- **`pct_benefiting_from_starvation`**: Percentage of tasks that were deferred then assigned
- **`mean_deferral_duration_sec`**: Average time tasks spend deferred
- **`median_deferral_duration_sec`**: Median deferral duration
- **`std_deferral_duration_sec`**: Standard deviation of deferral durations
- **`p95_deferral_duration_sec`**: 95th percentile deferral duration
- **`max_deferral_duration_sec`**: Maximum deferral duration
- **`mean_deferral_count`**: Average number of deferrals per task
- **`max_deferral_count`**: Maximum number of deferrals for any task
- **`total_deferral_events`**: Total number of deferral events
- **`unique_deferred_tasks`**: Number of unique tasks that were deferred

### 2.4 Deferral Reason Breakdown
- **`pct_below_threshold`**: Percentage of deferrals due to score below threshold
- **`pct_no_candidates`**: Percentage of deferrals due to no eligible workers

---

## 3. Diagnostic Tracking (`diagnostic_tracker.py` - `DiagnosticTracker`)

### 3.1 Assignment Records
Detailed per-assignment score component analysis:

#### Raw Component Values
- **`fairness_raw`**: Raw fairness component value
- **`starvation_raw`**: Raw starvation component value
- **`utility_raw`**: Raw utility component value

#### Normalized Component Values
- **`fairness_norm`**: Normalized fairness component (if normalization enabled)
- **`starvation_norm`**: Normalized starvation component (if normalization enabled)
- **`utility_norm`**: Normalized utility component (if normalization enabled)

#### Component Weights
- **`fairness_weight`**: Weight applied to fairness component
- **`starvation_weight`**: Weight applied to starvation component
- **`utility_weight`**: Weight applied to utility component

#### Weighted Component Values
- **`fairness_weighted`**: Weight × normalized (or raw) fairness value
- **`starvation_weighted`**: Weight × normalized (or raw) starvation value
- **`utility_weighted`**: Weight × normalized (or raw) utility value

#### Dominance Metrics
- **`dominant_component`**: Which component ('fairness', 'starvation', or 'utility') has highest weighted value
- **`dominant_value`**: Value of the dominant component
- **`dominance_ratio`**: Ratio of dominant component to sum of others

#### Final Score and Metadata
- **`final_score`**: Final composite score used for assignment decision
- **`was_deferred_before`**: Whether this task was previously deferred
- **`normalization_used`**: Boolean indicating if normalization was applied
- **`assignment_id`**: Unique identifier for this assignment
- **`task_id`**: Task identifier
- **`worker_id`**: Worker identifier
- **`timestamp`**: When assignment occurred

### 3.2 Deferral Records
- **`task_id`**: Task identifier
- **`best_score`**: Best score achieved among candidates
- **`threshold`**: Threshold value that wasn't met
- **`score_gap`**: Difference between threshold and best score
- **`reason`**: Reason for deferral
- **`timestamp`**: When deferral occurred
- **`best_worker_id`**: ID of worker with best score (if any)

### 3.3 Summary Statistics

#### Component Dominance Patterns
- **`dominance_counts`**: Count of assignments where each component dominated
- **`dominance_percentages`**: Percentage of assignments where each component dominated
- **`avg_dominance_ratio_by_component`**: Average dominance ratio when each component dominates
- **`overall_avg_dominance_ratio`**: Overall average dominance ratio

#### Score Statistics
- **`mean_final_score`**: Average final composite score
- **`median_final_score`**: Median final composite score
- **`std_final_score`**: Standard deviation of final scores

#### Component Value Statistics (Raw)
For each component (fairness, starvation, utility):
- **`mean`**: Mean raw value
- **`std`**: Standard deviation
- **`min`**: Minimum value
- **`max`**: Maximum value

#### Deferral Statistics
- **`total_assignments`**: Total number of assignments recorded
- **`total_deferrals`**: Total number of deferrals recorded
- **`deferral_rate`**: Percentage of tasks deferred
- **`mean_score_gap`**: Average gap between threshold and best score
- **`median_score_gap`**: Median score gap
- **`mean_best_score`**: Average best score among deferred tasks

---

## 4. Fairness Metrics (`fairness.py`)

### 4.1 Core Fairness Functions

#### Jain's Fairness Index (JFI)
- **`jains_fairness_index(task_counts)`**: Calculates JFI for a list of task counts
  - Returns value between 0 (unfair) and 1 (perfectly fair)

#### Utility Difference (UD)
- **`utility_difference(worker_utilities)`**: Calculates UD as max - min
  - Lower values indicate better fairness

#### Fairness Loss (FL)
- **`fairness_loss(actual, ideal)`**: Deviation from ideal fair assignment
  - Formula: `sum(|actual_i - ideal_i|) / sum(ideal_i)`

#### Supervisor's Fairness Loss
- **`fairness_loss_supervisor_definition(worker_stats)`**: FL based on ideal task share
  - Uses Input/Output Ratio (IOR) concept from F-Aware paper

#### Input/Output Ratio (IOR)
- **`calculate_input_output_ratio(worker_stats)`**: IOR per worker
  - Formula: `(tasks received) / (tasks within worker's reachable area)`

### 4.2 FairnessMetricsTracker

#### Worker Statistics
- **`completed_tasks`**: Number of tasks completed by worker
- **`total_idle_time`**: Total idle time (in seconds)
- **`fairness_ewma`**: EWMA-based fairness value
- **`last_active_time`**: Timestamp of last activity

#### Current Fairness Metrics
- **`jains_fairness_index_tasks`**: JFI based on task counts
- **`utility_difference_tasks`**: UD based on task counts
- **`utility_difference_idle_time`**: UD based on idle time
- **`fairness_loss_tasks`**: FL based on task counts
- **`ewma_fairness_mean`**: Mean EWMA fairness across workers
- **`ewma_fairness_std`**: Standard deviation of EWMA fairness
- **`ewma_fairness_coefficient_variation`**: CV of EWMA fairness
- **`total_completed_tasks`**: Total tasks completed
- **`active_workers`**: Number of active workers

#### Task Eligibility Tracking
- **`task_eligibility_log`**: Per-task log of eligible workers
  - Fields: `eligible_workers`, `release_time`, `assigned_worker`, `assignment_time`
- **`worker_eligibility_stats`**: Per-worker eligibility statistics
  - Fields: `eligible_tasks`, `actual_tasks`, `total_eligible_distance`

#### Fairness Summary
- **`final_jains_fairness_index`**: Final JFI value
- **`final_utility_difference_tasks`**: Final UD value
- **`final_fairness_loss`**: Final FL value
- **`final_ewma_cv`**: Final EWMA coefficient of variation
- **`mean_jfi_over_time`**: Mean JFI across all timesteps
- **`min_jfi_over_time`**: Minimum JFI value
- **`mean_ewma_cv_over_time`**: Mean EWMA CV across time
- **`max_ewma_cv_over_time`**: Maximum EWMA CV
- **`supervisor_utility_difference`**: UD using supervisor's definition
- **`supervisor_fairness_loss`**: FL using supervisor's definition
- **`mean_input_output_ratio`**: Mean IOR across workers
- **`min_input_output_ratio`**: Minimum IOR
- **`max_input_output_ratio`**: Maximum IOR
- **`workers_with_eligibility_data`**: Number of workers with eligibility tracking
- **`total_task_assignments_tracked`**: Total assignments tracked

---

## 5. Temporal Summary Statistics

### 5.1 Wait Time Evolution
- **`avg_wait_time_trend`**: Time-series of average wait times
- **`max_wait_time_trend`**: Time-series of maximum wait times
- **`min_wait_time_trend`**: Time-series of minimum wait times
- **`time_points`**: Timestamps for all measurements

### 5.2 Task Age Evolution
- **`avg_task_age_trend`**: Time-series of average task ages
- **`max_task_age_trend`**: Time-series of maximum task ages

### 5.3 Fairness Evolution
- **`fairness_mean_trend`**: Time-series of mean fairness
- **`fairness_max_trend`**: Time-series of max fairness
- **`fairness_min_trend`**: Time-series of min fairness
- **`fairness_std_trend`**: Time-series of fairness standard deviation
- **`jfi_trend`**: Time-series of JFI values

### 5.4 Distribution Summaries

#### Wait Time Statistics
- **`mean`**, **`std`**, **`min`**, **`max`**, **`median`**
- **`p25`**, **`p75`**, **`p90`**, **`p95`**: Percentiles
- **`count`**: Number of samples

#### Travel Distance Statistics
- **`mean`**, **`std`**, **`min`**, **`max`**, **`median`**, **`p90`**, **`count`**

#### Completion Time Statistics
- **`mean`**, **`std`**, **`min`**, **`max`**, **`median`**, **`count`**

### 5.5 Simulation Metadata
- **`num_workers_tracked`**: Number of workers in fairness history
- **`simulation_duration`**: Total simulation duration in hours

---

## 6. Data Export Formats

### 6.1 Main Time-Series Metrics
- **CSV/Parquet**: All per-timestep metrics
- Columns: All metrics listed in Section 1

### 6.2 Per-Worker Fairness Data
- **CSV**: Time-series of EWMA fairness per worker
- Columns: `worker_id`, `time`, `ewma_fairness`, `completed_tasks`, `available`

### 6.3 Distribution Data
- **Wait Times CSV**: All wait time samples
- **Travel Distances CSV**: All travel distance samples
- **Completion Times CSV**: All completion time samples

### 6.4 Summary Statistics
- **JSON**: Comprehensive summary with temporal trends and distribution statistics

---

## 7. Metric Collection Points

### 7.1 Per-Timestep Collection
- **`MetricTracker.snapshot()`**: Called every simulation tick
  - Captures current state of all active/assigned/completed tasks
  - Calculates fairness metrics from worker completion counts
  - Aggregates wait times, travel distances, completion times

### 7.2 Per-Assignment Collection
- **`DiagnosticTracker.record_assignment()`**: Called when task is assigned
  - Captures score components, weights, dominance patterns
- **`DeferralTracker.record_assignment()`**: Called when task is assigned
  - Tracks deferral history and duration

### 7.3 Per-Deferral Collection
- **`DiagnosticTracker.record_task_deferred()`**: Called when task is deferred
  - Captures score gap and threshold information
- **`DeferralTracker.record_deferral()`**: Called when task is deferred
  - Tracks deferral reason and score

### 7.4 Per-Task-Release Collection
- **`FairnessMetricsTracker.record_task_release()`**: Called when task is released
  - Determines eligible workers based on reachable distance
  - Updates worker eligibility statistics

### 7.5 Per-Task-Assignment Collection
- **`FairnessMetricsTracker.record_task_assignment()`**: Called when task is assigned
  - Updates actual task counts for eligibility-based fairness metrics

---

## 8. Key Metric Relationships

### 8.1 Fairness Metrics
- **JFI** and **UD** are complementary: JFI measures overall fairness (0-1), UD measures disparity (lower is better)
- **EWMA fairness** provides per-worker temporal fairness tracking
- **FL** measures deviation from ideal distribution

### 8.2 Efficiency Metrics
- **Wait time** = time from task release to assignment
- **Travel distance** = distance worker travels to pickup task
- **Completion time** = time from assignment to task completion
- **Task age** = current age of unassigned/assigned tasks

### 8.3 Deferral Metrics
- **Deferral rate** indicates how often starvation prevention is triggered
- **Deferral duration** measures how long tasks wait before assignment
- **Score gap** indicates how far below threshold deferred tasks were

### 8.4 Diagnostic Metrics
- **Dominance patterns** show which component (F/S/U) drives most assignments
- **Component statistics** reveal distribution of raw and normalized values
- **Weighted components** show actual contribution to final score

---

## 9. Research Questions Addressed

### RQ1: Fairness-Efficiency Tradeoff
- **JFI**, **UD**, **FL** for fairness
- **Wait time**, **travel distance**, **completion time** for efficiency
- **Pareto frontier** analysis using these metrics

### RQ2: Component Effectiveness
- **Dominance patterns** from DiagnosticTracker
- **Component statistics** (mean, std, min, max)
- **Weight analysis** via weighted component values

### RQ3: Starvation Prevention
- **Deferral statistics** from DeferralTracker
- **Deferral duration** and **count** distributions
- **Percentage benefiting from starvation prevention**

### RQ4: Temporal Behavior
- **Time-series trends** for all metrics
- **EWMA fairness evolution** per worker
- **Task age evolution** over time

---

This comprehensive metrics collection system enables detailed analysis of fairness, efficiency, starvation prevention, and temporal dynamics in the spatial crowdsourcing simulation.

