# Experiment 017: FATP-ANN Parameter Tuning

## Objective

Find optimal or near-optimal parameter settings for the FATP-ANN (Fairness-Aware Task Planning with Approximate Nearest Neighbor) strategy by exploring combinations of key parameters.

## Research Questions

1. **RQ1**: How does the decay factor (μ) affect the balance between waiting tasks and spatial efficiency?
2. **RQ2**: How does distance emphasis (α_scale) impact fairness and efficiency metrics?
3. **RQ3**: What parameter combination achieves the best fairness-efficiency trade-off?

## Setup

### Dataset
- **Source**: 3-hour peak Didi dataset
- **Size**: 4,000 workers / 20,000 tasks
- **Expiry**: 15-minute task expiry
- **Sampling**: Stratified temporal sampling for representativeness

### Strategy Configuration
- **Strategy**: `fatp_ann`
- **Optimization**: `use_k_nearest=False` (full scan, 22 min/sim)
- **Parameters Tested**:
  - **mu (decay factor)**: [0.01, 0.1, 0.5]
    - Controls how quickly task utility decays with wait time
    - Lower = slower decay = more emphasis on old tasks
    - Higher = faster decay = more emphasis on recent tasks
  - **alpha_scale (distance emphasis)**: [0.5, 1.0, 2.0, 5.0]
    - Scaling factor for base utility (task distance)
    - Lower = less emphasis on task distance
    - Higher = more emphasis on longer tasks

### Parameter Grid (12 Simulations)

| Run | mu   | alpha_scale | Notes                                    |
|-----|------|-------------|------------------------------------------|
| 1   | 0.01 | 0.5         | Slow decay, less distance emphasis       |
| 2   | 0.01 | 1.0         | Slow decay, default distance emphasis    |
| 3   | 0.01 | 2.0         | Slow decay, moderate distance emphasis   |
| 4   | 0.01 | 5.0         | Slow decay, high distance emphasis       |
| 5   | 0.1  | 0.5         | Paper's mu, less distance emphasis       |
| 6   | 0.1  | 1.0         | Paper's mu, default alpha_scale          |
| 7   | 0.1  | 2.0         | Paper's mu, moderate distance emphasis   |
| 8   | 0.1  | 5.0         | Paper's mu, high distance emphasis       |
| 9   | 0.5  | 0.5         | Faster decay, less distance emphasis     |
| 10  | 0.5  | 1.0         | Faster decay, default distance emphasis  |
| 11  | 0.5  | 2.0         | Faster decay, moderate distance emphasis |
| 12  | 0.5  | 5.0         | Faster decay, high distance emphasis     |

## Expected Runtime

- **Per Simulation**: ~22 minutes
- **Total Runtime**: ~4.4 hours (264 minutes)

## Metrics Tracked

### Fairness Metrics
- **Jain's Fairness Index (JFI)**: Overall task distribution fairness
- **Gini Coefficient**: Income inequality measure applied to task distribution

### Efficiency Metrics
- **Task Assignment Ratio (TAR)**: Percentage of tasks successfully assigned
- **Mean Wait Time**: Average task waiting time before assignment
- **P95 Wait Time**: 95th percentile task waiting time
- **Worker Utilization**: Percentage of time workers are busy
- **Empty Kilometers**: Total distance traveled without passengers

### Completion Metrics
- **Total Tasks**: Total tasks in simulation
- **Completed Tasks**: Successfully completed tasks

## Output Files

1. **Individual JSON Summaries**:
   - Format: `exp_017_run{NN}_mu{mu}_alpha{alpha}.json`
   - Contains: Full simulation summary with all metrics

2. **Aggregate CSV**:
   - File: `experiment_017_results.csv`
   - Contains: All runs with key metrics for comparison

3. **Analysis Notebook**:
   - File: `analysis.ipynb`
   - Contains: Visualizations and statistical analysis

## Running the Experiment

```bash
cd experiments_analysis/exp_017_fatp_ann_tuning
../../venv/bin/python run_experiment.py 2>&1 | tee experiment_017_run.log
```

Or in the background:
```bash
nohup ../../venv/bin/python -u run_experiment.py > experiment_017_run.log 2>&1 &
```

## Analysis Plan

1. **Parameter Sensitivity Analysis**:
   - Plot JFI vs. mu for each alpha_scale
   - Plot TAR vs. alpha_scale for each mu
   - Identify regions of parameter space with best performance

2. **Trade-off Analysis**:
   - Scatter plot: JFI vs. Mean Wait Time
   - Identify Pareto frontier
   - Highlight parameter combinations on frontier

3. **Heatmaps**:
   - mu × alpha_scale heatmaps for key metrics (JFI, TAR, Wait Time)
   - Identify optimal regions visually

4. **Recommendations**:
   - Best overall configuration
   - Best for fairness-focused scenarios
   - Best for efficiency-focused scenarios
   - Sensitivity analysis: How robust is each configuration?

## Expected Insights

1. **mu Impact**: Lower mu values should favor fairness (giving priority to waiting tasks)
2. **alpha_scale Impact**: Higher alpha_scale should favor longer tasks (better for workers)
3. **Trade-offs**: Clear trade-off between fairness and efficiency expected
4. **Optimal Range**: Identify parameter combinations that balance both objectives


