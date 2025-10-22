# Experiment 009: Comprehensive Parameter Sweep (Post-Normalization)

## Quick Summary

**Status**: 🔄 Planned

**Purpose**: Systematic exploration of the fairness-efficiency parameter space using the normalized scoring fix from Experiment 008.

**Innovation**: First comprehensive parameter sweep with score normalization enabled, resolving the worker idle time paradox that affected Experiment 006.

## Experiment Design

### Total Experiments: 42
- **1** Greedy baseline
- **41** Composite configurations across 7 groups

### Fixed Settings (All Composite Runs)
```python
normalize_scores = True      # Key fix from Experiment 008
gamma = 0.5                  # Stable EWMA value from Experiment 007
enable_diagnostics = False   # Disabled for fast path performance
k = 15                       # Standard nearest neighbor count
```

### Dataset
- **Source**: DiDi spatial crowdsourcing
- **Workers**: 15,000
- **Tasks**: 20,000
- **Replications**: 1 per configuration

### Estimated Duration
~7 hours (42 experiments × ~10 min each)

## Experimental Groups

| Group | Focus | Experiments | Fixed Parameters |
|-------|-------|-------------|------------------|
| **A** | Greedy Baseline | 1 | N/A |
| **B** | L1 × L3 Grid (Core RQ1) | 12 | L2=0.8, Threshold=0.5 |
| **C** | L2 Starvation Ablation | 4 | L1=1.0, L3=1.0, Threshold=0.5 |
| **D** | Threshold Sweep | 4 | L1=1.0, L2=0.8, L3=1.0 |
| **E** | Balanced Grid | 9 | L2=0.8, Threshold=0.5 |
| **F** | High-Fairness Edge (L1=5.0) | 4 | L2=0.8, Threshold=0.5 |
| **G** | Low-Fairness Edge (L1=0.1) | 4 | L2=0.8, Threshold=0.5 |
| **H** | Low-Utility Edge (L3=0.1) | 4 | Threshold=0.5 |

### Parameter Ranges

#### Group B: L1 × L3 Grid (Core Trade-off Mapping)
```
L1 (Fairness):  [0.0, 0.5, 1.0, 2.0]
L3 (Utility):   [0.5, 1.0, 2.0]
L2 (Starvation): 0.8 (fixed)
Threshold:       0.5 (fixed)
```

#### Group C: Starvation Ablation
```
L2 (Starvation): [0.0, 0.5, 1.0, 2.0]
L1 (Fairness):  1.0 (fixed)
L3 (Utility):   1.0 (fixed)
Threshold:      0.5 (fixed)
```

#### Group D: Threshold Sensitivity
```
Threshold:      [0.1, 0.3, 0.6, 0.9]
L1 (Fairness):  1.0 (fixed)
L2 (Starvation): 0.8 (fixed)
L3 (Utility):   1.0 (fixed)
```

## Key Metrics

### Primary
- **Jain's Fairness Index (JFI)**: Worker assignment fairness
- **Task Assignment Ratio (TAR)**: Throughput efficiency
- **Mean Worker Idle Time**: Critical diagnostic from Exp 008

### Secondary
- **Mean Task Wait Time**: System responsiveness
- **Mean Pickup Distance**: Spatial efficiency
- **Total Travel Distance**: System-wide efficiency cost
- **Peak Backlog**: System stress indicator

## Research Questions

1. **RQ1**: What are optimal weights (λ₁, λ₂, λ₃) for balancing fairness and efficiency?
2. **RQ2**: How does fairness weight affect the JFI vs efficiency trade-off?
3. **RQ3**: Does starvation weight significantly impact outcomes?
4. **RQ4**: How sensitive is the system to the soft threshold parameter?

## How to Run

### Prerequisites
```bash
# Ensure virtual environment is activated
cd /path/to/sc_sim
source venv/bin/activate

# Verify data files are available
ls data/didi_workers.csv
ls data/didi_tasks.csv
```

### Execute Experiment
```bash
cd experiments_analysis/exp_009_comprehensive_parameter_sweep/
python run_experiment.py
```

### For Long-Running Execution (Mac)
```bash
caffeinate -i python run_experiment.py
```

### For Windows
```powershell
# Run in PowerShell with execution policy allowing scripts
python run_experiment.py
```

## Expected Outcomes

### Hypothesis 1: Clear Fairness-Efficiency Trade-offs
With normalized scoring, we expect monotonic relationships between λ₁ and JFI, and clear trade-offs with efficiency metrics.

### Hypothesis 2: Minimal Starvation Impact
Based on Experiment 008 showing starvation contributes only ~11-12% to score dominance, Group C should show minimal variation as λ₂ varies.

### Hypothesis 3: Nonlinear Threshold Effects
Low thresholds (0.1-0.3) should allow immediate assignments, while high thresholds (0.6-0.9) may cause significant deferrals.

### Hypothesis 4: Diminishing Returns at Extremes
- High-fairness (λ₁=5.0): High JFI but severe efficiency cost
- Low-fairness (λ₁=0.1): Approaches greedy performance
- Low-utility (λ₃=0.1): System degradation

## Output Files

### Directory Structure
```
data/exp_009_YYYYMMDD_HHMMSS/
├── experiment_manifest.json                    # Metadata
├── experiment_009_aggregate_results.csv        # All results
├── exp_001_Greedy_Baseline_summary.json
├── exp_001_Greedy_Baseline_workers.csv
├── exp_002_L1_0.0_L3_0.5_summary.json
├── exp_002_L1_0.0_L3_0.5_workers.csv
└── ... (84 files: 42 JSON + 42 CSV)
```

### Aggregate CSV Columns
- Configuration: `experiment_id`, `group`, `name`, `strategy`
- Parameters: `fairness_weight`, `starvation_weight`, `utility_weight`, `soft_threshold`
- Primary Metrics: `jains_fairness_index`, `task_assignment_ratio`, `mean_worker_idle_time_min`
- Secondary Metrics: `mean_task_wait_time_min`, `mean_pickup_distance_km`, `total_travel_km`, `peak_backlog`
- Metadata: `duration_seconds`, `timestamp`

## Analysis Workflow

### Step 1: Load Results
```python
import pandas as pd

# Load aggregate results
df = pd.read_csv('data/exp_009_YYYYMMDD_HHMMSS/experiment_009_aggregate_results.csv')

# Filter by group
group_b = df[df['group'] == 'B']  # Core L1 × L3 grid
group_c = df[df['group'] == 'C']  # Starvation ablation
```

### Step 2: Generate Visualizations
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Pareto frontier (Group B)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=group_b, x='task_assignment_ratio', y='jains_fairness_index',
                hue='fairness_weight', size='utility_weight', sizes=(50, 200))
plt.title('Fairness-Efficiency Trade-off (Group B: L1 × L3 Grid)')
plt.xlabel('Task Assignment Ratio (Efficiency)')
plt.ylabel("Jain's Fairness Index (Fairness)")
plt.show()

# Heatmap (Group B)
pivot = group_b.pivot(index='fairness_weight', columns='utility_weight', 
                      values='jains_fairness_index')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('JFI Heatmap: Fairness Weight vs Utility Weight')
plt.show()
```

### Step 3: Statistical Analysis
```python
from scipy import stats

# Test starvation ablation (Group C)
group_c = df[df['group'] == 'C']
f_stat, p_value = stats.f_oneway(
    group_c[group_c['starvation_weight']==0.0]['jains_fairness_index'],
    group_c[group_c['starvation_weight']==0.5]['jains_fairness_index'],
    group_c[group_c['starvation_weight']==1.0]['jains_fairness_index'],
    group_c[group_c['starvation_weight']==2.0]['jains_fairness_index']
)
print(f"Starvation ablation ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
```

### Step 4: Identify Optimal Configurations
```python
# Define composite score for optimization
# Example: Equal weighting of normalized JFI and TAR
df['composite_score'] = (df['jains_fairness_index'] / df['jains_fairness_index'].max() +
                         df['task_assignment_ratio'] / df['task_assignment_ratio'].max()) / 2

# Find top configurations
top_configs = df.nlargest(5, 'composite_score')[['name', 'fairness_weight', 
                                                   'starvation_weight', 'utility_weight',
                                                   'jains_fairness_index', 
                                                   'task_assignment_ratio',
                                                   'composite_score']]
print(top_configs)
```

## Comparison with Experiment 006

### Key Differences
1. **Score normalization enabled** - Primary fix
2. **Diagnostics disabled** - Fast path for performance
3. **Single runs** - No replications
4. **Expanded ranges** - More extreme edge cases
5. **Fixed gamma** - Using optimized value

### Expected Improvements
- **Stable idle times** - No paradox across configurations
- **Clear trade-offs** - Monotonic relationships between parameters and outcomes
- **Faster execution** - ~10 min per experiment vs ~15-20 min in Exp 006
- **Interpretable results** - Component dominance resolved

## Files

- `run_experiment.py`: Main execution script
- `setup.md`: Detailed methodology and configuration tables
- `README.md`: This file
- `data/`: Experiment results (generated after execution)

## Next Steps After Completion

1. ✅ Verify all 42 experiments completed successfully
2. 📊 Generate analysis notebook
3. 📈 Create comprehensive visualization suite
4. 📝 Document key findings and optimal configurations
5. 🔄 Update main experiments README with results
6. 📄 Prepare results for research paper

## Related Experiments

- **Experiment 006**: Original parameter sweep (pre-normalization, showed paradox)
- **Experiment 007**: EWMA gamma optimization (identified γ=0.5 as stable)
- **Experiment 008**: Score normalization diagnostic (resolved idle time paradox)

## References

- Setup details: `setup.md`
- Main experiments index: `../README.md`
- Configuration system: `../../config.py`
- Composite strategy: `../../simulator/strategies/composite.py`



