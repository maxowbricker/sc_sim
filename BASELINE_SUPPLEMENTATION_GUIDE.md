# Baseline Supplementation Guide

## Overview

This guide explains how to add missing baseline strategies to completed experiments using the `add_baselines_to_experiment.py` tool.

## Why Do This?

As your research evolves, you may add new baseline strategies (like FATP-ANN, Random, etc.) that weren't included in earlier experiments. Rather than rerunning entire experiments, this tool lets you:

1. **Add only missing baselines** to existing experiments
2. **Preserve original data** and configurations
3. **Maintain experiment integrity** by using the same dataset
4. **Save time** by avoiding redundant simulations

## Available Baselines

| Strategy | Key | Description |
|----------|-----|-------------|
| **Greedy** | `greedy` | Distance-only baseline (minimum travel distance) |
| **LAF** | `laf` | Least Allocated First (fairness-only baseline) |
| **EWMA-Only** | `ewma_only` | EWMA fairness metric baseline (γ=0.5) |
| **Random** | `random_assign` | Random assignment baseline |
| **FATP-ANN** | `fatp_ann` | Fairness-Aware Task Planning (µ=0.5, α=0.5) |

## Basic Usage

### 1. Check What's Missing

First, see what strategies exist in an experiment:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim

# Dry run to see what would be added
./venv/bin/python add_baselines_to_experiment.py \
    --exp exp_013_fairness_efficiency_tradeoff \
    --all-missing \
    --dry-run
```

### 2. Add All Missing Baselines

Add all baselines that aren't already in the experiment:

```bash
./venv/bin/python add_baselines_to_experiment.py \
    --exp exp_013_fairness_efficiency_tradeoff \
    --all-missing
```

### 3. Add Specific Baselines

Add only specific baselines:

```bash
./venv/bin/python add_baselines_to_experiment.py \
    --exp exp_013_fairness_efficiency_tradeoff \
    --baselines fatp_ann random_assign
```

## Example Workflows

### Scenario 1: Add FATP-ANN to All Old Experiments

You've just implemented FATP-ANN and want to add it to experiments 011-015:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim

for exp in exp_011_scalability_analysis \
           exp_012_worker_ratio_analysis \
           exp_013_fairness_efficiency_tradeoff \
           exp_014_ewma_tradeoff_exploration \
           exp_015_temporal_ewma_validation; do
    
    echo "Adding FATP-ANN to $exp..."
    ./venv/bin/python add_baselines_to_experiment.py \
        --exp "$exp" \
        --baselines fatp_ann
done
```

### Scenario 2: Comprehensive Baseline Suite

Add all baselines to an experiment that only has Greedy:

```bash
./venv/bin/python add_baselines_to_experiment.py \
    --exp exp_013_fairness_efficiency_tradeoff \
    --baselines laf ewma_only random_assign fatp_ann
```

### Scenario 3: Batch Processing with Log

Add missing baselines to multiple experiments with logging:

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim

for exp in exp_0{11..15}_*; do
    echo "Processing $exp..."
    ./venv/bin/python add_baselines_to_experiment.py \
        --exp "$exp" \
        --all-missing 2>&1 | tee "logs/${exp}_baseline_supplement.log"
done
```

## How It Works

### 1. **Analysis Phase**
The tool analyzes the target experiment to determine:
- Existing strategies in the CSV
- Data configuration (worker/task counts)
- Experiment structure and numbering

### 2. **Data Loading**
- Loads the same dataset (3-hour peak Didi data)
- Applies the same sampling strategy
- Ensures identical test conditions

### 3. **Simulation**
- Runs each missing baseline with deep-copied data
- Uses optimal default parameters from tuning experiments
- Saves individual JSON summaries

### 4. **Integration**
- Appends results to existing CSV
- Maintains experiment ID numbering
- Preserves all original results

## Expected Runtime

| Baseline | Approximate Time (4K workers, 20K tasks) |
|----------|------------------------------------------|
| Greedy | ~5-7 minutes |
| LAF | ~6-8 minutes |
| EWMA-Only | ~7-9 minutes |
| Random | ~6-8 minutes |
| FATP-ANN | ~22-25 minutes |

**Total for all 5 baselines:** ~46-57 minutes per experiment

## Output

### Console Output
```
================================================================================
BASELINE SUPPLEMENTATION TOOL
================================================================================

📊 Analyzing experiment: exp_013_fairness_efficiency_tradeoff
   Results CSV: data/experiment_013_aggregate_results.csv
   Existing runs: 45
   Existing strategies: Greedy_Baseline, L1_2.5_L3_0.5, ...

   📦 Inferred data config:
      Dataset: didi
      Tasks: 20000

📋 Baselines to add:
   • LAF_Baseline (laf)
   • EWMA_Only (ewma_only)
   • Random_Baseline (random_assign)
   • FATP_ANN_Baseline (fatp_ann)

[Simulation progress...]

================================================================================
✅ Added 4 baseline(s) to experiment_013_aggregate_results.csv
================================================================================
```

### Files Created
1. **JSON Summaries**: `exp_XXX_{Baseline_Name}_summary.json`
2. **Updated CSV**: Original CSV with new rows appended

## Verification

After adding baselines, verify the results:

```python
import pandas as pd

# Load updated results
df = pd.read_csv('experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/experiment_013_aggregate_results.csv')

# Check strategies
print("Strategies in experiment:")
print(df['strategy'].value_counts())

# Check new baselines
new_baselines = df[df['exp_id'] > 45]  # Assuming 45 was the last original ID
print("\nNewly added baselines:")
print(new_baselines[['exp_id', 'exp_name', 'jains_fairness_index', 'task_assignment_ratio']])
```

## Troubleshooting

### Issue: "No results CSV found"
**Solution:** Check that the experiment has a CSV file with "aggregate" or "results" in the name.

### Issue: Data configuration mismatch
**Solution:** The tool infers data size from existing JSONs. If it can't detect the correct size, manually edit the data_config in the script.

### Issue: Simulation hangs
**Solution:** Check that all strategies are properly implemented. You can test individual strategies first:

```bash
# Test just one baseline
./venv/bin/python add_baselines_to_experiment.py \
    --exp exp_013_fairness_efficiency_tradeoff \
    --baselines greedy
```

## Advanced Usage

### Custom Parameters

To use custom parameters for a baseline, modify the `BASELINE_STRATEGIES` dictionary in `add_baselines_to_experiment.py`:

```python
BASELINE_STRATEGIES = {
    'ewma_only': {
        'name': 'EWMA_Only_Custom',
        'assignment_strategy': 'ewma_only',
        'strategy_params': {'gamma': 0.7}  # Custom gamma
    },
    ...
}
```

### Different Data Sizes

If your experiment used different data sizes, update the script's data loading:

```python
data_config = {
    'workers': 2000,  # Instead of auto-detected
    'tasks': 10000,
    'dataset': 'didi'
}
```

## Best Practices

1. **Always run --dry-run first** to see what will be added
2. **Back up your CSV files** before running (git commit is your friend!)
3. **Run baselines in batches** if you have many experiments to update
4. **Verify results** after adding baselines
5. **Update analysis notebooks** to include new baselines in plots

## Research Questions Coverage

After adding all baselines, you can comprehensively answer:

- **RQ1**: How do different strategies compare on fairness?
- **RQ2**: What are the efficiency trade-offs?
- **RQ3**: How does your Composite strategy perform vs. all baselines?
- **RQ4**: Is FATP-ANN better than simpler alternatives?

## See Also

- `config.py` - Default parameters for each strategy
- Individual experiment READMEs
- Strategy implementation files in `simulator/strategies/`

