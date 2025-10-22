# Experiment 011: Scalability Analysis

**Status**: RUNNING ⏳  
**Started**: October 21, 2025, 1:32 PM  
**Expected Completion**: ~5:00-5:30 PM (3-3.5 hours)

---

## Quick Overview

**Research Question**: How does system performance and fairness scale with worker population size?

**Design**:
- **Worker Counts**: 2.5K, 5K, 10K, 15K
- **Configurations**: 2 Pareto-efficient setups (Balanced & High Fairness)
- **Total Experiments**: 8
- **Duration**: ~3-3.5 hours

---

## Experiment Matrix

| ID | Workers | Config | λ₁ | λ₂ | λ₃ | θ | Runtime |
|----|---------|--------|-----|-----|-----|-----|---------|
| 1 | 2,500 | Balanced | 2.0 | 0.8 | 1.0 | 0.5 | ~15 min |
| 2 | 2,500 | HighFairness | 3.5 | 0.8 | 0.5 | 0.5 | ~15 min |
| 3 | 5,000 | Balanced | 2.0 | 0.8 | 1.0 | 0.5 | ~20 min |
| 4 | 5,000 | HighFairness | 3.5 | 0.8 | 0.5 | 0.5 | ~20 min |
| 5 | 10,000 | Balanced | 2.0 | 0.8 | 1.0 | 0.5 | ~23 min |
| 6 | 10,000 | HighFairness | 3.5 | 0.8 | 0.5 | 0.5 | ~23 min |
| 7 | 15,000 | Balanced | 2.0 | 0.8 | 1.0 | 0.5 | ~25 min |
| 8 | 15,000 | HighFairness | 3.5 | 0.8 | 0.5 | 0.5 | ~25 min |

---

## Monitoring Progress

**Check if running**:
```bash
ps aux | grep run_experiment.py | grep -v grep
```

**View live output**:
```bash
tail -f exp_011_output.log
```

**Check completed experiments**:
```bash
ls -lh experiments_analysis/exp_011_scalability_analysis/data/exp_011_*/
```

---

## What's Being Collected

All **78+ metrics** from v2.0 including:

### Core Metrics
- Task Assignment Ratio (TAR)
- Jain's Fairness Index (JFI)
- Mean wait time

### NEW v2.0 Metrics ✨
- **Gini coefficient** - inequality measure
- **Worker utilization** - efficiency per worker
- **Task deferrals** - threshold impact
- **Distribution statistics** - std, percentiles, CV for all metrics

### Scalability Insights
- Performance per worker
- Fairness improvement rate
- Resource utilization efficiency

---

## Expected Findings

1. **TAR should increase** with more workers (more assignment options)
2. **Wait time should decrease** with more workers (closer workers available)
3. **Gini should decrease** with more workers (easier to distribute fairly)
4. **Utilization should decrease** with more workers (more competition for tasks)

---

## Files Generated

### Individual Results
- `data/exp_011_TIMESTAMP/exp_001_2K_Balanced_summary.json`
- `data/exp_011_TIMESTAMP/exp_002_2K_HighFairness_summary.json`
- ... (8 total)

### Aggregate Results
- `data/experiment_011_aggregate_results.csv` - All metrics for all experiments

---

## Next Steps (After Completion)

1. **Load results**: `df = pd.read_csv('data/experiment_011_aggregate_results.csv')`
2. **Plot scaling trends**: TAR, Gini, wait time vs worker count
3. **Compare configs**: How do Balanced vs HighFairness scale?
4. **Identify optimal density**: Best task-to-worker ratio

---

## Documentation

- **Detailed Setup**: See `setup.md`
- **Data Dictionary**: See `../../DATA_DICTIONARY.md`
- **Implementation**: See `run_experiment.py`

---

**Process ID**: 59339  
**Log File**: `exp_011_output.log`  
**Output**: `data/exp_011_20251021_133251/`



