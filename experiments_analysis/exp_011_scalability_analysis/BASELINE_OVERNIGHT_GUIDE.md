# Experiment 011: Overnight Baseline Run Guide

## 🌙 **Tonight: Start the Run**

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_011_scalability_analysis

# Activate venv and run
../../venv/bin/python run_baselines_exp011.py
```

**What it will do:**
- Run 28 simulations (4 baselines × 7 worker counts)
- Estimated time: **7-8 hours**
- Saves individual JSONs to `baselines/` directory
- Creates `baselines_exp011_results.csv` (separate from original)
- **Does NOT touch your original data**

**Output files:**
```
baselines/
├── exp_001_Greedy_2K_summary.json
├── exp_002_Greedy_4K_summary.json
├── ... (28 JSON files)
└── ...

baselines_exp011_results.csv  ← Summary of all baseline runs
```

---

## ☀️ **Morning: Review and Merge**

### Step 1: Check if it completed

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_011_scalability_analysis

# Check for the results file
ls -lh baselines_exp011_results.csv

# Count JSON files (should be 28)
ls baselines/*.json | wc -l
```

### Step 2: Review results

```bash
../../venv/bin/python review_and_merge_baselines.py
```

This interactive script will show you:
- ✅ Summary of all baseline results
- ✅ Comparison with Composite at 4K workers
- ✅ Sanity checks (failed simulations, anomalies)
- ✅ Option to merge into main CSV

### Step 3: Inspect individual results (if needed)

```python
import pandas as pd
import json

# Quick look at summary CSV
df = pd.read_csv('baselines_exp011_results.csv')
print(df.groupby('baseline')[['jains_fairness_index', 'task_assignment_ratio']].describe())

# Inspect a specific run
with open('baselines/exp_001_Greedy_2K_summary.json') as f:
    data = json.load(f)
    print(f"JFI: {data['final_jains_fairness_index']:.4f}")
    print(f"Completed: {data['completed_tasks']}/{data.get('total_tasks', 20000)}")
```

### Step 4: Merge (if satisfied)

Option A: **Use the review script** (recommended)
```bash
../../venv/bin/python review_and_merge_baselines.py
# Choose option 1 to create merged CSV
```

Option B: **Manual merge** (pandas)
```python
import pandas as pd

original = pd.read_csv('data/experiment_011_aggregate_results.csv')
baselines = pd.read_csv('baselines_exp011_results.csv')

# Manually align columns and concatenate
# (See review_and_merge_baselines.py for full logic)
```

### Step 5: Update analysis notebook

```python
# In analysis.ipynb, update to include baselines
df = pd.read_csv('data/experiment_011_aggregate_results_WITH_BASELINES.csv')

# Now you can plot Composite vs. all baselines across worker counts
for strategy in df['config_name'].unique():
    subset = df[df['config_name'] == strategy]
    plt.plot(subset['worker_count'], subset['jains_fairness_index'], label=strategy)
```

---

## 📊 **What You'll Get**

### Before (Current)
```
experiment_011_aggregate_results.csv
├── 7 rows (one per worker count)
└── Only "Balanced" Composite strategy
```

### After (With Baselines)
```
experiment_011_aggregate_results_WITH_BASELINES.csv
├── 35 rows total
│   ├── 7 × Composite (original)
│   ├── 7 × Greedy
│   ├── 7 × LAF
│   ├── 7 × EWMA-Only
│   ├── 7 × Random
│   └── 7 × FATP-ANN
```

### New Plots You Can Create
1. **Scalability curves:** JFI vs. worker count for each strategy
2. **Efficiency scaling:** Wait time vs. worker count
3. **Strategy comparison:** At each worker count, compare all strategies
4. **Throughput scaling:** TAR vs. worker count

---

## ⚠️ **Safety Features**

✅ **Original data never touched** - all new files have different names  
✅ **Review before merging** - interactive script with sanity checks  
✅ **Individual JSONs preserved** - can always re-extract metrics  
✅ **Backup available** - original CSV unchanged until you explicitly replace it  

---

## 🛠️ **Troubleshooting**

### Run didn't complete
Check the terminal output (might have crashed). Look for error messages in the last few lines.

### Some simulations failed (TAR = 0)
The review script will flag these. Check the corresponding JSON file for error details.

### Results look weird
1. Compare with original Composite results at the same worker count
2. Check that worker/task counts are correct
3. Verify strategy parameters in the JSON files

### Want to rerun specific configurations
Edit `run_baselines_exp011.py` to comment out baselines you don't need:

```python
BASELINES = {
    # 'greedy': {...},  # Comment out if already done
    'fatp_ann': {...},  # Only run FATP-ANN
}
```

---

## 🎯 **Expected Results**

Based on Experiment 017 tuning, at 4K workers:

| Strategy | JFI | TAR | Wait Time | Notes |
|----------|-----|-----|-----------|-------|
| **Composite** | ~0.70 | Variable | Variable | Your main strategy |
| **Greedy** | ~0.65 | High | Low | Distance-only baseline |
| **LAF** | ~0.75 | Low | High | Fairness-only baseline |
| **EWMA-Only** | ~0.72 | Medium | Medium | Fairness metric baseline |
| **Random** | ~0.60 | Medium | Medium | Sanity check |
| **FATP-ANN** | ~0.68 | High | Low | State-of-art baseline |

---

## 📧 **Questions?**

If anything goes wrong or you need help interpreting results, the individual JSON files contain:
- Full simulation summary
- All metrics
- Temporal data
- Diagnostic information

Good luck with the overnight run! 🚀

