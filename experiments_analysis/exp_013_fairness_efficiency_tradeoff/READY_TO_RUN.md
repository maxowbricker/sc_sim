# Experiment 013: Pre-Flight Checklist & Execution Guide

**Last Updated**: October 24, 2025

---

## ✅ Pre-Flight Checklist

### 1. Data Files
- [ ] `data/didi/gps_3hour_peak.txt` exists
- [ ] `data/didi/order_3hour_peak.txt` exists
- [ ] Data files are from validated 3-hour peak window (05:18-08:18)

### 2. Code Dependencies
- [ ] Virtual environment activated (`venv/bin/python`)
- [ ] All required packages installed (pandas, numpy, etc.)
- [ ] `data/stratified_sampler.py` exists and works
- [ ] Recent bug fixes applied (deep copy fix from Exp 012)

### 3. System Resources
- [ ] ~8 hours of runtime available
- [ ] ~2-3GB RAM available
- [ ] ~500MB disk space for results
- [ ] Laptop can run unattended (power, no sleep mode)

### 4. Previous Experiments
- [ ] Exp 012 completed and validated (4K workers optimal)
- [ ] Exp 011 validated θ=0.0 configuration
- [ ] Exp 009 validated λ₂=0.5 configuration

---

## 🚀 Execution Steps

### Option 1: Interactive Run (Recommended for Testing)

```bash
# Navigate to experiment directory
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_013_fairness_efficiency_tradeoff

# Run directly (monitor in terminal)
../../venv/bin/python run_experiment.py
```

**Pros**: Can see progress in real-time, easy to interrupt  
**Cons**: Requires terminal to stay open

---

### Option 2: Background Run (Recommended for Production)

```bash
# Navigate to project root
cd /Users/maxapple/Documents/GitHub/sc_sim

# Run in background with unbuffered output
cd experiments_analysis/exp_013_fairness_efficiency_tradeoff && \
nohup ../../venv/bin/python -u run_experiment.py > experiment_013_run.log 2>&1 &
```

**Pros**: Can close terminal, survives disconnects  
**Cons**: Need to check log file for progress

---

## 📊 Monitoring Progress

### Check if Running
```bash
ps aux | grep "run_experiment.py" | grep -v grep
```

### View Real-Time Progress
```bash
tail -f experiments_analysis/exp_013_fairness_efficiency_tradeoff/experiment_013_run.log
```

### Check Last 50 Lines
```bash
tail -50 experiments_analysis/exp_013_fairness_efficiency_tradeoff/experiment_013_run.log
```

### Count Completed Experiments
```bash
ls experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/exp_013_*/exp_*_summary.json | wc -l
```

---

## ⏱️ Expected Timeline

| Time | Progress | Experiments Complete |
|------|----------|---------------------|
| T+0:00 | Start | 0/73 |
| T+0:30 | Sampling Complete | 0/73 |
| T+1:00 | Running | ~7/73 (10%) |
| T+2:00 | Running | ~18/73 (25%) |
| T+4:00 | Running | ~36/73 (50%) |
| T+6:00 | Running | ~55/73 (75%) |
| T+8:00 | Complete | 73/73 (100%) |

**Average per experiment**: 6.6 minutes (based on Exp 012 with 4K workers)

---

## 🛑 Emergency Stop

### Stop Gracefully
```bash
# Find process ID
ps aux | grep "run_experiment.py" | grep -v grep

# Send interrupt signal (allow current experiment to finish)
kill -INT <PID>
```

### Force Stop
```bash
pkill -f "run_experiment.py"
```

---

## ✅ Verification After Completion

### 1. Check All Experiments Completed
```bash
# Should show 73
ls experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/exp_013_*/exp_*_summary.json | wc -l
```

### 2. Check Aggregate CSV Exists
```bash
ls -lh experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/experiment_013_aggregate_results.csv
```

### 3. Quick Data Quality Check
```bash
# Check CSV has 74 rows (1 header + 73 experiments)
wc -l experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/experiment_013_aggregate_results.csv
```

### 4. Verify No Failed Experiments
```bash
# Search for error messages in log
grep -i "FAILED" experiments_analysis/exp_013_fairness_efficiency_tradeoff/experiment_013_run.log
```

### 5. Check TAR Range
```bash
# Quick Python check
venv/bin/python << 'EOF'
import pandas as pd
df = pd.read_csv('experiments_analysis/exp_013_fairness_efficiency_tradeoff/data/experiment_013_aggregate_results.csv')
print(f"TAR Range: {df['task_assignment_ratio'].min():.1%} to {df['task_assignment_ratio'].max():.1%}")
print(f"Mean TAR: {df['task_assignment_ratio'].mean():.1%}")
print(f"Experiments with TAR < 85%: {(df['task_assignment_ratio'] < 0.85).sum()}")
EOF
```

---

## 📈 Next Steps After Completion

1. **Verify Results** (use checklist above)
2. **Create Analysis Notebook** (`analysis.ipynb`)
3. **Generate Visualizations**:
   - 2D heatmaps (JFI, Wait Time, Gini)
   - Pareto frontier plot
   - 3D surface plots
   - Contour plots
4. **Identify Optimal Parameters**
5. **Write Results Summary** (`RESULTS.md`)

---

## 🔧 Troubleshooting

### Problem: Experiment Fails Immediately
**Solution**: Check data files exist and venv is activated

### Problem: Low TAR (<85%)
**Solution**: Should not happen with 4K workers and θ=0.0, check logs for issues

### Problem: Runs Too Slow (>10 min per experiment)
**Solution**: Normal variation, but check system resources (CPU, memory)

### Problem: Process Killed
**Solution**: Check system memory, may need to close other applications

---

## 📋 Execution Log

**Date Started**: _________________  
**Date Completed**: _________________  
**Duration**: _________________  
**Successful Experiments**: _______ / 73  
**Issues Encountered**: _________________  
**Notes**: _________________

---

**Experiment designed by**: Max  
**Validation**: Exp 011 (θ=0.0), Exp 012 (4K workers), Exp 009 (λ₂=0.5)  
**Ready to run**: YES ✅



