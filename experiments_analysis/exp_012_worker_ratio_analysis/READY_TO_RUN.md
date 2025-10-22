# Experiment 012: Ready to Run ✅

**Status**: All systems GO  
**Date**: October 22, 2025  
**Expected Duration**: 2.5-3 hours

---

## 🎯 What This Experiment Does

Finds the **optimal worker count** for a fixed 20K task workload by testing 11 different worker populations (2K to 15K) using **stratified temporal sampling** to ensure workers are available when tasks arrive.

---

## ✅ Pre-flight Checklist

- [x] **Stratified sampler implemented** (`data/stratified_sampler.py`)
- [x] **3-hour peak dataset created** (`data/didi/gps_3hour_peak.txt`, `order_3hour_peak.txt`)
- [x] **Experiment script ready** (`run_experiment.py`)
- [x] **Configuration validated** (λ₁=2.0, λ₂=0.5, λ₃=1.0, θ=0.0)
- [x] **Quick test passed** (stratified sampling works, 8.5% early availability)
- [x] **Validation test passed** (94.3% TAR with 4K workers, θ=0.0)

---

## 🚀 How to Run

### Option 1: Foreground (Watch Progress)
```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_012_worker_ratio_analysis
../../venv/bin/python run_experiment.py
```

### Option 2: Background (Unattended - RECOMMENDED)
```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_012_worker_ratio_analysis
nohup ../../venv/bin/python run_experiment.py > experiment_012_run.log 2>&1 &
```

### Monitor Progress
```bash
# Watch log in real-time
tail -f experiment_012_run.log

# Check if still running
ps aux | grep run_experiment.py

# See current experiment number
grep "EXPERIMENT" experiment_012_run.log | tail -1
```

---

## 📊 What to Expect

### Timeline
| Time | Event |
|------|-------|
| 0:00 | Data loading (~1 min) |
| 0:01 | Stratified sampling (~30 sec) |
| 0:02 | Exp 1 starts (2K workers) |
| 0:10 | Exp 2 starts (3K workers) |
| ~2:30 | Exp 11 completes (15K workers) |
| ~2:30 | Results saved |

### Output Files
```
data/exp_012_YYYYMMDD_HHMMSS/
  ├── exp_001_2000workers_summary.json
  ├── exp_002_3000workers_summary.json
  ├── ...
  └── exp_011_15000workers_summary.json

data/
  └── experiment_012_aggregate_results.csv  ← Main results file
```

---

## 🔍 Expected Results

Based on validation tests and Exp 011 findings:

| Worker Count | Expected TAR | Expected Gini | Expected Util |
|--------------|--------------|---------------|---------------|
| 2K | >90% | ~0.35 | ~65% |
| 4K | >90% | ~0.32 | ~45% |
| 6K | >90% | ~0.30 | ~35% |
| 8K | >90% | ~0.30 | ~28% |
| 10K | >90% | ~0.32 | ~23% |
| 15K | >90% | ~0.35 | ~16% |

**Key Insight**: Expect TAR to plateau around 6K workers, with Gini coefficient having a "sweet spot" at 6K-8K workers.

---

## 🎯 What We're Looking For

1. **The "Knee"**: Minimum worker count for >90% TAR
2. **Fairness Sweet Spot**: Worker count with best Gini coefficient
3. **Realistic Ratios**: Tasks-per-worker that produces realistic wait times
4. **Utilization Trade-off**: Balance between efficiency and resource use

---

## ⚠️ Troubleshooting

### If Experiment Fails
1. Check log: `tail -100 experiment_012_run.log`
2. Look for errors in latest experiment JSON files
3. Verify dataset exists: `ls ../../data/didi/gps_3hour_peak.txt`

### If Taking Too Long
- Expected: 10-15 min per experiment
- If >20 min per experiment, check CPU usage
- Safe to interrupt (Ctrl+C) and resume later

### If Low TAR (<85%)
- This would indicate a problem with stratified sampling
- Review `experiment_012_run.log` for worker availability stats
- Each experiment should show >8% workers available at first task

---

## 📈 Next Steps (After Completion)

1. **Analyze Results**
   - Create `analysis.ipynb` following `ANALYSIS_PLAN.md`
   - Generate 6 key plots (TAR, Gini, Wait Time, Utilization, etc.)

2. **Identify Optimal Config**
   - Determine recommended worker count for 20K tasks
   - Document trade-offs in `results.md`

3. **Apply to Future Experiments**
   - Use optimal worker-task ratio for Exp 013+
   - Validate if λ₂=0.5 holds at optimal density

---

## 🎉 Success Criteria

- ✅ All 11 experiments complete successfully
- ✅ All achieve >85% TAR (validates stratified sampling)
- ✅ Clear "knee" observable in plots
- ✅ Results ready for analysis by Friday evening

---

## 🔬 Technical Details

**Sampling Strategy**:
- 12 temporal bins (15 minutes each)
- ~1,667 tasks per bin (stratified)
- Workers sampled proportionally per bin
- Ensures 8-9% availability at first task

**Configuration**:
- Soft threshold: 0.0 (disabled)
- Starvation weight: 0.5 (validated)
- Fairness weight: 2.0 (balanced)
- Utility weight: 1.0 (balanced)

**Dataset**:
- 3-hour peak window (05:18-08:18)
- 16,133 workers available
- 45,464 tasks available
- Sampling: 20,000 tasks, 2K-15K workers

---

**Ready when you are! 🚀**

Run command:
```bash
cd experiments_analysis/exp_012_worker_ratio_analysis && nohup ../../venv/bin/python run_experiment.py > experiment_012_run.log 2>&1 &
```




