# 🚀 Experiment 010: Quick Start Guide

**Total Runtime**: ~8-9 hours (21 experiments)  
**Ready to Execute**: ✅ Yes  
**Focus**: High-resolution sweep of critical λ₁ range (2.5-4.5)

---

## ⚡ Run the Experiment (Choose One)

### Option 1: Background with nohup (Recommended)
```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_010_extended_boundaries

# Activate venv
source ../../venv/bin/activate

# Run in background
nohup python run_experiment.py > experiment_log.txt 2>&1 &

# Note the process ID
echo $! > experiment_pid.txt

# Monitor progress
tail -f experiment_log.txt
```

### Option 2: Keep Mac Awake (Alternative)
```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_010_extended_boundaries

# Activate venv
source ../../venv/bin/activate

# Run with caffeinate
caffeinate -i python run_experiment.py
```

---

## 📊 Monitor Progress

### Check How Many Experiments Complete
```bash
# Count completed experiments
ls data/exp_010_*/exp_*.json 2>/dev/null | wc -l

# Should go from 0 to 21
```

### View Live Logs
```bash
# Follow the log file
tail -f experiment_log.txt

# View last 50 lines
tail -50 experiment_log.txt
```

### Check Process Status
```bash
# Is it still running?
ps aux | grep run_experiment.py

# Or using saved PID
ps -p $(cat experiment_pid.txt)
```

---

## ⏱️ Expected Timeline

| Time Elapsed | Experiments Complete | What's Happening |
|--------------|---------------------|-------------------|
| 0:00 | 0/21 | Loading dataset |
| 0:25 | 1/21 | Greedy baseline complete |
| 2:00 | 5/21 | ~25% complete |
| 4:00 | 10/21 | ~50% complete |
| 6:00 | 15/21 | ~75% complete |
| 8:00 | 21/21 | ✅ Complete! |

**Each experiment**: ~25 minutes  
**Total duration**: 8-9 hours (much faster than original 18-hour design!)

---

## 🎯 What's Being Tested

### 21 Total Experiments - Focused Pareto Sweep

**1. Greedy Baseline** (1 exp)
- Reference point

**2. High-Resolution Grid** (20 exp)
- **λ₁ (Fairness)**: [2.5, 3.0, 3.5, 4.0, 4.5] ← **5 values in critical gap!**
- **λ₃ (Utility)**: [0.5, 1.0, 1.5, 2.0] ← **4 practical values**
- **Fixed**: λ₂=0.5, θ=0.5

**Grid**: 5 × 4 = 20 configurations

### Why This Design?

**❌ Original Plan**: 44 experiments, ~18 hours
- Many parameters to test
- Some already validated by Exp 009

**✅ Focused Plan**: 21 experiments, ~8-9 hours
- **Fixed λ₂=0.5**: Validated by Exp 009
- **Fixed θ=0.5**: Minimal impact (Exp 008/009)
- **Focus on λ₁ × λ₃**: The most important interaction
- **50% time savings!** ⚡

---

## ✅ Verify It's Working

### After ~30 minutes, check:

```bash
# Should have 1-2 completed experiments
ls data/exp_010_*/exp_*.json | wc -l

# Check the log for success messages
tail -30 experiment_log.txt | grep "✅"
```

**Expected output**:
```
✅ Loaded: 15,000 workers, 20,000 tasks
✅ Simulation complete in XXX seconds
✅ 1 experiment configured
```

---

## 🎯 The Critical Gap We're Filling

### Problem from Exp 009
```
λ₁ values tested: 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 2.0, 5.0
                                                      ^^^^  ^^^
                                                HUGE GAP HERE!
```

**JFI Results from Exp 009**:
- λ₁=2.0: JFI ≈ 0.288
- λ₁=5.0: JFI = 0.294
- **Missing**: λ₁=2.5, 3.0, 3.5, 4.0, 4.5

### Solution (Exp 010)
```
λ₁ values to test: [2.5, 3.0, 3.5, 4.0, 4.5]
                    ^^^^^^^^^^^^^^^^^^^^^^^^
                       FILLING THE GAP!
```

**Goal**: Find the "knee" - where increasing λ₁ stops being worth the cost

---

## ⚠️ Troubleshooting

### Process Won't Start
```bash
# Check if venv is activated
which python  # Should show path with 'venv'

# Check if dependencies are installed
python -c "import pandas, numpy; print('✅ OK')"
```

### Process Appears Stuck
```bash
# Check if running
ps aux | grep python

# Check latest output file timestamp
ls -lt data/exp_010_*/exp_*.json | head -1

# Should show recent timestamp if progressing
```

### Mac Goes to Sleep
```bash
# Kill and restart with caffeinate
kill $(cat experiment_pid.txt)

# Restart
caffeinate -i python run_experiment.py
```

---

## 🎉 When Complete

### Find the Results
```bash
# Find the latest run directory
ls -td data/exp_010_* | head -1

# View aggregate results
cd $(ls -td data/exp_010_* | head -1)
head experiment_010_aggregate_results.csv
```

### Quick Analysis - Did We Find the Knee?
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('data/exp_010_YYYYMMDD_HHMMSS/experiment_010_aggregate_results.csv')

# Filter composite configs
composite = df[df['strategy'] == 'composite']

# Plot JFI vs λ₁ for each λ₃
fig, ax = plt.subplots(figsize=(10, 6))

for l3 in [0.5, 1.0, 1.5, 2.0]:
    subset = composite[composite['utility_weight'] == l3].sort_values('fairness_weight')
    ax.plot(subset['fairness_weight'], subset['jains_fairness_index'], 
            marker='o', label=f'λ₃={l3}')

ax.set_xlabel('λ₁ (Fairness Weight)')
ax.set_ylabel('JFI (Jain\'s Fairness Index)')
ax.set_title('Finding the Knee: JFI vs λ₁')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find best config
best = composite.loc[composite['jains_fairness_index'].idxmax()]
print(f"\n🏆 Best Configuration:")
print(f"   JFI: {best['jains_fairness_index']:.4f}")
print(f"   λ₁: {best['fairness_weight']}")
print(f"   λ₃: {best['utility_weight']}")
print(f"   Wait time: {best['mean_task_wait_time_min']:.2f} min")
```

---

## 📈 Key Questions to Answer

After running, you'll be able to answer:

1. **Where is the knee?**
   - At λ₁=3.0? 3.5? 4.0?
   - Diminishing returns threshold

2. **Is λ₁=5.0 worth it?**
   - Compare JFI at 4.5 vs. 5.0
   - What's the wait time cost?

3. **What's optimal λ₃ for each λ₁?**
   - Does it change as λ₁ increases?
   - Complete interaction map

4. **Best production config?**
   - Highest JFI with reasonable trade-offs
   - Top 3-5 recommendations

---

## 🆘 Need Help?

**Common Issues**:
- Process killed unexpectedly: Check system memory, restart
- Wrong data loaded: Check `load_data('didi', max_workers=15000, max_tasks=20000)` in script
- Slow progress: Normal! Each experiment takes ~25 minutes

**Kill the Process** (if needed):
```bash
# Using saved PID
kill $(cat experiment_pid.txt)

# Or find and kill
ps aux | grep run_experiment
kill <PID>
```

---

## 📋 After Completion Checklist

- [ ] Verify all 21 experiments completed
- [ ] Merge with Exp 009 data (42 + 21 = 63 total)
- [ ] Generate 2D heatmap with filled gap
- [ ] Create Pareto curve with knee highlighted
- [ ] Identify top 5 configurations
- [ ] Compare λ₁=4.5 vs. 5.0
- [ ] Determine optimal practical config
- [ ] Prepare production deployment recommendations

---

## 📊 Grid Visualization

```
Visual representation of the 20 experiments:

     λ₃=0.5   λ₃=1.0   λ₃=1.5   λ₃=2.0
λ₁=2.5  exp_02  exp_03  exp_04  exp_05
λ₁=3.0  exp_06  exp_07  exp_08  exp_09
λ₁=3.5  exp_10  exp_11  exp_12  exp_13
λ₁=4.0  exp_14  exp_15  exp_16  exp_17
λ₁=4.5  exp_18  exp_19  exp_20  exp_21

Plus exp_01: Greedy Baseline
Total: 21 experiments
```

---

## 🎁 Benefits of This Focused Design

1. **50% Faster**: 8-9 hours vs. 18 hours
2. **High Resolution**: 5 values in critical range
3. **Focused**: Tests what matters most
4. **Validated Fixed Params**: λ₂=0.5, θ=0.5 from Exp 009
5. **Complete Map**: Combined with Exp 009 = 63 experiments total

---

**🎯 Current Status**: READY TO RUN  
**⏰ Best Time to Start**: Morning or afternoon (completes same day/evening)  
**📊 Expected Output**: 21 experiment JSON files + 1 aggregate CSV  
**🎁 Main Benefit**: Find the Pareto knee with high resolution! 📈

---

*See `README.md` for full details, `setup.md` for methodology*
