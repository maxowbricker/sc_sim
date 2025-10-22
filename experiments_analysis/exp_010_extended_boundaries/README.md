# Experiment 010: Pareto Frontier High-Resolution Sweep

**Status**: PLANNED  
**Created**: October 20, 2025  
**Expected Duration**: 8-9 hours  
**Total Experiments**: 21 (1 Greedy + 20 Composite)

---

## 🎯 Quick Summary

**Objective**: Conduct a high-resolution sweep of the critical λ₁ range (2.5-4.5) to precisely map the "knee" of the Pareto frontier in the fairness-efficiency trade-off space.

**Why**: Experiment 009 had a major gap between λ₁=2.0 and 5.0. We need to know:
- Is there a sweet spot at λ₁=3.0, 3.5, or 4.0?
- Can we achieve high JFI with lower wait times than λ₁=5.0?
- Where exactly is the "knee" of the Pareto curve?

**Design Philosophy**: Focus on what matters most, with high resolution where it counts.

---

## 📊 What's Being Tested

### High-Resolution 5 × 4 Grid

| Parameter | Value(s) | Why |
|-----------|----------|-----|
| **λ₁ (Fairness)** | **[2.5, 3.0, 3.5, 4.0, 4.5]** | Fill critical gap! |
| **λ₃ (Utility)** | [0.5, 1.0, 1.5, 2.0] | Practical range |
| λ₂ (Starvation) | **0.5** (fixed) | Validated value from Exp 009 |
| θ (Threshold) | **0.5** (fixed) | Minimal impact (Exp 008/009) |

**Grid**: 5 λ₁ values × 4 λ₃ values = **20 configurations**

### Experimental Groups

- **Group A** (1): Greedy baseline
- **Group B** (20): Pareto frontier high-resolution grid

**Total**: 21 experiments (~8-9 hours)

---

## 🚀 How to Run

### Quick Start

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_010_extended_boundaries

# Activate venv
source ../../venv/bin/activate

# Run in background (recommended for 8-9 hour runtime)
nohup python run_experiment.py > experiment_log.txt 2>&1 &

# Monitor progress
tail -f experiment_log.txt

# Or run with caffeinate (Mac - prevents sleep)
caffeinate -i python run_experiment.py
```

### Check Progress

```bash
# Count completed experiments
ls data/exp_010_*/exp_*_summary.json | wc -l

# View latest results
tail -100 experiment_log.txt

# Check if process is running
ps aux | grep run_experiment.py
```

---

## 📂 Output Structure

```
exp_010_extended_boundaries/
├── run_experiment.py
├── setup.md (detailed methodology)
├── README.md (this file)
├── QUICK_START.md (fast execution guide)
├── experiment_log.txt (runtime logs)
└── data/
    └── exp_010_YYYYMMDD_HHMMSS/
        ├── experiment_manifest.json
        ├── experiment_010_aggregate_results.csv  ← Main results
        ├── exp_001_Greedy_Baseline_summary.json
        ├── exp_002_ParetoSweep_L1_2.5_L3_0.5_summary.json
        ├── ...
        └── exp_021_ParetoSweep_L1_4.5_L3_2.0_summary.json
```

---

## 🔬 Key Research Questions

### RQ1: Where is the "knee" of the Pareto curve?
- Exp 009 jumped from λ₁=2.0 (JFI~0.28) to λ₁=5.0 (JFI=0.294)
- **Now testing**: 2.5, 3.0, 3.5, 4.0, 4.5
- Goal: Find where diminishing returns start

### RQ2: Is λ₁=5.0 worth the cost?
- Exp 009: λ₁=5.0 achieved JFI=0.294 but with high wait times
- Can λ₁=4.0 or 4.5 achieve similar JFI more efficiently?
- What's the marginal benefit of each 0.5 increase?

### RQ3: How does λ₃ interact with λ₁?
- Does optimal λ₃ change as λ₁ increases?
- 4 λ₃ values (0.5, 1.0, 1.5, 2.0) for each λ₁
- Complete interaction surface

### RQ4: What's the best production config?
- High-resolution mapping enables confident recommendations
- Identify top 3-5 configurations for different use cases

---

## 📈 Expected Findings

### Primary Outcomes
1. ✅ **Knee Location**: Precise λ₁ value where diminishing returns begin
2. ✅ **Optimal Config**: Best practical balance (likely λ₁ ∈ [3.0, 4.0])
3. ✅ **Complete 2D Map**: High-resolution λ₁ × λ₃ surface
4. ✅ **Production Top 5**: Ranked configurations for deployment

### Comparison to Exp 009
- **Exp 009**: 42 experiments, JFI max=0.294, major gap 2.0-5.0
- **Exp 010**: 21 experiments, fills gap, finds knee
- **Combined**: 63 experiments, complete practical map

---

## 📊 Analysis After Completion

### Planned Visualizations
1. **2D Heatmap**: JFI vs. (λ₁, λ₃) with filled gap
2. **Pareto Curve**: JFI vs. Wait Time (identify knee)
3. **λ₁ Progression**: How JFI, wait time scale with λ₁
4. **Cost-Benefit**: JFI gain per λ₁ increment
5. **Combined with Exp 009**: Complete parameter map

### Deliverables
- High-resolution heatmaps
- Pareto frontier with knee highlighted
- Top 3-5 production configurations
- Deployment guidelines

---

## ⚠️ Important Notes

### Before Running
- ✅ Ensure 8-9 hours of runtime availability
- ✅ Verify system won't sleep (`caffeinate` on Mac)
- ✅ Same dataset as Exp 009 (15K workers, 20K tasks)
- ✅ Confirm sufficient disk space (~50 MB)

### Why This Design?
✅ **Focused**: Targets the most important gap  
✅ **Efficient**: 8-9 hours vs. 18 hours (50% savings!)  
✅ **High Resolution**: 5 values in critical range  
✅ **Practical**: All configs deployable  
✅ **Validated**: Fixed params from Exp 009  

---

## 🔗 Related Experiments

| Experiment | Purpose | Key Finding |
|------------|---------|-------------|
| **Exp 007** | EWMA gamma sensitivity | γ=0.5 is optimal |
| **Exp 008** | Normalization ablation | Normalization essential |
| **Exp 009** | Moderate parameter sweep | Max JFI=0.294, gap identified |
| **Exp 010** | **High-res sweep** | **TBD - Find knee** |

---

## 📝 Quick Reference: Fixed Settings

All Composite experiments use:
- `normalize_scores = True`
- `gamma = 0.5`
- `starvation_weight = 0.5` (fixed - validated from Exp 009)
- `soft_threshold = 0.5` (fixed - minimal impact from Exp 008/009)
- `enable_diagnostics = False`
- `k = 15`
- Dataset: DiDi (15K workers, 20K tasks)

---

## 🎓 Academic Value

This experiment enables us to:
1. State: "We mapped the critical fairness-efficiency trade-off with high resolution"
2. Identify optimal parameters with confidence
3. Provide production-ready recommendations with justification
4. Demonstrate focused, efficient parameter space exploration

**Focus**: High-resolution where it matters, not exhaustive everywhere

---

## 📞 Troubleshooting

### Process Not Starting
```bash
# Check if venv is activated
which python  # Should show path with 'venv'

# Check if dependencies are installed
python -c "import pandas; print('OK')"
```

### Process Appears Stuck
```bash
# Check if actually running
ps aux | grep python

# Check latest output file
ls -lht data/exp_010_*/exp_*.json | head -5
```

### Out of Memory
- Same memory requirements as Exp 009
- ~500 MB per simulation
- Close other applications if needed

---

## 🎯 Why 21 Experiments Instead of 44?

### ❌ Original Design (44 experiments, 18 hours)
- Tested many parameters (λ₂, θ, extreme λ₁ values)
- Comprehensive but time-consuming
- Some parameters already validated by Exp 009

### ✅ New Design (21 experiments, 8-9 hours)
- **Fixed λ₂=0.5**: Exp 009 showed moderate impact, 0.5 is good baseline
- **Fixed θ=0.5**: Exp 008/009 validated minimal impact
- **Focus on λ₁ × λ₃**: The most important interaction
- **High resolution in critical range**: 2.5-4.5 (5 values)

**Result**: 50% time savings, same critical information! ⚡

---

## 📊 Grid Visualization

```
     λ₃=0.5   λ₃=1.0   λ₃=1.5   λ₃=2.0
λ₁=2.5  •       •        •        •
λ₁=3.0  •       •        •        •
λ₁=3.5  •       •        •        •
λ₁=4.0  •       •        •        •
λ₁=4.5  •       •        •        •

Total: 20 configurations
+ 1 Greedy baseline
= 21 experiments
```

---

**Status**: Ready to run  
**Next Action**: Execute `python run_experiment.py`  
**Expected Completion**: 8-9 hours from start time

---

*For detailed methodology, see `setup.md`  
For quick execution guide, see `QUICK_START.md`*
