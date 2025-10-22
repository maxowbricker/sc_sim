# Experiment 010: Pareto Frontier High-Resolution Sweep - Setup & Methodology

**Created**: October 20, 2025  
**Status**: PLANNED  
**Predecessor**: Experiment 009 (Comprehensive Parameter Sweep)

---

## 🎯 Experimental Objective

**Primary Goal**: Conduct a high-resolution sweep of the critical λ₁ range (2.5-4.5) to precisely map the "knee" of the Pareto frontier in the fairness-efficiency trade-off space.

**Motivation**: Analysis of Experiment 009 revealed:
- **Major gap**: λ₁ between 2.0 and 5.0 (only tested 2.0 and 5.0)
- **Max JFI achieved**: 0.294 at λ₁=5.0
- **Key question**: Is there a "sweet spot" at λ₁=3.0, 3.5, or 4.0 that provides high JFI with lower wait times than λ₁=5.0?

**Design Philosophy**: 
- Focus on the most critical range
- High resolution where it matters most
- Fix validated parameters to isolate λ₁ × λ₃ interaction
- Keep experiment duration practical (~8-9 hours)

---

## 🔬 Research Questions

### RQ1: Where is the "knee" of the Pareto curve?
- At what λ₁ value do diminishing returns start?
- Is λ₁=3.0 or 4.0 better than 5.0 for practical use?
- What is the optimal fairness-efficiency balance?

### RQ2: How does λ₃ (utility) interact with λ₁ in this range?
- Does optimal λ₃ change as λ₁ increases?
- Are there unexpected interaction effects?
- Can we achieve high JFI with moderate wait times?

### RQ3: Can we improve upon Exp 009's maximum JFI?
- Current max: JFI=0.294 at λ₁=5.0, λ₃=0.5
- Can λ₁=4.5 achieve similar JFI with better efficiency?
- Is there a configuration that dominates the λ₁=5.0 configs?

---

## 📊 Experimental Design

### Total Experiments: 21
- **1 Greedy Baseline** (reference)
- **20 Composite configurations** (5 × 4 grid)

### Fixed Global Settings (All Composite Runs)
| Parameter | Value | Justification |
|-----------|-------|---------------|
| `normalize_scores` | `True` | Essential fix from Experiment 008 |
| `gamma` (EWMA) | 0.5 | Stable value from Experiment 007 |
| `λ₂` (Starvation) | **0.5** | Validated "safety net" value from Exp 009 |
| `θ` (Threshold) | **0.5** | Minimal impact (validated in Exp 009) |
| `enable_diagnostics` | `False` | Fast path for performance |
| `k` (nearest neighbors) | 15 | Standard value |

**Rationale for Fixed Parameters**:
- **λ₂=0.5**: Experiment 009 showed starvation weight has moderate impact; 0.5 provides basic starvation prevention without dominating
- **θ=0.5**: Experiment 008 and 009 confirmed minimal impact post-normalization; using middle value

### Dataset
- **Source**: DiDi Chengdu  
- **Workers**: 15,000  
- **Tasks**: 20,000  
- **Same as Exp 009** for direct comparison

---

## 🧪 Experimental Groups

### Group A: Greedy Baseline (1 experiment)
**Purpose**: Efficiency reference point

| Exp ID | Strategy | Description |
|--------|----------|-------------|
| 001 | Greedy | Pure nearest-worker assignment |

**Expected Metrics** (from Exp 009):
- JFI: ~0.259
- Wait time: ~1.9 min
- TAR: ~86%

---

### Group B: Pareto Frontier High-Resolution Grid (20 experiments)
**Purpose**: Map the critical fairness-efficiency trade-off region with high resolution

**Grid Design**:
- **λ₁ (Fairness)**: [2.5, 3.0, 3.5, 4.0, 4.5] ← *5 values in critical gap*
- **λ₃ (Utility)**: [0.5, 1.0, 1.5, 2.0] ← *4 values spanning practical range*
- **Fixed**: λ₂=0.5, θ=0.5

**Grid Visualization**:
```
λ₃ = 2.0:  [exp_005] [exp_010] [exp_015] [exp_020] [exp_021]
λ₃ = 1.5:  [exp_004] [exp_009] [exp_014] [exp_019] [exp_020]
λ₃ = 1.0:  [exp_003] [exp_008] [exp_013] [exp_018] [exp_019]
λ₃ = 0.5:  [exp_002] [exp_007] [exp_012] [exp_017] [exp_018]
           λ₁=2.5    λ₁=3.0    λ₁=3.5    λ₁=4.0    λ₁=4.5
```

| Exp ID Range | λ₁ Range | λ₃ Range | Total Configs |
|--------------|----------|----------|---------------|
| 002-021 | 2.5-4.5 | 0.5-2.0 | 20 |

**Hypothesis**: 
1. JFI will increase with λ₁ but with diminishing returns
2. Optimal configuration likely at λ₁ ∈ [3.0, 4.0]
3. Lower λ₃ (higher distance sensitivity) may provide better fairness at cost of wait time

**Key Predictions**:
- **λ₁=2.5**: Slightly better than Exp 009's λ₁=2.0 (JFI ~0.28)
- **λ₁=3.0**: Potential sweet spot (JFI ~0.285)
- **λ₁=3.5**: High fairness with reasonable trade-offs (JFI ~0.290)
- **λ₁=4.0**: Very high fairness, approaching λ₁=5.0 (JFI ~0.292)
- **λ₁=4.5**: Close to maximum JFI (JFI ~0.293)

---

## 📈 Expected Outcomes

### Primary Findings
1. **Pareto Knee Identification**: Precise λ₁ value where diminishing returns begin
2. **Optimal Practical Configuration**: Best balance for production deployment
3. **λ₁ × λ₃ Interaction Surface**: Complete 2D map of critical region
4. **Comparison to Exp 009**: Direct validation of gap-filling strategy

### Secondary Findings
1. **JFI Ceiling Estimation**: Extrapolate maximum achievable JFI
2. **Cost-Benefit Quantification**: JFI gain per 0.5 increment in λ₁
3. **Wait Time Scaling**: How wait times scale with λ₁ in this range
4. **Utility Weight Optimization**: Best λ₃ for each λ₁ level

---

## ⏱️ Runtime Estimates

**Per Experiment**: ~25 minutes (based on Exp 009 data)  
**Total Experiments**: 21  
**Total Duration**: **~8-9 hours**

**Schedule**:
- Start: Morning or afternoon
- Complete: Same evening or next morning
- Much more manageable than 18-hour runs!

**Comparison to Original Design**:
- Original Exp 010 design: 44 experiments, ~18 hours
- New focused design: 21 experiments, ~8-9 hours
- **Time savings**: ~50%!

---

## 📊 Success Criteria

✅ **Complete Success**:
- All 21 experiments run without errors
- Clear Pareto frontier knee identified
- Optimal λ₁ value determined
- Smooth JFI progression across λ₁ values

⚠️ **Partial Success**:
- 18+ experiments complete (85%)
- Major trends visible
- Knee location approximately identified

❌ **Failure**:
- System instability in this parameter range (unexpected)
- Highly non-linear behavior (surprising given Exp 009 results)
- No improvement over Exp 009 configurations

---

## 🔬 Analysis Plan

After completion, analyze:

1. **2D Heatmap**: JFI as function of λ₁ × λ₃
   - Identify global maximum
   - Highlight Pareto frontier
   - Compare to Exp 009 results

2. **Pareto Curve**: JFI vs. Wait Time
   - Plot all 20 configs + Greedy
   - Identify knee point
   - Overlay Exp 009 results for comparison

3. **λ₁ Progression**: JFI, Wait Time, TAR vs. λ₁
   - Separate curves for each λ₃ value
   - Identify diminishing returns threshold
   - Determine optimal λ₁

4. **Cost-Benefit Analysis**: 
   - JFI gain per 0.5 increase in λ₁
   - Wait time cost per 0.01 increase in JFI
   - Optimal trade-off point

5. **Combined Dataset**: Merge with Exp 009
   - Complete parameter map with filled gap
   - Comprehensive Pareto frontier
   - Production recommendations

---

## 📝 Deliverables

1. **experiment_010_aggregate_results.csv**: All results in one file
2. **Individual JSON summaries**: Per-experiment detailed results
3. **Combined Analysis with Exp 009**: Merged dataset (63 total experiments)
4. **High-Resolution Heatmap**: 2D visualization of critical region
5. **Pareto Frontier Plot**: Clear knee identification
6. **Production Recommendations**: Top 3-5 configurations for deployment

---

## 🚀 How to Run

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_010_extended_boundaries

# Activate virtual environment
source ../../venv/bin/activate

# Run experiment (background with logging)
nohup python run_experiment.py > experiment_log.txt 2>&1 &

# Or run with caffeinate to prevent sleep (Mac)
caffeinate -i python run_experiment.py

# Monitor progress
tail -f experiment_log.txt
```

---

## ⚠️ Warnings & Considerations

1. **Same Dataset as Exp 009**: Essential for valid comparison
2. **Moderate Runtime**: ~8-9 hours (much more manageable)
3. **Disk Space**: Each experiment generates ~2 KB JSON, total ~45 KB
4. **Memory**: Same as Exp 009 (~500 MB per simulation)
5. **Fixed Parameters**: λ₂=0.5, θ=0.5 chosen deliberately

---

## 🔗 Related Experiments

- **Experiment 007**: EWMA gamma sensitivity (established γ=0.5)
- **Experiment 008**: Score normalization ablation (established normalize_scores=True)
- **Experiment 009**: Comprehensive parameter sweep (tested 42 configs, identified gap)
- **Experiment 010** (this): High-resolution sweep of critical λ₁ range (tests 21 configs)

---

## 📊 Coverage Comparison

| Metric | Exp 009 | Exp 010 | Combined |
|--------|---------|---------|----------|
| λ₁ values tested | 9 | +5 | 14 |
| λ₂ values tested | 6 | 0 (fixed) | 6 |
| λ₃ values tested | 7 | 0 (4 overlap) | 7 |
| θ values tested | 5 | 0 (fixed) | 5 |
| Total experiments | 42 | 21 | 63 |
| Critical gap filled | - | ✅ | - |

**Combined Coverage**: Dense mapping of practical parameter space

---

## 🎯 Why This Design?

### ✅ Advantages
1. **Focused**: Targets the most important gap
2. **Efficient**: 8-9 hours vs. 18 hours (50% time savings)
3. **High Resolution**: 5 λ₁ values in critical range
4. **Practical**: All configs are deployable
5. **Validated Fixed Params**: λ₂=0.5, θ=0.5 from Exp 009

### ⚠️ Trade-offs
1. **No λ₂ exploration**: Fixed at 0.5 (but Exp 009 showed minimal impact)
2. **No θ exploration**: Fixed at 0.5 (but Exp 008/009 validated minimal impact)
3. **Limited λ₃ range**: 0.5-2.0 (but covers practical range)

### 🎁 Overall
**Much better balance** of coverage vs. time compared to original 44-experiment design!

---

## 🔍 Key Questions This Will Answer

1. **"Where is the sweet spot?"**
   - Precisely identify optimal λ₁ in the 2.5-4.5 range

2. **"Is λ₁=5.0 worth the cost?"**
   - Compare wait times and efficiency at λ₁=4.5 vs. 5.0
   - Determine if JFI gain justifies the cost

3. **"What's the best production config?"**
   - High-resolution mapping enables confident recommendations

4. **"How smooth is the trade-off?"**
   - Validate assumption of smooth JFI progression
   - Identify any unexpected non-linearities

---

**Next Steps After Completion**:
1. Run comprehensive analysis notebook
2. Merge with Exp 009 results for complete map
3. Generate high-resolution heatmaps and Pareto plots
4. Identify top 3-5 production-ready configurations
5. Prepare deployment recommendations
6. If needed, plan Experiment 011 based on discoveries
