# Experiment 009: Comprehensive Parameter Sweep (Post-Normalization) - Setup & Methodology

**Created**: October 20, 2025  
**Status**: ✅ COMPLETE  
**Duration**: ~18 hours  
**Predecessor**: Experiment 008 (Score Normalization Ablation)

---

## 🎯 Experimental Objective

**Primary Goal**: Conduct the first comprehensive parameter sweep with normalized scoring to map the complete fairness-efficiency trade-off space.

**Motivation**: Experiment 008 successfully resolved the "Worker Idle Time Paradox" by implementing score normalization (min-max scaling). With this fix in place, we can now confidently explore the parameter space without the confounding effects of mis-scaled score components.

**Key Innovation**: First large-scale parameter sweep (42 experiments) with `normalize_scores=True`, enabling valid exploration of fairness-efficiency trade-offs.

---

## 🔬 Research Questions

### RQ1: What is the optimal balance between fairness, starvation, and utility?
- Which parameter combinations achieve the best fairness-efficiency trade-offs?
- Where is the Pareto frontier?
- What is the "sweet spot" for practical deployment?

### RQ2: Has normalization eliminated the idle time paradox?
- Do all configurations maintain reasonable wait times?
- Are score components properly balanced?
- No catastrophic feedback loops?

### RQ3: Which parameters have the strongest impact?
- How does λ₁ (fairness) affect JFI?
- How does λ₃ (utility) affect wait times?
- How does λ₂ (starvation) affect max wait times?
- Does θ (threshold) still matter post-normalization?

### RQ4: Are there non-linear sweet spots?
- Do parameter interactions reveal unexpected optima?
- Are there diminishing returns thresholds?
- Any regions to avoid?

---

## 📊 Experimental Design

### Total Experiments: 42
- **1 Greedy Baseline** (reference)
- **41 Composite configurations** (8 groups, systematic exploration)

### Fixed Global Settings (All Composite Runs)
| Parameter | Value | Justification |
|-----------|-------|---------------|
| `normalize_scores` | `True` | **Essential fix from Experiment 008** |
| `gamma` (EWMA) | 0.5 | Stable value from Experiment 007 |
| `enable_diagnostics` | `False` | Fast path for performance (no overhead) |
| `k` (nearest neighbors) | 15 | Standard value |

### Dataset
- **Source**: DiDi Chengdu  
- **Workers**: 15,000  
- **Tasks**: 20,000  
- **Same as Experiments 007, 008** for continuity

---

## 🧪 Experimental Groups

### Group A: Greedy Baseline (1 experiment)
**Purpose**: Efficiency reference point

| Exp ID | Strategy | Description |
|--------|----------|-------------|
| 001 | Greedy | Pure nearest-worker assignment |

**Expected Metrics**:
- JFI: ~0.26 (low fairness)
- Wait time: ~1.9 min (high efficiency)
- TAR: ~86% (baseline)

---

### Group B: L1 × L3 Grid Sweep (12 experiments)
**Purpose**: Core RQ1 mapping - explore primary fairness-efficiency trade-off

**Parameters**:
- λ₁ (Fairness): [0.0, 0.5, 1.0, 2.0] ← *From none to high*
- λ₃ (Utility): [0.5, 1.0, 2.0] ← *Low to high*
- Fixed: λ₂=0.8, θ=0.5

**Design Rationale**:
- λ₁=0.0: No fairness (test utility+starvation only)
- λ₁=0.5, 1.0: Moderate fairness
- λ₁=2.0: High fairness
- λ₃ varies to see interaction with utility

| Exp ID | λ₁ | λ₃ | Expected Behavior |
|--------|----|----|--------------------|
| 002-013 | 0.0-2.0 | 0.5-2.0 | Core trade-off mapping |

**Hypothesis**: JFI increases with λ₁; wait time increases with λ₁ and decreases with λ₃.

---

### Group C: L2 Starvation Ablation (4 experiments)
**Purpose**: Test impact of starvation component (λ₂)

**Parameters**:
- λ₂ (Starvation): [0.0, 0.5, 1.0, 2.0] ← *Ablation study*
- Fixed: λ₁=1.0, λ₃=1.0, θ=0.5

**Design Rationale**:
- λ₂=0.0: Completely disable starvation component
- λ₂=0.5, 1.0, 2.0: Increasing starvation prevention

| Exp ID | λ₂ | Expected Impact |
|--------|----|----------------|
| 014-017 | 0.0-2.0 | Max wait time reduction |

**Hypothesis**: λ₂ primarily affects max wait times, minimal impact on mean metrics.

**Key Finding from Exp 008**: Starvation only contributed ~11-12% to score dominance, suggesting it may have limited impact.

---

### Group D: Soft Threshold Sweep (4 experiments)
**Purpose**: Test threshold sensitivity post-normalization

**Parameters**:
- θ (Threshold): [0.1, 0.3, 0.6, 0.9] ← *Low to high*
- Fixed: λ₁=1.0, λ₂=0.8, λ₃=1.0

**Design Rationale**:
- θ=0.1: Nearly immediate assignment
- θ=0.3, 0.6: Moderate thresholds
- θ=0.9: Very strict threshold

| Exp ID | θ | Expected Behavior |
|--------|---|-------------------|
| 018-021 | 0.1-0.9 | Deferral rate changes |

**Hypothesis**: Post-normalization, threshold should have minimal impact (Exp 008 showed low deferral rates).

---

### Group E: Balanced Grid Sweep (9 experiments)
**Purpose**: Fine-grained exploration of promising regions

**Parameters**:
- λ₁ (Fairness): [0.2, 0.4, 0.6] ← *Medium fairness*
- λ₃ (Utility): [1.2, 1.6, 2.0] ← *Medium-high utility*
- Fixed: λ₂=0.8, θ=0.5

**Design Rationale**:
- Focus on "balanced" region (not extreme fairness or utility)
- Finer granularity to find sweet spot
- Practical configurations for deployment

| Exp ID | λ₁ | λ₃ | Purpose |
|--------|----|----|--------------------|
| 022-030 | 0.2-0.6 | 1.2-2.0 | Sweet spot identification |

**Hypothesis**: Optimal practical configuration likely in this region.

---

### Group F: High-Fairness Edge (4 experiments)
**Purpose**: Explore extreme fairness (λ₁=5.0)

**Parameters**:
- λ₁ (Fairness): [5.0] ← *Very high fairness*
- λ₃ (Utility): [0.5, 1.0, 1.5, 2.0] ← *Various utility weights*
- Fixed: λ₂=0.8, θ=0.5

**Design Rationale**:
- Push fairness to high value (λ₁=5.0)
- Test how different λ₃ values interact with extreme fairness
- Find maximum achievable JFI

| Exp ID | λ₁ | λ₃ | Expected JFI |
|--------|----|----|--------------------|
| 031-034 | 5.0 | 0.5-2.0 | High (>0.28?) |

**Hypothesis**: JFI will be highest in this group, but at cost of increased wait times.

---

### Group G: Low-Fairness Edge (4 experiments)
**Purpose**: Explore near-Greedy behavior (λ₁=0.1)

**Parameters**:
- λ₁ (Fairness): [0.1] ← *Very low fairness*
- λ₃ (Utility): [0.5, 1.0, 1.5, 2.0] ← *Various utility weights*
- Fixed: λ₂=0.8, θ=0.5

**Design Rationale**:
- Minimal fairness weight
- Should behave close to Greedy
- Test if small λ₁ provides any benefit

| Exp ID | λ₁ | λ₃ | Expected Behavior |
|--------|----|----|--------------------|
| 035-038 | 0.1 | 0.5-2.0 | Near-Greedy performance |

**Hypothesis**: Metrics should closely match Greedy baseline.

---

### Group H: Low-Utility Edge (4 experiments)
**Purpose**: Explore fairness-dominated configurations (λ₃=0.1)

**Parameters**:
- λ₁ (Fairness): [1.0] ← *Moderate fairness*
- λ₂ (Starvation): [0.5, 1.0, 1.5, 2.0] ← *Various starvation weights*
- λ₃ (Utility): [0.1] ← *Very low utility*
- Fixed: θ=0.5

**Design Rationale**:
- Minimize utility component
- Let fairness and starvation dominate
- Test extreme trade-off scenario

| Exp ID | λ₂ | Expected Behavior |
|--------|----|-------------------|
| 039-042 | 0.5-2.0 | High fairness, high wait times |

**Hypothesis**: Fairness will be high, but efficiency will suffer (high wait times).

---

## 📈 Expected Outcomes

### Primary Findings
1. **Pareto Frontier Identification**: Clear trade-off curve between JFI and wait time
2. **Optimal Configuration**: "Sweet spot" for balanced performance (likely in Group E)
3. **Parameter Sensitivity**: Ranking of λ₁, λ₂, λ₃, θ by impact
4. **Normalization Validation**: Confirmation that paradox is resolved

### Secondary Findings
1. **Non-linear Effects**: Any unexpected interactions or sweet spots
2. **Diminishing Returns**: Thresholds where increasing parameters stops helping
3. **Threshold Impact**: Confirm minimal effect post-normalization
4. **Starvation Utility**: Validate if λ₂ is worth keeping

---

## ⏱️ Runtime Details

**Per Experiment**: ~25 minutes (based on Exp 008)  
**Total Experiments**: 42  
**Total Duration**: **~17.5 hours**

**Actual Completion**:
- Start: October 19, 2025, ~6 PM
- Part 1 (exp 1-21): ~9 hours
- Part 2 (exp 22-42): ~9 hours  
- Complete: October 20, 2025, ~6 PM
- **Actual Duration**: ~18 hours

---

## 📊 Success Criteria

✅ **Complete Success** (Achieved):
- All 42 experiments completed without errors ✅
- Clear Pareto frontier identified ✅
- Optimal configurations found ✅
- Normalization validated (no paradox) ✅

⚠️ **Partial Success**:
- 35+ experiments complete (80%)
- Major trends visible
- Some groups incomplete

❌ **Failure**:
- Paradox reappears (idle times increase unexpectedly)
- System instability at certain parameters
- No clear trends or optima

---

## 🔬 Metrics Tracked

### Primary Metrics
1. **Jain's Fairness Index (JFI)**: Worker fairness (0-1, higher is better)
2. **Task Assignment Ratio (TAR)**: % of tasks assigned (target: >85%)
3. **Mean Task Wait Time**: Average time from task arrival to pickup (minutes)
4. **Mean Pickup Distance**: Average empty travel distance (km)

### Secondary Metrics
1. **Total Travel Distance**: Overall system efficiency (km)
2. **Peak Backlog**: Maximum simultaneous unassigned tasks
3. **Empty-KM Ratio**: % of travel without passenger
4. **EWMA CV**: Coefficient of variation of fairness signals
5. **Max Wait Time**: P90 or maximum task wait time

---

## 🎯 Parameter Ranges Tested

| Parameter | Values Tested | Count |
|-----------|---------------|-------|
| λ₁ (Fairness) | 0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 2.0, 5.0 | 9 |
| λ₂ (Starvation) | 0.0, 0.5, 0.8, 1.0, 1.5, 2.0 | 6 |
| λ₃ (Utility) | 0.1, 0.5, 1.0, 1.2, 1.5, 1.6, 2.0 | 7 |
| θ (Threshold) | 0.1, 0.3, 0.5, 0.6, 0.9 | 5 |

**Total Unique Combinations Tested**: 42 (strategic selection, not full factorial)  
**Coverage**: ~2.2% of all possible combinations (9 × 6 × 7 × 5 = 1,890)

---

## 🔗 Related Experiments

### Experiment 007: EWMA Gamma Sensitivity
- **Finding**: γ=0.5 is optimal
- **Impact on Exp 009**: Fixed γ=0.5 for all runs

### Experiment 008: Score Normalization Ablation
- **Finding**: Normalization essential; soft threshold minimal impact
- **Impact on Exp 009**: Enabled `normalize_scores=True` globally

### Experiment 010: Gap-Filling Parameter Exploration (Planned)
- **Purpose**: Fill gaps identified in Exp 009
- **Focus**: λ₁ values 3.0, 3.5, 4.0, 4.5 (between 2.0 and 5.0)

---

## 📝 Key Findings (Post-Completion)

### Maximum JFI Achieved
- **Best Configuration**: λ₁=5.0, λ₃=0.5 (exp_032)
- **JFI**: 0.294
- **Trade-off**: Higher wait times (~3.0 min vs. 1.9 min for Greedy)

### Optimal Practical Configuration (Hypothesis)
- **Sweet Spot**: λ₁ ∈ [0.5, 1.0], λ₃ ∈ [1.0, 1.5]
- **Balance**: Good JFI improvement with reasonable wait time increase
- **Deployment-Ready**: Groups B, E configurations

### Parameter Impact Ranking
1. **λ₁ (Fairness)**: Strongest impact on JFI
2. **λ₃ (Utility)**: Strong impact on wait times
3. **θ (Threshold)**: Minimal impact post-normalization
4. **λ₂ (Starvation)**: Moderate impact on max wait times

### Normalization Validation
✅ **Paradox Resolved**: No configurations showed excessive idle times  
✅ **Balanced Components**: Utility, fairness, starvation all contribute appropriately  
✅ **Stable System**: All 42 experiments completed successfully

---

## 🚀 How to Run (Reference)

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_009_comprehensive_parameter_sweep

# Activate virtual environment
source ../../venv/bin/activate

# Run experiment (background with logging)
nohup python run_experiment.py > experiment_log.txt 2>&1 &

# If interrupted, continue with part 2
nohup python run_experiment_part2.py > part2_log.txt 2>&1 &

# Monitor progress
tail -f experiment_log.txt

# Or run with caffeinate to prevent sleep (Mac)
caffeinate -i python run_experiment.py
```

---

## ⚠️ Lessons Learned

### What Worked
1. ✅ **Score normalization**: Essential for valid parameter exploration
2. ✅ **Strategic sampling**: 42 experiments provided comprehensive coverage
3. ✅ **Group-based design**: Systematic exploration of different hypotheses
4. ✅ **Background execution**: `nohup` + `caffeinate` for long runs

### Challenges
1. **Runtime interruption**: First run stopped at exp 21, required part 2 script
2. **Log buffering**: Despite `-u` flag, logs didn't update in real-time
3. **Missing λ₁ values**: Gap between 2.0 and 5.0 (addressed in Exp 010)

### Improvements for Future Experiments
1. Create "part 2" script proactively for very long runs
2. Test with smaller dataset first to estimate runtime
3. Plan for denser coverage in critical ranges (e.g., λ₁: 2-5)
4. Consider checkpoint/resume mechanism in main script

---

## 📊 Data Files Generated

### Aggregate Results
- `experiment_009_combined_results.csv`: All 42 experiments in one file

### Individual Summaries
- `exp_001_Greedy_Baseline_summary.json`
- `exp_002_L1_0.0_L3_0.5_summary.json`
- ...
- `exp_042_LowUtility_L2_2.0_summary.json`

### Analysis Artifacts
- `analysis.ipynb`: Comprehensive analysis notebook
- `analysis.pdf`: Exported analysis results
- `figures/`: Generated visualizations

---

## 🎓 Academic Contribution

This experiment enables us to:
1. **State definitively**: "We systematically explored 42 configurations with normalized scoring"
2. **Provide evidence**: Score normalization successfully resolved the paradox
3. **Identify optima**: Clear Pareto frontier and sweet spot configurations
4. **Guide deployment**: Production-ready parameter recommendations
5. **Establish baseline**: Foundation for future parameter space exploration

---

## 📚 References

- **Experiment 006**: First identification of worker idle time paradox
- **Experiment 007**: EWMA gamma sensitivity analysis
- **Experiment 008**: Paradox diagnosis and normalization fix
- **Experiment 009** (this): First comprehensive post-normalization sweep
- **Experiment 010** (planned): Gap-filling to complete parameter map

---

**Status**: ✅ COMPLETE  
**Data Available**: Yes (`data/experiment_009_combined_results.csv`)  
**Analysis Complete**: Yes (`analysis.ipynb`)  
**Next Experiment**: Experiment 010 (Gap-Filling)

