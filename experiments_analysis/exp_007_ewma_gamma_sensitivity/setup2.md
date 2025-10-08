# Experiment 007: EWMA Gamma Sensitivity Analysis

## 🎯 **Experiment Overview**

**Research Priority**: **CRITICAL** - Diagnose worker idle time paradox from Experiment 006
**Research Questions**: RQ2.2 (EWMA γ Sensitivity), RQ1.2 (Weight Interactions)
**Expected Duration**: 2-3 hours
**Dataset**: DiDi (15K workers, 20K tasks)

## 🔍 **Problem Statement**

### **Critical Issue Identified**
From Experiment 006, we discovered a **counter-intuitive worker idle time paradox**:

- **Greedy Strategy**: Mean Worker Idle Time ≈ 23.78 minutes (23.9% ≥ 30 min)
- **Composite Strategy**: Mean Worker Idle Time ≈ 27.25 minutes (33.3% ≥ 30 min)

**This is the OPPOSITE of expected behavior** - the fairness-aware Composite strategy should REDUCE worker idle times, not increase them.

### **Root Cause Hypothesis**
The EWMA (Exponentially Weighted Moving Average) fairness component may have a poorly calibrated smoothing factor (γ) that is:
1. **Too slow** (γ too high): Prioritizing long-term fairness over immediate efficiency
2. **Too fast** (γ too low): Creating "spiky" fairness corrections that confuse assignment logic

## 🧪 **Experimental Design**

### **Phase 1: EWMA γ Sensitivity Analysis**

**Fixed Parameters** (Sweet Spot Configuration from Exp 006):
- λ₁ (fairness_weight) = 0.5
- λ₂ (starvation_weight) = 0.8  
- λ₃ (utility_weight) = 0.8
- soft_threshold = 0.5

**Variable Parameter**:
- **EWMA γ (gamma)**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Runs per γ**: 3 (for statistical significance)
- **Total Experiments**: 15 (5 γ values × 3 runs)

### **Phase 2: Weight Interaction Analysis**

**Hypothesis**: λ₃ (utility_weight) too low relative to λ₁ (fairness_weight)

**Test Configurations**:
1. **Higher Utility Weight**: λ₁=0.5, λ₂=0.8, λ₃=[1.5, 2.0], threshold=0.5
2. **Lower Fairness Weight**: λ₁=[0.3], λ₂=0.8, λ₃=0.8, threshold=0.5
3. **Balanced Approach**: λ₁=0.4, λ₂=0.8, λ₃=1.2, threshold=0.5

**Total Additional Experiments**: 12 (4 configurations × 3 runs)

## 📊 **Key Metrics to Track**

### **Primary Metrics** (Focus on Idle Time Issue):
1. **Mean Worker Idle Time** (minutes)
2. **P95 Worker Idle Time** (95th percentile)
3. **Percentage of Workers with Idle Time ≥ 30 minutes**
4. **EWMA Coefficient of Variation** (fairness effectiveness)

### **Secondary Metrics** (Ensure No Regression):
1. **Jain's Fairness Index (JFI)**
2. **Task Assignment Ratio (TAR)**
3. **Average Task Wait Time**
4. **Average Pickup Distance**

### **Diagnostic Metrics**:
1. **EWMA Temporal Evolution** (smoothness vs responsiveness)
2. **Assignment Decision Distribution** (fairness vs utility vs starvation priorities)
3. **Worker Utilization Patterns**

## 🎯 **Success Criteria**

### **Primary Goal**: Find γ value that **minimizes worker idle time**
- Target: Mean Worker Idle Time < 20 minutes
- Target: < 15% of workers with idle time ≥ 30 minutes

### **Secondary Goal**: Maintain fairness performance
- JFI ≥ 0.85 (maintain fairness gains)
- TAR ≥ 95% (maintain assignment effectiveness)

### **Optimal Configuration**: 
Find the γ and λ combination that achieves:
- **Highest JFI** 
- **Lowest Average Task Wait Time**
- **Lowest Mean Worker Idle Time**

## 🔬 **Experimental Methodology**

### **Statistical Approach**:
- **3 runs per configuration** for statistical reliability
- **Mann-Whitney U tests** for significance testing
- **Effect size calculation** for practical significance

### **Data Collection**:
- **Enhanced temporal tracking** of EWMA evolution
- **Worker-level idle time distributions**
- **Assignment decision logging** (which component dominated each decision)

### **Analysis Framework**:
- **Correlation analysis** between γ and idle time metrics
- **Pareto frontier analysis** of fairness vs idle time trade-offs
- **Temporal pattern analysis** of EWMA responsiveness

## 📈 **Expected Outcomes**

### **Scenario 1: γ Optimization Resolves Issue**
- Find optimal γ that balances EWMA responsiveness
- Achieve < 20 min mean worker idle time while maintaining JFI > 0.85
- **Next Step**: Proceed to full RQ1/RQ3 exploration with optimized parameters

### **Scenario 2: Weight Rebalancing Required**
- γ optimization insufficient; need higher λ₃ or lower λ₁
- Find new "sweet spot" configuration with better idle time performance
- **Next Step**: Validate new configuration before broader parameter sweep

### **Scenario 3: Fundamental Issue Identified**
- Neither γ nor weight adjustments resolve the paradox
- May indicate need for composite function redesign or bug investigation
- **Next Step**: Deep dive into assignment decision logic analysis

## 🚀 **Implementation Plan**

### **Phase 1 Execution** (1.5 hours):
1. Run EWMA γ sensitivity sweep (15 experiments)
2. Generate worker idle time distributions for each γ
3. Identify optimal γ value

### **Phase 2 Execution** (1 hour):
1. Run weight interaction experiments (12 experiments)
2. Compare against Phase 1 optimal γ configuration
3. Identify best overall parameter combination

### **Analysis Phase** (30 minutes):
1. Statistical significance testing
2. Visualization of γ effects on idle time
3. Recommendation for next experimental phase

## 📋 **Deliverables**

1. **Experimental Results**: JSON files with detailed metrics
2. **Analysis Notebook**: Statistical analysis and visualization
3. **Parameter Recommendations**: Optimal γ and λ configuration
4. **Next Phase Plan**: Either proceed to RQ1/RQ3 or deeper diagnosis

## 🔗 **Connection to Research Framework**

This experiment directly addresses:
- **RQ2.2**: EWMA parameter sensitivity and optimization
- **RQ1.2**: Parameter interaction effects on fairness-efficiency trade-offs
- **Foundation for RQ8**: Understanding parameter space before PPO adaptive weighting

**Critical Path**: This experiment must be completed before proceeding with comprehensive RQ1/RQ3 exploration, as the current parameter set produces counter-intuitive results that would invalidate broader conclusions.
