# 🔍 Experiment 007: EWMA Gamma Sensitivity - Discussion & Plan

## 🎯 **Experimental Rationale**

Based on your comprehensive analysis from Experiment 006, we've identified a **critical counter-intuitive finding** that must be resolved before proceeding with broader research:

### **The Worker Idle Time Paradox**
- **Expected**: Fairness-aware Composite strategy should REDUCE worker idle times
- **Observed**: Composite strategy INCREASES worker idle times by ~15% vs Greedy
- **Impact**: This contradicts the fundamental premise of EWMA fairness optimization

This is not just a parameter tuning issue - it's a **fundamental validity concern** that could invalidate all subsequent research if not properly diagnosed and resolved.

## 🧪 **Proposed Experimental Design**

### **Phase 1: EWMA γ Sensitivity (Primary Focus)**
**Hypothesis**: The EWMA smoothing factor (γ) is poorly calibrated, causing either:
1. **Over-smoothing** (γ too high): Long-term fairness memory creates delayed reactions
2. **Under-smoothing** (γ too low): Rapid fairness changes create assignment instability

**Test Design**:
- **Fixed Parameters**: Sweet Spot from Exp 006 (λ₁=0.5, λ₂=0.8, λ₃=0.8, threshold=0.5)
- **Variable**: EWMA γ ∈ [0.1, 0.3, 0.5, 0.7, 0.9]
- **Runs**: 3 per γ value (15 total experiments)
- **Focus Metric**: Mean Worker Idle Time

### **Phase 2: Weight Interaction Analysis (Secondary)**
**Hypothesis**: λ₃ (utility_weight) is too low relative to λ₁ (fairness_weight), causing over-prioritization of global fairness at expense of local efficiency.

**Test Configurations**:
1. **Higher Utility**: λ₃ = [1.5, 2.0] (vs current 0.8)
2. **Lower Fairness**: λ₁ = [0.3] (vs current 0.5)  
3. **Balanced**: λ₁=0.4, λ₃=1.2

## 📊 **Key Questions for Discussion**

### **1. Experimental Scope**
- **Q**: Is 15 + 12 = 27 experiments sufficient, or should we expand the γ range?
- **Consideration**: Each experiment takes ~6-8 minutes, so 27 experiments ≈ 3 hours total
- **Alternative**: Could start with quick γ sweep [0.1, 0.5, 0.9] to identify promising regions

### **2. Success Criteria**
- **Primary Goal**: Mean Worker Idle Time < 20 minutes (vs current ~27 minutes)
- **Secondary Goal**: Maintain JFI > 0.85 and TAR > 95%
- **Q**: Are these thresholds appropriate, or should we be more/less aggressive?

### **3. Diagnostic Depth**
- **Current Plan**: Focus on aggregate idle time metrics
- **Q**: Should we also track individual worker idle time patterns to understand the mechanism?
- **Q**: Should we log which assignment component (fairness/utility/starvation) dominated each decision?

### **4. Fallback Strategy**
If γ optimization doesn't resolve the paradox:
- **Option A**: Deeper parameter exploration (expand λ ranges)
- **Option B**: Composite function redesign investigation
- **Option C**: Bug hunting in EWMA calculation or assignment logic
- **Q**: Which fallback do you prefer, and what would trigger it?

### **5. Integration with Broader Research**
- **If Successful**: Proceed immediately to comprehensive RQ1/RQ3 exploration with optimized parameters
- **If Partially Successful**: Additional targeted optimization before broader sweep
- **Q**: Should we plan the next experiment (RQ1/RQ3) now, or wait for these results?

## 🔬 **Technical Implementation Questions**

### **1. EWMA γ Parameter**
- **Q**: Is the γ parameter currently configurable in your simulation, or does it need to be added?
- **Current**: Most systems use γ ≈ 0.1-0.3 for responsiveness, 0.7-0.9 for stability

### **2. Idle Time Calculation**
- **Q**: How is "worker idle time" currently measured in your system?
- **Definition Needed**: Time between task completions? Time waiting for assignment? Total inactive time?

### **3. Temporal Data Collection**
- **Q**: Can we capture EWMA evolution over time to visualize smoothing effects?
- **Value**: This would help understand whether γ affects responsiveness vs stability

### **4. Assignment Decision Logging**
- **Q**: Can we log which component (fairness/utility/starvation) had the highest score for each assignment?
- **Value**: This would reveal if fairness is over-dominating utility decisions

## 🎯 **Recommendations & Next Steps**

### **Immediate Actions**:
1. **Validate Implementation**: Confirm γ parameter is configurable and idle time metrics are accurate
2. **Quick Pilot**: Run 3-5 experiments with γ=[0.1, 0.5, 0.9] to verify approach
3. **Full Experiment**: Execute complete Phase 1 + Phase 2 if pilot shows promise

### **Decision Points**:
- **After Phase 1**: If optimal γ found → proceed to Phase 2
- **After Phase 2**: If paradox resolved → plan RQ1/RQ3 comprehensive sweep
- **If Unresolved**: Deep dive into composite function mechanics

### **Success Metrics**:
- **Immediate**: Find γ that reduces idle time to < 22 minutes
- **Full Success**: Find configuration with idle time < 20 minutes AND JFI > 0.85
- **Research Impact**: Validate that fairness-aware assignment can improve BOTH fairness AND efficiency

## 🤔 **Discussion Points**

1. **Do you agree this paradox must be resolved before broader research?**
2. **Are the proposed γ values and weight adjustments reasonable?**
3. **Should we add any additional diagnostic metrics or logging?**
4. **What's your preferred fallback if γ optimization doesn't work?**
5. **Any concerns about the experimental design or timeline?**

This experiment is **critical path** - all subsequent RQ1/RQ3/RQ8 research depends on having a validated parameter set that doesn't produce counter-intuitive results.
