# 🧪 Spatial Crowdsourcing Experiments Analysis

**Organized Research Framework for Fairness-Efficiency Trade-offs in Spatial Crowdsourcing**

This directory contains a systematically organized collection of **completed experiments** with their data, analysis, and results for the spatial crowdsourcing research project. Each experiment is self-contained with setup, execution script, data, and analysis notebooks.

## 📁 Directory Structure

```
experiments_analysis/
├── README.md                                    # This file
├── exp_001_rq1_1_fairness_weights/              # RQ1.1: Optimal λ₁ fairness weight analysis
├── exp_002_comprehensive_parameter_sweep/       # Comprehensive λ₁,λ₂,λ₃,threshold grid search
├── exp_003_custom_parameter_sweep/              # Custom parameter combinations
├── exp_004_focused_parameter_sweep/             # Focused parameter exploration
├── exp_005_bottleneck_analysis/                 # System bottleneck identification
├── exp_006_comparative_parameter_sweep/         # 🌟 MOST COMPREHENSIVE: Enhanced metrics analysis
├── exp_007_ewma_gamma_sensitivity/              # EWMA gamma parameter optimization
├── exp_008_score_normalization_ablation/        # 🔄 IN PROGRESS: Worker idle time paradox diagnosis
├── shared_utils/                                # Shared analysis utilities
├── testing/                                    # Testing and validation scripts
├── additional/                                 # Additional experimental scripts
├── shared_data/                                # Cross-experiment analysis data
└── migration_tools/                            # Historical migration scripts (reference only)
```

## 🌟 Key Experiments (All Completed)

### **Experiment 006: Comparative Parameter Sweep** ⭐ PRIMARY ANALYSIS
- **Status**: ✅ COMPLETE with comprehensive analysis
- **Scope**: 36 experiments (6 Greedy + 30 Composite parameter combinations)
- **Dataset**: 15K workers, 20K tasks
- **Enhanced Metrics**: Supervisor's spatial fairness, temporal evolution, EWMA trends
- **Analysis**: `exp_006_comparative_parameter_sweep/analysis_comprehensive.ipynb`
- **Key Findings**: Fairness-efficiency trade-offs, worker idle time paradox
- **RQs Addressed**: RQ1, RQ2, RQ4, RQ5, RQ10-11

### **Experiment 008: Score Normalization and Threshold Ablation** 🔄 IN PROGRESS
- **Status**: 🔄 IN PROGRESS - Diagnostic experiment
- **Scope**: 12 experiments (4 groups × 3 replications)
- **Dataset**: 15K workers, 20K tasks
- **Focus**: Resolve worker idle time paradox from Experiment 006
- **Hypotheses**: (1) Mis-scaled score components, (2) Soft-threshold feedback loop
- **Experimental Groups**:
  - Group A: Greedy baseline
  - Group B: Composite current (replicate paradox)
  - Group C: Composite + normalization (test H1)
  - Group D: Composite + normalization + no threshold (test H1+H2)
- **Key Metrics**: Worker idle time, component dominance, deferral rate
- **RQs Addressed**: RQ1.2 (parameter interactions), RQ2 (optimization)

### **Experiment 001: RQ1.1 Fairness Weights**
- **Status**: ✅ COMPLETE with multiple runs
- **Focus**: Optimal λ₁ (fairness weight) for >95% TAR and high JFI
- **Data**: 11 experimental runs with different parameters
- **Analysis**: `exp_001_rq1_1_fairness_weights/analysis_honours.ipynb`

### **Experiment 002: Comprehensive Parameter Sweep**
- **Status**: ✅ COMPLETE with grid search results
- **Focus**: Systematic exploration of λ₁, λ₂, λ₃, soft_threshold combinations
- **Data**: 3 comprehensive parameter sweep result files

### **Experiment 003: Custom Parameter Sweep**
- **Status**: ✅ COMPLETE with targeted configurations
- **Focus**: Custom parameter combinations for specific research questions
- **Data**: 1 targeted experimental result file

### **Experiment 004: Focused Parameter Sweep**
- **Status**: ✅ COMPLETE with focused exploration
- **Focus**: Focused parameter exploration in promising regions
- **Data**: 2 focused parameter sweep result files

### **Experiment 005: Bottleneck Analysis**
- **Status**: ✅ COMPLETE with performance analysis
- **Focus**: System bottleneck identification and performance optimization
- **Data**: 4 bottleneck analysis log/JSON files

## 🔬 Research Questions Coverage

| RQ | Focus | Primary Experiments | Status |
|---|---|---|---|
| **RQ1** | Fairness-Efficiency Trade-offs | Exp 001, 006 | ✅ Complete |
| **RQ2** | Parameter Optimization | Exp 002, 003, 006 | ✅ Complete |
| **RQ3** | Spatial Distribution Impact | Exp 006 | ✅ Complete |
| **RQ4** | Strategy Comparison | Exp 006 | ✅ Complete |
| **RQ5** | Event-Driven Dynamics | Exp 006 | ✅ Complete |
| **RQ10** | Parameter Sensitivity | Exp 002, 003, 004 | ✅ Complete |
| **RQ11** | Baseline Comparison | Exp 006 | ✅ Complete |

## 🎯 Current Research Status

### ✅ **Completed Analysis (All Experiments)**
1. **Fairness-Efficiency Trade-offs**: Comprehensive analysis showing clear trade-offs between JFI and efficiency metrics
2. **Strategy Comparison**: Greedy vs Composite detailed comparison with statistical significance testing
3. **Parameter Effects**: Systematic exploration of λ₁, λ₂, λ₃ effects on performance
4. **Temporal Evolution**: EWMA fairness trends and wait time evolution analysis
5. **Enhanced Metrics**: Supervisor's spatial fairness and Input-Output Ratio analysis
6. **Bottleneck Identification**: Performance bottleneck analysis and optimization insights

### 🔍 **Key Research Findings**
- **Fairness Improvement**: Composite strategy achieves 9.3% JFI improvement over Greedy
- **Efficiency Trade-off**: 72% increase in wait times and pickup distances for fairness gains
- **Worker Idle Time Paradox**: Composite increases mean idle time from 24 to 33 minutes (+19%)
  - Workers idle >30min increased from 23.9% to 33.3% (+9.5 percentage points)
  - **Experiment 008 underway** to diagnose and resolve this paradox
- **Parameter Sensitivity**: λ₁ (fairness weight) has strongest impact on JFI
- **Statistical Significance**: All major findings confirmed with Mann-Whitney U tests (p < 0.001)
- **Optimal Parameter Ranges**: Sweet spot identified (λ₁=0.5, λ₂=0.8, λ₃=0.8, threshold=0.5)

### 🔄 **Current Research Status**
**Core experiments complete**, with ongoing diagnostic work:
- **Comprehensive parameter exploration** across multiple dimensions ✅
- **Statistical validation** of all major findings ✅
- **Temporal evolution analysis** of fairness and efficiency metrics ✅
- **Performance optimization insights** from bottleneck analysis ✅
- **Diagnostic investigation** of worker idle time paradox 🔄 IN PROGRESS (Experiment 008)

**Next Steps:**
- Complete Experiment 008 to identify root cause of idle time paradox
- Implement validated solutions (normalization and/or threshold redesign)
- Validate fixes in Experiment 009
- Finalize thesis contributions

## 📊 Usage Instructions

### Analyzing Results
```bash
# Primary comprehensive analysis
cd exp_006_comparative_parameter_sweep/
jupyter notebook analysis_comprehensive.ipynb

# Fairness weight analysis
cd exp_001_rq1_1_fairness_weights/
jupyter notebook analysis_honours.ipynb

# Cross-experiment analysis
# Use shared_utils/ analysis scripts for meta-analysis
```

### Re-running Experiments
```bash
# Navigate to any experiment
cd exp_XXX_experiment_name/

# Run the experiment
python run_experiment.py

# Analyze results
jupyter notebook analysis.ipynb
```

## 🔗 Integration with Main Project

- **Data Sources**: `../data/` (DiDi dataset, synthetic data)
- **Core Simulation**: `../simulator/` (event-driven simulation engine)
- **Metrics Framework**: `../metrics/` (fairness and efficiency metrics)
- **Configuration**: `../config.py` (experiment parameter management)

## 📈 Publication Readiness

This organized structure supports:
- ✅ **Reproducible Research**: All experiments can be re-run with documented parameters
- ✅ **Publication Quality**: Analysis notebooks generate publication-ready figures
- ✅ **Systematic Investigation**: Complete coverage of research questions
- ✅ **Comprehensive Documentation**: Each experiment fully documented with setup and results
- ✅ **Statistical Validation**: All findings supported by appropriate statistical tests

## 📊 Experiment Data Summary

| Experiment | Data Files | Analysis Ready | Key Metrics |
|------------|------------|----------------|-------------|
| **001** | 11 JSON files | ✅ | JFI, TAR, Wait Times |
| **002** | 3 JSON files | ✅ | Parameter Grid Results |
| **003** | 1 JSON file | ✅ | Custom Configurations |
| **004** | 2 JSON files | ✅ | Focused Exploration |
| **005** | 4 Log/JSON files | ✅ | Performance Bottlenecks |
| **006** | 162 files (temporal) | ✅ | **Complete Enhanced Metrics** |

---

**🏆 Research Framework Complete**
- All core experiments executed and analyzed
- Comprehensive fairness-efficiency trade-off analysis complete
- Statistical validation of all major findings
- Ready for publication and future research extensions