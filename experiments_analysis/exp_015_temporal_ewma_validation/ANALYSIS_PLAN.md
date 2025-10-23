# Experiment 015 Analysis Plan
## EWMA Temporal & Baseline Validation

**Status**: Experiment completed (32/33 successful, Random baseline failed)  
**Data**: 3-hour peak dataset, 4K workers / 20K tasks, 15-min expiry  
**Total Simulations**: 32 successful runs

---

## Primary Research Questions Addressed

- **RQ2.1**: EWMA vs JFI correlation
- **RQ2.2**: Gamma sensitivity (with temporal data)
- **RQ2.3**: Temporal EWMA fairness patterns ⭐ **NEW**
- **RQ2.4**: EWMA convergence speed ⭐ **NEW**
- **RQ4.2**: Random baseline comparison (FAILED - need to fix)
- **RQ4.3**: LAF baseline comparison

---

## Analysis Sections & Plots

### **Section 1: Executive Summary & Key Findings**
**Purpose**: High-level overview and context  
**Content**:
- Summary statistics table (all strategies)
- Key findings bullet points
- Research question answers (quick reference)
- Data quality notes (Random baseline failed)

**Plot 1a**: **4-Panel Overview** (2x2 grid)
- Top-left: TAR by strategy type
- Top-right: JFI by strategy type  
- Bottom-left: Mean wait time by strategy type
- Bottom-right: EWMA final mean by strategy type
- Compare: Greedy, LAF, EWMA-Only vs Best Composite

**Plot 1b**: **Fairness vs Efficiency Scatter**
- X-axis: Mean Wait Time (efficiency)
- Y-axis: JFI (fairness)
- Color: Strategy type (Greedy, LAF, EWMA-Only, Composite)
- Size: TAR (bigger = higher TAR)
- Annotate: Pareto frontier, key configurations

---

### **Section 2: Baseline Comparison Analysis**
**Purpose**: Compare Greedy, LAF, EWMA-Only baselines  
**Research Questions**: RQ4.1, RQ4.3

**Plot 2a**: **Baseline Strategy Comparison - Bar Chart**
- Grouped bars for: TAR, JFI, Gini, Wait Time, Utilization
- Strategies: Greedy, LAF, EWMA-Only
- Normalize to [0, 1] for visibility

**Plot 2b**: **Baseline Fairness Distributions** (Box plots)
- X-axis: Strategy (Greedy, LAF, EWMA-Only)
- Y-axis: Per-worker task count distribution (P10, P25, P50, P75, P90)
- Show: Distribution width as fairness indicator

**Key Finding Box**: Which baseline is best?
- Greedy: Best efficiency, worst fairness
- LAF: Best JFI, poor efficiency?
- EWMA-Only: Balance between the two?

---

### **Section 3: EWMA-JFI Correlation Analysis** ⭐
**Purpose**: Validate EWMA as alternative fairness metric  
**Research Question**: RQ2.1

**Plot 3a**: **EWMA vs JFI Correlation**
- X-axis: Final JFI
- Y-axis: Final EWMA mean
- Color: Strategy type
- Add: Correlation coefficient (r²)
- Trend line with confidence interval

**Plot 3b**: **EWMA CV vs Gini Coefficient**
- X-axis: Gini Coefficient
- Y-axis: EWMA CV (coefficient of variation)
- Color: Strategy type
- Show: Alternative fairness metric correlation

**Key Finding Box**: 
- Correlation strength: Strong/Moderate/Weak?
- Does EWMA capture different fairness aspects than JFI?
- Which metric is more sensitive to fairness changes?

---

### **Section 4: Temporal EWMA Evolution** ⭐⭐ **STAR SECTION**
**Purpose**: Analyze EWMA convergence patterns over time  
**Research Questions**: RQ2.3, RQ2.4

**Plot 4a**: **EWMA Evolution - Baseline Comparison**
- X-axis: Completed Tasks (0 → 20K)
- Y-axis: EWMA Mean
- Lines: Greedy, LAF, EWMA-Only
- Shaded regions: ±1 std deviation
- Show: Convergence speed and stability

**Plot 4b**: **EWMA Evolution - Best vs Worst Fairness**
- X-axis: Completed Tasks
- Y-axis: EWMA Mean
- Lines: Best JFI composite, Worst JFI composite, Greedy
- Show: Fairness divergence over time

**Plot 4c**: **EWMA Distribution Width Over Time**
- X-axis: Completed Tasks
- Y-axis: EWMA Standard Deviation OR (P90 - P10)
- Lines: Different strategy types
- Show: Fairness inequality evolution

**Plot 4d**: **EWMA Convergence Speed Comparison**
- Calculate: Time to 90% of final EWMA value
- Bar chart: Convergence time by strategy
- Show: Which approaches reach fairness equilibrium fastest?

**Key Finding Box**:
- Convergence patterns: Linear, exponential, step-wise?
- Time to equilibrium: Early (< 5K tasks) vs late (> 15K)?
- Stability: Which strategies have most stable EWMA over time?

---

### **Section 5: Pareto Frontier Analysis**
**Purpose**: Map fairness-efficiency trade-off space  
**Research Question**: RQ1.3, RQ1.4

**Plot 5a**: **JFI Heatmap (λ₁ × λ₃)**
- X-axis: λ₃ (Utility) [0.5, 1.0, 1.5, 2.0, 2.5]
- Y-axis: λ₁ (Fairness) [2.5, 3.0, 3.5, 4.0, 4.5]
- Color: JFI value
- Annotate: Best configuration

**Plot 5b**: **Wait Time Heatmap (λ₁ × λ₃)**
- Same structure as Plot 5a
- Color: Mean Wait Time
- Show: Efficiency surface

**Plot 5c**: **Pareto Frontier with EWMA**
- X-axis: Mean Wait Time
- Y-axis: JFI
- Color: Final EWMA Mean (3rd dimension)
- Mark: Pareto-optimal points
- Show: Does EWMA add insight beyond JFI?

**Key Finding Box**:
- Pareto-optimal configurations
- Sweet spot: Balanced fairness + efficiency
- Comparison to Exp 013/014 results

---

### **Section 6: Gamma Sensitivity Analysis**
**Purpose**: Re-validate gamma with temporal data  
**Research Question**: RQ2.2

**Plot 6a**: **Gamma Impact on Final Metrics**
- X-axis: Gamma [0.1, 0.3, 0.5, 0.7, 0.9]
- Y-axis (left): JFI
- Y-axis (right): Mean Wait Time
- Fixed config: Balanced (λ₁=3.5, λ₃=1.0)
- Show: Minimal impact (expected)

**Plot 6b**: **Gamma Impact on EWMA Temporal Patterns** ⭐
- X-axis: Completed Tasks
- Y-axis: EWMA Mean
- Lines: Different gamma values
- Show: Does gamma affect convergence speed?

**Plot 6c**: **Gamma Impact on EWMA Stability**
- X-axis: Gamma
- Y-axis: EWMA Std (averaged over time)
- Show: Does gamma affect fairness volatility?

**Key Finding Box**:
- Gamma robustness confirmed?
- Optimal gamma: 0.5 still best?
- Does gamma affect convergence or just final values?

---

### **Section 7: Weight Sensitivity Analysis**
**Purpose**: Understand λ₁ and λ₃ impact  
**Research Questions**: RQ1.1, RQ1.2

**Plot 7a**: **λ₁ (Fairness) Marginal Effect**
- X-axis: λ₁ [2.5 → 4.5]
- Y-axis: JFI (averaged across λ₃ values)
- Error bars: Standard deviation
- Show: Diminishing returns at high λ₁?

**Plot 7b**: **λ₃ (Utility) Marginal Effect**
- X-axis: λ₃ [0.5 → 2.5]
- Y-axis: Mean Wait Time (averaged across λ₁ values)
- Show: Linear or non-linear relationship?

**Plot 7c**: **λ₁/λ₃ Ratio Analysis**
- Calculate: Fairness/Utility ratio for each run
- X-axis: λ₁/λ₃ ratio
- Y-axis: JFI / Wait Time ratio (normalized performance)
- Show: Optimal balance ratio

---

### **Section 8: Composite vs Baselines Performance**
**Purpose**: Demonstrate composite value  
**Research Questions**: RQ4.1, RQ4.3

**Plot 8a**: **Fairness Improvement Over Baselines**
- Bar chart: JFI improvement (%)
- Reference lines: Greedy (0%), LAF baseline
- Show: Best composite, Median composite, Worst composite

**Plot 8b**: **Efficiency Cost Analysis**
- X-axis: JFI improvement over Greedy (%)
- Y-axis: Wait time increase over Greedy (%)
- Points: All composite runs
- Quadrants: Win-win (top-left), Trade-off (top-right), etc.

**Plot 8c**: **EWMA-Only vs Composite Comparison**
- Radar chart or spider plot
- Axes: TAR, JFI, Gini, Wait Time, Utilization, EWMA
- Lines: EWMA-Only baseline, Best Composite, Balanced Composite

---

### **Section 9: Temporal Pattern Deep Dive** ⭐⭐
**Purpose**: Extract temporal insights unique to Exp 015  
**Research Questions**: RQ2.3, RQ2.4

**Plot 9a**: **Early vs Late Fairness**
- X-axis: Strategy
- Y-axis: JFI or EWMA Mean
- Grouped bars: Early (0-5K tasks), Mid (5K-15K), Late (15K-20K)
- Show: Fairness trajectory differences

**Plot 9b**: **EWMA Percentile Evolution**
- X-axis: Completed Tasks
- Y-axis: EWMA value
- Lines: P10, P50, P90 for selected strategy
- Show: Distribution shape changes over time

**Plot 9c**: **Fairness Volatility Analysis**
- Calculate: Standard deviation of EWMA mean across temporal snapshots
- Bar chart: Volatility by strategy type
- Show: Which strategies have most stable fairness?

---

### **Section 10: Summary & Deployment Recommendations**
**Purpose**: Synthesize findings into actionable insights

**Table 10a**: **Strategy Performance Summary**
| Strategy | TAR | JFI | Wait Time | EWMA Final | Convergence Time | Recommended Use Case |
|----------|-----|-----|-----------|------------|------------------|----------------------|
| Greedy   | ... | ... | ...       | ...        | ...              | ...                  |
| LAF      | ... | ... | ...       | ...        | ...              | ...                  |
| EWMA-Only| ... | ... | ...       | ...        | ...              | ...                  |
| Balanced | ... | ... | ...       | ...        | ...              | ...                  |
| High Fairness | ... | ... | ... | ...        | ...              | ...                  |

**Table 10b**: **Research Questions Answered**
| RQ | Question | Answer | Evidence |
|----|----------|--------|----------|
| RQ2.1 | EWMA-JFI correlation | ... | Plot 3a: r²=... |
| RQ2.2 | Optimal gamma | ... | Section 6 |
| RQ2.3 | Temporal patterns | ... | Section 4 |
| RQ2.4 | Convergence speed | ... | Plot 4d |
| ... | ... | ... | ... |

**Deployment Decision Guide**:
1. **High Fairness Priority** (e.g., regulated market):
   - Configuration: λ₁=X, λ₃=Y, γ=0.5
   - Expected: JFI=X%, Wait Time=Y min
   
2. **Balanced Operation** (e.g., general platform):
   - Configuration: λ₁=3.5, λ₃=1.0, γ=0.5
   - Expected: JFI=X%, Wait Time=Y min

3. **Efficiency Priority** (e.g., peak demand):
   - Configuration: λ₁=X, λ₃=Y, γ=0.5
   - Expected: JFI=X%, Wait Time=Y min

**Key Insights**:
- EWMA validity as fairness metric
- Temporal fairness patterns
- Convergence characteristics
- Baseline comparisons

---

## Implementation Notes

### Data Loading
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load aggregate results
df = pd.read_csv('data/experiment_015_aggregate_results.csv')

# Load temporal data from individual JSON files
import json
import glob

temporal_data = {}
for filepath in glob.glob('data/exp_015_*/exp_*_summary.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
        exp_name = data['experiment_name']
        if 'ewma_temporal_history' in data:
            temporal_data[exp_name] = data['ewma_temporal_history']
```

### Key Temporal Analysis Functions
```python
def calculate_convergence_time(temporal_history, threshold=0.9):
    """Calculate time to reach threshold% of final EWMA value."""
    final_ewma = temporal_history[-1]['ewma_mean']
    target = threshold * final_ewma
    for snapshot in temporal_history:
        if snapshot['ewma_mean'] >= target:
            return snapshot['completed_tasks']
    return None

def calculate_temporal_volatility(temporal_history):
    """Calculate EWMA volatility over time."""
    ewma_values = [s['ewma_mean'] for s in temporal_history]
    return np.std(ewma_values)
```

---

## Expected Insights

1. **EWMA Validation** (RQ2.1):
   - Strong correlation with JFI expected (r² > 0.7)
   - EWMA may capture temporal dynamics better

2. **Temporal Patterns** (RQ2.3, RQ2.4):
   - Fairness-aware strategies converge faster
   - Greedy may show degrading fairness over time
   - Composite strategies maintain stable fairness

3. **Baseline Comparison** (RQ4.3):
   - LAF expected to have high JFI but poor efficiency
   - EWMA-Only should balance both
   - Composite should outperform both

4. **Gamma Robustness** (RQ2.2):
   - Minimal impact on final metrics (confirmed from Exp 014)
   - May affect convergence speed

---

## Total Plots: 24 plots across 10 sections

**Section Distribution**:
- Section 1: 2 plots (overview)
- Section 2: 2 plots (baselines)
- Section 3: 2 plots (correlation)
- Section 4: 4 plots (temporal EWMA) ⭐
- Section 5: 3 plots (Pareto)
- Section 6: 3 plots (gamma)
- Section 7: 3 plots (weights)
- Section 8: 3 plots (composite vs baselines)
- Section 9: 3 plots (temporal deep dive) ⭐
- Section 10: Summary tables (no plots)

**Estimated Completion Time**: 3-4 hours

