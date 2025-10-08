# Experiment 008: Score Normalization and Threshold Ablation

## Status
🔄 **In Progress** - Diagnostic experiment for worker idle time paradox

## Overview
This experiment systematically tests hypotheses about the worker idle time paradox observed in Experiment 006, where the fairness-aware Composite strategy unexpectedly increased worker idle times compared to the Greedy baseline (~33 min vs ~24 min mean idle time, with a 9.5 percentage point increase in workers idle >30min).

## Research Context

### The Worker Idle Time Paradox
**Observation from Experiment 006:**
- **Greedy Strategy**: Mean worker idle time = 23.78-27.82 minutes
- **Composite Strategy**: Mean worker idle time = 27.25-33.09 minutes
- **Increase**: ~19% (+5-6 minutes)
- **Workers idle >30min**: Increased from 23.9% to 33.3% (+9.5 percentage points)

**Key Question**: Why does adding fairness increase idle times?

### Research Hypotheses

#### Hypothesis 1: Mis-scaled Composite Score Components
The three score components (Fairness, Starvation, Utility) operate on vastly different numerical scales:
- **Fairness (EWMA)**: Based on idle time in hours, can grow large (e.g., 0-10+ range)
- **Utility**: Inverse distance, typically small (e.g., 0-1 range)
- **Starvation**: Logarithmic task age, moderate values (e.g., 0-8 range)

**Predicted Mechanism**: Without normalization, the Fairness term with large values dominates the composite score, making the system insensitive to spatial efficiency (Utility). This forces spatially inefficient assignments that increase travel times and cascade into higher system-wide idle times.

**Test**: Compare normalized vs. non-normalized scoring to isolate the effect of component scale mismatch.

#### Hypothesis 2: Soft-Threshold Feedback Loop
The soft threshold mechanism (default threshold=0.5) delays task assignments when no candidate exceeds the threshold, creating an "artificial task drought."

**Predicted Mechanism**:
1. New task arrives when workers have low idle time (low Fairness scores)
2. Even nearby workers score below threshold due to fairness component
3. Task is deferred, creating artificial scarcity
4. All workers become idle while waiting, their Fairness scores skyrocket
5. When task is reconsidered, system is in chaotic state with inflated scores
6. Meanwhile, other tasks are also deferred, cascading the problem

**Test**: Disable the soft threshold (immediate assignment) to test if deferral mechanism causes the paradox.

## Experimental Design

### Experimental Groups

| Group | Strategy | normalize_scores | disable_soft_threshold | Purpose |
|-------|----------|-----------------|----------------------|---------|
| **A** | Greedy | N/A | N/A | Efficiency baseline |
| **B** | Composite | False | False | Replicate paradox (control) |
| **C** | Composite | True | False | Test Hypothesis 1 (normalization only) |
| **D** | Composite | True | True | Test both hypotheses (normalization + no threshold) |

### Parameters (Sweet Spot from Experiment 006)
- `fairness_weight` = 0.5
- `starvation_weight` = 0.8
- `utility_weight` = 0.8
- `soft_threshold` = 0.5
- `gamma` (EWMA) = 0.5
- `k` (nearest neighbors) = 15

### Dataset
- **Workers**: 15,000
- **Tasks**: 20,000
- **Source**: DiDi dataset
- **Replications**: 3 per group (12 total experiments)

## Key Metrics

### Primary Validation Metrics
1. **Mean Worker Idle Time** (minutes)
   - *Prediction*: B >> C > D ≈ A
   - *Target*: If D significantly reduces idle time vs. B, hypotheses are supported

2. **Task Assignment Ratio** (%)
   - *Monitoring*: Ensure intervention doesn't break system throughput

### Diagnostic Metrics (New)
3. **Component Dominance Distribution**
   - Percentage of assignments where each component (F, S, U) has the highest weighted value
   - *Prediction*: In Group B, Fairness dominates >70% of assignments
   - *Prediction*: In Groups C & D, dominance is more balanced (30-40% each)

4. **Dominance Ratio**
   - Ratio of dominant component to sum of others
   - *Prediction*: Group B has high ratios (>2.0), Groups C & D have lower ratios (<1.5)

5. **Task Deferral Rate** (%)
   - Percentage of tasks initially deferred by soft threshold
   - *Prediction*: Groups B & C have high deferral (>30%), Group D has 0%

6. **Score Component Statistics**
   - Raw values of F, S, U components across all assignments
   - Reveals scale mismatches in Groups A & B

### Secondary Metrics
7. **Jain's Fairness Index (JFI)**
   - Ensure fairness is maintained
   - *Monitoring*: Groups C & D should maintain JFI comparable to Group B

8. **Mean Task Wait Time** (minutes)
   - Should be lowest in Group D (immediate assignment)

9. **Mean Pickup Distance** (km)
   - Spatial efficiency indicator

## Expected Outcomes

### If Hypothesis 1 is Correct (Mis-scaled Components)
- Group C shows **significant improvement** over Group B
- Idle time drops by 30-50%
- Dominance distribution becomes more balanced
- Dominance ratios decrease

### If Hypothesis 2 is Correct (Soft-Threshold Feedback)
- Group D shows **significant improvement** over Groups B & C
- Idle time approaches Group A (Greedy baseline)
- Deferral rate is the key differentiator
- Wait times are minimized

### If Both Hypotheses are Correct
- Group C shows **partial improvement**
- Group D shows **full resolution** of the paradox
- This indicates compound effects requiring both fixes

### If Neither Hypothesis is Correct
- No significant differences between B, C, and D
- Requires investigation of other mechanisms (e.g., EWMA calculation, starvation metric interaction)

## Data Collection

### Outputs per Experiment
1. **Standard Metrics JSON** (`exp_XXX_summary.json`)
   - All existing metrics from Experiment 006
   
2. **Diagnostic Data** (new)
   - Assignment records CSV (via `diagnostic_tracker.to_dataframe('assignments')`)
   - Deferral records CSV (via `diagnostic_tracker.to_dataframe('deferrals')`)
   - Diagnostic summary JSON (via `diagnostic_tracker.get_summary_stats()`)

### Analysis Notebook
- `analysis.ipynb`: Comprehensive analysis of all metrics
- Visualizations:
  - Idle time comparison across groups
  - Component dominance distributions
  - Dominance ratio box plots
  - Deferral rate comparison
  - Score component raw value distributions

## Implementation Notes

### New Code Components
1. **DiagnosticTracker** (`metrics/diagnostic_tracker.py`)
   - Tracks per-assignment score component values
   - Records deferral events
   - Exports to DataFrame for Jupyter analysis

2. **Modified Composite Strategy** (`simulator/strategies/composite.py`)
   - Added `normalize_scores` parameter
   - Added `disable_soft_threshold` parameter
   - Integrated diagnostic tracking

3. **Config Flags** (`config.py`)
   - `normalize_scores = False` (default)
   - `disable_soft_threshold = False` (default)

### Running the Experiment
```bash
cd experiments_analysis/exp_008_score_normalization_ablation/
python run_experiment.py
```

Expected duration: ~3-4 hours (12 experiments × 15-20 minutes each)

## Success Criteria

This experiment is successful if:
1. ✅ All 12 experiments complete without errors
2. ✅ Diagnostic data is collected for all Composite experiments (B, C, D)
3. ✅ We can definitively determine which hypothesis (or both) explains the paradox
4. ✅ We identify a path forward to resolve the idle time paradox

## Next Steps

Based on outcomes:
- **If H1 confirmed**: Implement normalization as default in Composite strategy
- **If H2 confirmed**: Redesign threshold mechanism with bounded deferrals
- **If both confirmed**: Implement both fixes and validate in Experiment 009
- **If neither confirmed**: Investigate alternative hypotheses (EWMA formulation, starvation metric interaction)

## References
- **Experiment 006**: Baseline comparative analysis
- **Research Framework**: `../../Research_Questions_Framework.md`
- **Composite Strategy**: `../../simulator/strategies/composite.py`
- **DiagnosticTracker**: `../../metrics/diagnostic_tracker.py`

