# Experiment 011: Scalability Analysis

**Status**: IN PROGRESS  
**Started**: October 21, 2025  
**Duration**: ~3-4 hours (8 experiments × ~25 minutes)

---

## Research Context

**Primary Question**: How does system performance and fairness scale with worker population size?

**Motivation**: 
- Validate that composite strategy performance holds across different market sizes
- Understand computational and fairness trade-offs at different scales
- Determine if Pareto-efficient configurations remain optimal at different worker densities

---

## Experimental Design

### Fixed Parameters (All Experiments)
- **Dataset**: DiDi Quarter (20,000 tasks)
- **normalize_scores**: True
- **gamma**: 0.5
- **enable_diagnostics**: False (fast path)

### Variable Parameters

#### Worker Counts (4 levels)
1. **2,500 workers** - Small market (8:1 task-to-worker ratio)
2. **5,000 workers** - Medium market (4:1 ratio)
3. **10,000 workers** - Large market (2:1 ratio)
4. **15,000 workers** - Full market (1.33:1 ratio)

#### Parameter Configurations (2 Pareto-efficient setups)

**Config A: Balanced Fairness-Efficiency**
- λ₁ (Fairness): 2.0
- λ₂ (Starvation): 0.8
- λ₃ (Utility): 1.0
- θ (Threshold): 0.5
- *Rationale*: Best balance from Exp 009/010 analysis

**Config B: High Fairness Focus**
- λ₁ (Fairness): 3.5
- λ₂ (Starvation): 0.8
- λ₃ (Utility): 0.5
- θ (Threshold): 0.5
- *Rationale*: Higher fairness weight for equity-focused scenarios

---

## Experimental Matrix

| Exp ID | Workers | Config | λ₁ | λ₂ | λ₃ | θ | Expected Runtime |
|--------|---------|--------|-----|-----|-----|-----|------------------|
| 1 | 2,500 | A | 2.0 | 0.8 | 1.0 | 0.5 | ~15 min |
| 2 | 2,500 | B | 3.5 | 0.8 | 0.5 | 0.5 | ~15 min |
| 3 | 5,000 | A | 2.0 | 0.8 | 1.0 | 0.5 | ~20 min |
| 4 | 5,000 | B | 3.5 | 0.8 | 0.5 | 0.5 | ~20 min |
| 5 | 10,000 | A | 2.0 | 0.8 | 1.0 | 0.5 | ~23 min |
| 6 | 10,000 | B | 3.5 | 0.8 | 0.5 | 0.5 | ~23 min |
| 7 | 15,000 | A | 2.0 | 0.8 | 1.0 | 0.5 | ~25 min |
| 8 | 15,000 | B | 3.5 | 0.8 | 0.5 | 0.5 | ~25 min |

**Total Experiments**: 8  
**Expected Total Duration**: ~3-3.5 hours

---

## Research Questions

### RQ1: Scalability of Performance
- Does TAR (Task Assignment Ratio) improve or degrade with more workers?
- How does mean wait time scale with worker count?
- Does computational time scale linearly?

### RQ2: Fairness at Scale
- Does Gini coefficient improve with more workers (more assignment options)?
- How does % workers with zero tasks change?
- Does worker utilization become more uniform at higher scales?

### RQ3: Configuration Robustness
- Do Config A and B maintain their relative performance across all scales?
- Does high fairness weight (Config B) become more or less costly at scale?

### RQ4: Resource Utilization
- How does worker utilization rate change with worker count?
- Is there a "sweet spot" worker count for efficiency?

---

## Hypotheses

**H1**: TAR should increase with worker count (more options = higher assignment success)

**H2**: Mean wait time should decrease with worker count (more available workers nearby)

**H3**: Gini coefficient should decrease with worker count (more workers = easier to distribute fairly)

**H4**: Worker utilization should decrease with worker count (more workers competing for same tasks)

**H5**: Computational time should scale sub-linearly (event count dominated by tasks, not workers)

**H6**: Config B (high fairness) will show larger Gini improvement at higher worker counts

---

## Success Criteria

✅ **All 8 experiments complete successfully**
✅ **All 78+ metrics collected for each run**
✅ **Clear scaling trends observable in key metrics**
✅ **Configuration robustness validated across scales**

---

## Data Collection

### Per-Experiment Metrics (78+ total)
All metrics from v2.0 data dictionary including:
- Core: completed_tasks, TAR, JFI
- Task wait times: mean, std, p90, p95, max, CV
- Worker metrics: idle time, task distribution (Gini!), utilization
- System: pickup distance, deferrals, assignment timing

### Derived Scaling Metrics
- Performance per worker
- Efficiency per unit resource
- Fairness improvement rate

---

## Expected Insights

1. **Optimal Market Density**: Identify task-to-worker ratio for best fairness-efficiency balance
2. **Scalability Limits**: Determine if composite strategy scales to larger markets
3. **Configuration Sensitivity**: Understand if parameter tuning needs adjustment at different scales
4. **Resource Planning**: Guide deployment recommendations for different market sizes

---

## Analysis Plan

### Primary Plots
1. **TAR vs Worker Count** (by config)
2. **Gini Coefficient vs Worker Count** (scalability of fairness)
3. **Mean Wait Time vs Worker Count** (efficiency scaling)
4. **Worker Utilization vs Worker Count** (resource efficiency)
5. **Computational Time vs Worker Count** (performance scaling)

### Secondary Analysis
- Pareto frontiers at each scale
- Config A vs B performance delta across scales
- Per-worker efficiency metrics
- Starvation rates by scale

---

## Notes

- Using new v2.0 metrics implementation (all 23 distribution metrics)
- First experiment to systematically vary worker count
- Will inform deployment recommendations for real-world systems
- Complements Experiments 009/010 parameter studies

---

**Experiment executed on**: macOS, Python 3.13, event-driven simulator v2.0



