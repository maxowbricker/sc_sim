# Experiment 008: Score Normalization and Threshold Ablation

## Quick Start

### Completed Experiment
✅ **Run completed**: October 19, 2025 (5.74 hours, 12/12 successful)  
📁 **Data location**: `data/exp_008_20251019_112545/`

### Analyzing Results
```bash
cd experiments_analysis/exp_008_score_normalization_ablation/
jupyter notebook results_analysis.ipynb
```

### Running a New Experiment
```bash
cd experiments_analysis/exp_008_score_normalization_ablation/
python run_experiment.py
```

Expected duration: ~6 hours for 12 experiments

## What This Experiment Does

This experiment diagnoses the **worker idle time paradox** discovered in Experiment 006, where the fairness-aware Composite strategy unexpectedly increased worker idle times from 24 to 33 minutes (+19%).

### Experimental Design

**4 Groups × 3 Replications = 12 Experiments:**

1. **Group A (Greedy)**: Baseline efficiency reference
2. **Group B (Composite Current)**: Reproduce the paradox (`normalize_scores=False`, `enable_diagnostics=True`)
3. **Group C (Normalized)**: Test if score normalization fixes it (`normalize_scores=True`, `enable_diagnostics=True`)
4. **Group D (Normalized + No Threshold)**: Test both interventions (`normalize_scores=True`, `disable_soft_threshold=True`, `enable_diagnostics=True`)

### Key Hypotheses

**H1: Mis-scaled Components**
- Fairness (0-10 range) dominates Utility (0-1 range)
- Solution: Normalize F, S, U to common [0,1] scale

**H2: Soft-Threshold Feedback Loop**
- Threshold delays create artificial task shortages
- Solution: Disable threshold (immediate assignment)

### What Gets Measured

**Primary Metrics:**
- Mean worker idle time (the paradox we're investigating)
- Task assignment ratio (ensure throughput maintained)

**Diagnostic Metrics (New!):**
- Component dominance: Which of F/S/U controls each assignment?
- Dominance ratio: How much does the dominant component overwhelm others?
- Task deferral rate: How often are tasks delayed by threshold?

**Secondary Metrics:**
- Jain's Fairness Index (ensure fairness maintained)
- Task wait times and pickup distances

## Files Created

### Core Implementation
- `metrics/diagnostic_tracker.py` - Tracks score component behavior
- `simulator/strategies/composite.py` - Enhanced with normalization & ablation flags
- `simulator/simulation.py` - Integrates diagnostic tracking
- `config.py` - Added experimental flags

### Experiment Files
- `setup.md` - Complete experimental design and methodology
- `run_experiment.py` - Automated experiment execution
- `analysis.ipynb` - Comprehensive results analysis template
- `data/` - Directory for experiment outputs

### Documentation
- `experiments_analysis/README.md` - Updated with Experiment 008
- `simulator/strategies/README.md` - Documents new flags and usage

## Expected Outcomes

### If H1 is Supported (Mis-scaled Components)
- Group C shows significant improvement over Group B
- Fairness dominance drops from >70% to ~30-40%
- Dominance ratios decrease significantly

### If H2 is Supported (Threshold Delays)
- Group D shows significant improvement over Group C
- Deferral rate drops to near-zero in Group D
- Wait times minimized

### If Both are Supported
- Progressive improvement: B > C > D
- Group D achieves efficiency close to Group A (Greedy)
- Clear path forward: implement both fixes

## Configuration Flags

The experiment uses three new flags in the Composite strategy:

```python
# config.py
STRATEGY_PARAMS = {
    "composite": {
        # ... existing params ...
        "normalize_scores": False,           # Min-max normalize F, S, U
        "disable_soft_threshold": False,     # Skip threshold check
        "enable_diagnostics": False,         # Enable detailed diagnostic tracking (opt-in)
    }
}
```

**Performance Note:** When `enable_diagnostics=True`, the system uses a "slow path" that collects detailed score component data for analysis. When disabled (default), it uses the original fast path for maximum performance.

These can be used in any simulation:

```python
config = create_composite_config(
    fairness_weight=0.5,
    starvation_weight=0.8,
    utility_weight=0.8,
    normalize_scores=True,        # ⚠️ Experimental
    disable_soft_threshold=True,  # ⚠️ Experimental
    enable_diagnostics=True       # ⚠️ Performance impact - only enable when needed
)
```

## Next Steps (Analysis)

1. **Run results_analysis.ipynb** to visualize results
2. **Check component dominance** plots - does Fairness dominate in Group B?
3. **Compare wait times** - which group(s) show improvement?
4. **Review deferral rates** - does Group D have significantly lower deferrals?
5. **Statistical tests** - are improvements significant?
6. **Document findings** based on which hypothesis is supported

## References

- **Experiment 006 Results**: `../exp_006_comparative_parameter_sweep/results.md`
- **Research Context**: The scientist's analysis in your initial message
- **Actual Idle Time Numbers**: 24-33 minutes (NOT 140 minutes as initially claimed)

## Data Files

### Completed Run (October 19, 2025)
- **Aggregate results**: `data/exp_008_20251019_112545/experiment_008_aggregate_results.csv`
- **Metadata**: `data/exp_008_20251019_112545/experiment_008_metadata.json`
- **Individual experiments**: `data/exp_008_20251019_112545/exp_001_*.json` (12 files)
- **Diagnostic data**: `data/exp_008_20251019_112545/exp_*_assignments.csv` and `exp_*_deferrals.csv` (for composite groups)

### Analysis Notebook
- **Primary analysis**: `results_analysis.ipynb` - Ready to run with completed data

