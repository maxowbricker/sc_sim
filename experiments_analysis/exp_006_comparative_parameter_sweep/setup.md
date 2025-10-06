# Experiment 004: Comprehensive Comparative Parameter Sweep

## Objective
**Primary Goal**: Conduct the most comprehensive comparison between Greedy and Composite strategies across multiple parameter configurations and advanced metrics.

**Research Questions Addressed**: 
- RQ1.1-1.4: Complete fairness-efficiency trade-off analysis
- RQ4.1: Definitive strategy comparison with statistical rigor
- RQ5.1-5.4: Temporal evolution and event-driven simulation validation
- RQ2.1: Initial EWMA fairness metric validation

## Hypothesis
Composite strategy will demonstrate significant fairness improvements across multiple parameter configurations, with identifiable sweet spots that balance fairness and efficiency effectively.

## Experimental Design

### Configuration
- **Dataset**: DiDi (15,000 workers, 20,000 tasks) - **Largest scale**
- **Strategies**: 
  - Greedy (6 runs for statistical significance)
  - Composite (21 parameter combinations)

### Parameter Matrix
| Parameter | Values | Count |
|-----------|--------|-------|
| fairness_weight (λ₁) | 0.3, 0.5, 0.7 | 3 |
| starvation_weight (λ₂) | 0.8, 1.0, 1.2 | 3 |
| utility_weight (λ₃) | 0.8, 1.0, 1.2 | 3 |
| soft_threshold | 0.2, 0.5 | 2 |

**Total Configurations**: 6 Greedy + 21 Composite = 27 experiments

### Enhanced Metrics Collection
- **Core**: JFI, TAR (%), Wait Time, Pickup Distance
- **Advanced**: Worker idle times, EWMA evolution, temporal trends
- **Temporal**: Complete evolution data for all metrics
- **Spatial**: Worker fairness distributions, task completion patterns

## Expected Outcomes
- **Definitive Strategy Validation**: Clear statistical evidence
- **Sweet Spot Identification**: Optimal parameter configuration
- **Trade-off Quantification**: Precise fairness-efficiency relationships
- **Temporal Insights**: Event-driven simulation advantages

## Data Structure
```
data/
├── raw_results/
│   ├── greedy_runs/              # 6 baseline runs
│   └── composite_configs/        # 21 parameter combinations
├── processed/
│   ├── experiment_summary.csv
│   ├── statistical_analysis.csv
│   └── sweet_spot_analysis.csv
├── temporal/
│   ├── wait_time_evolution/      # Time series data
│   ├── fairness_trends/
│   └── worker_data/              # Individual worker tracking
└── comparative_analysis/
    ├── strategy_comparison.json
    └── pareto_frontier.csv
```

## Success Criteria
1. **Statistical Significance**: p < 0.001 for key comparisons
2. **Comprehensive Coverage**: All RQ1 sub-questions addressed
3. **Sweet Spot Found**: Optimal λ configuration identified
4. **Temporal Validation**: Event-driven benefits demonstrated
5. **Research-Grade Results**: Publication-quality analysis

## Special Features
- **Most comprehensive experiment** in the research project
- **Enhanced temporal data collection** (162 files)
- **Advanced fairness metrics** (EWMA, worker idle times)
- **Statistical rigor** (Mann-Whitney U tests, effect sizes)
