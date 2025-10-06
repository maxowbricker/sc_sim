# Experiment 001: RQ1.1 Fairness Weights Analysis

## Objective
**Primary Goal**: Determine the optimal λ₁ (fairness weight) range for maximizing JFI while maintaining >95% task assignment ratio.

**Research Questions Addressed**: 
- RQ1.1: What is the optimal λ₁ (fairness weight) range for maximizing JFI while maintaining >95% task assignment ratio?

## Hypothesis
Higher fairness weights will improve JFI but may reduce task assignment efficiency. There exists an optimal range that balances both objectives.

## Experimental Design

### Configuration
- **Dataset**: DiDi (5K workers, 10K tasks for quick iteration)
- **Strategy**: Composite scoring
- **Fixed Parameters**: 
  - λ₂ (starvation): 1.0
  - λ₃ (utility): 1.0
  - soft_threshold: 1.0

### Parameter Matrix
| Parameter | Values | Count |
|-----------|--------|-------|
| fairness_weight (λ₁) | 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 | 7 |

**Total Configurations**: 7 experiments

### Metrics to Collect
- **Primary**: JFI, TAR (%), Average Wait Time
- **Secondary**: Pickup Distance, Total Travel Distance
- **Target**: JFI maximization with TAR >95%

## Expected Outcomes
- **JFI**: Monotonic increase with λ₁
- **TAR**: Potential decrease at very high λ₁ values
- **Optimal Range**: λ₁ ∈ [1.0, 2.0] for balanced performance

## Data Structure
```
data/
├── raw_results/
│   └── rq1_1_results_[timestamp].json
├── processed/
│   └── fairness_weight_analysis.csv
└── temporal/
    └── parameter_evolution/
```

## Success Criteria
1. Clear relationship between λ₁ and JFI established
2. TAR >95% threshold identified  
3. Optimal λ₁ range for future experiments determined
