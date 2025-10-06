# Experiment 005: Custom Parameter Sweep

## Objective
**Primary Goal**: Targeted parameter exploration with exactly ~30 experiments using 15,000 tasks for manageable experiment size and realistic timing.

**Research Questions Addressed**: 
- RQ1.2: Utility weight effects on wait time and travel distance
- RQ10.1: Parameter resolution for sufficient trade-off curve detail

## Experimental Design

### Configuration
- **Dataset**: DiDi (15,000 tasks, 10,000 workers) - 0.67 ratio
- **Strategy**: Composite scoring
- **Focus**: Most promising parameter ranges from previous research
- **Grid**: 3×3×2×2×1 = 36 combinations

### Parameter Matrix
| Parameter | Values | Count |
|-----------|--------|-------|
| fairness_weight (λ₁) | 0.5, 1.0, 2.0 | 3 |
| starvation_weight (λ₂) | 0.5, 1.0, 2.0 | 3 |
| utility_weight (λ₃) | 1.0, 1.5 | 2 |
| soft_threshold | 0.5, 1.0 | 2 |

**Total Configurations**: 36 experiments (~6 hours execution)

## Data Structure
```
data/
└── custom_parameter_sweep_20250917_232026.json
```

## Success Criteria
1. Focused parameter exploration completed
2. Manageable experiment size maintained
3. Realistic timing expectations met
