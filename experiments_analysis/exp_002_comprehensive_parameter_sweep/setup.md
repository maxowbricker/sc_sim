# Experiment 003: Comprehensive Parameter Sweep

## Objective
**Primary Goal**: Systematically explore the relationship between all three lambda parameters and the soft threshold to understand their interactions and find optimal configurations.

**Research Questions Addressed**: 
- RQ1.1-1.4: Complete fairness-efficiency trade-off analysis
- RQ10.1-10.4: Parameter sensitivity analysis across full parameter space

## Hypothesis
Multi-dimensional parameter exploration will reveal optimal configurations and parameter interactions that single-parameter studies cannot identify.

## Experimental Design

### Configuration
- **Dataset**: Variable (5K-50K tasks based on mode)
- **Strategy**: Composite scoring with full parameter grid
- **Modes**: 
  - quick: 30 minutes (36 experiments)
  - standard: 1-2 hours (162 experiments)  
  - extensive: 3-4 hours (400 experiments)
  - overnight: 6-8 hours (1,225 experiments)

### Parameter Matrix
| Parameter | Values | Focus Areas |
|-----------|--------|-------------|
| fairness_weight (λ₁) | 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 | 7 values |
| starvation_weight (λ₂) | 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 | 7 values |
| utility_weight (λ₃) | 0.2, 0.5, 1.0, 1.5, 2.0 | 5 values |
| soft_threshold | 0.05, 0.1, 0.5, 1.0, 2.0 | 5 values |

**Maximum Configurations**: 7×7×5×5 = 1,225 experiments

### Focus Areas
- **all**: Test all combinations (default)
- **fairness**: Focus on fairness-optimized ranges
- **efficiency**: Focus on efficiency-optimized ranges  
- **balanced**: Focus on balanced configurations

## Data Structure
```
data/
├── comprehensive_parameter_sweep_20250914_231517.json
├── comprehensive_parameter_sweep_20250914_231602.json
├── comprehensive_parameter_sweep_20250915_073812.json
└── analysis/
    ├── parameter_interactions.csv
    ├── pareto_analysis.json
    └── optimal_configurations.csv
```

## Success Criteria
1. Complete parameter space exploration
2. Identification of parameter interactions
3. Pareto frontier establishment
4. Optimal configuration recommendations
