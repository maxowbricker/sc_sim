# Experiment 006: Focused Parameter Sweep

## Objective
**Primary Goal**: Deep exploration of promising parameter ranges with optimized bounds for finding optimal configurations within refined parameter space.

**Research Questions Addressed**: 
- RQ1.4: Sweet spot weight configuration optimization
- RQ10.2: Parameter interactions with strongest effects

## Experimental Design

### Configuration
- **Dataset**: Large scale (50,000 tasks for robust results)
- **Strategy**: Composite scoring with refined parameter ranges
- **Focus**: High-fairness and efficiency-focused configurations

### Optimized Parameter Ranges
| Parameter | Values | Rationale |
|-----------|--------|-----------|
| fairness_weight (λ₁) | 1.0-2.0 | Focused around high-fairness configurations |
| starvation_weight (λ₂) | 0.5-1.5 | Balanced starvation prevention |
| utility_weight (λ₃) | 0.5-2.0 | Efficiency-focused range |
| soft_threshold | 0.25-1.25 | Refined threshold exploration |

### Modes
- **standard**: 5×5×5×5 grid = 625 experiments (~10-12 hours)
- **fine**: 7×7×7×7 grid = 2,401 experiments (~20-24 hours)  
- **ultra**: 9×9×9×9 grid = 6,561 experiments (~40-48 hours)

## Data Structure
```
data/
├── focused_parameter_sweep_20250916_103413.json
├── focused_parameter_sweep_20250916_120243.json
└── analysis/
    ├── refined_pareto_frontier.csv
    └── optimal_configurations.json
```

## Success Criteria
1. Refined parameter space thoroughly explored
2. Optimal configurations identified within focused ranges
3. Parameter interactions quantified
