# FATP-ANN Strategy Implementation

## Overview

**FATP-ANN** (Fairness-Aware Task Planning with Approximate Nearest Neighbor) is a published baseline algorithm for fair task allocation in spatial crowdsourcing. This implementation provides a comparison point for evaluating custom strategies.

## Key Features

### 1. Fairness Cap Mechanism
- **Dynamic Fairness Cap (ĉ)**: Calculated as `sum(Count_i²) / sum(Count_i)` where `Count_i` is the number of tasks completed by worker `i`
- **O(1) Incremental Updates**: Efficient tracking using running sums
- **Prevents Overloading**: Workers cannot be assigned tasks if they've reached the fairness cap

### 2. Two Event Handlers

#### Task-Process (TP) - NEW_TASK Event
When a new task arrives:
1. Scan all available workers (or k-nearest if optimization enabled)
2. Filter candidates: `worker.completed_tasks < ĉ` AND task is feasible
3. Assign to **nearest eligible worker** (minimum pickup distance)
4. If no eligible worker, task remains in pool

#### Worker-Process (WP) - FREE_WORKER Event
When a worker becomes available:
1. Initialize shadow state (location, time) for hypothetical multi-task planning
2. **Iteratively assign tasks** while `worker.completed_tasks < ĉ`:
   - Find all valid tasks from task pool
   - Calculate **utility with exponential decay** for each
   - Select task with **maximum utility**
   - Assign task and update shadow state
3. Return first assigned task

### 3. Utility-Based Task Selection

**Formula:**
```
u_r = α_r × exp(-μ × (completion_time - release_time))
```

Where:
- `α_r` = Base utility (proportional to task distance in km)
- `μ` = Decay factor (default: 0.1)
- `completion_time - release_time` = Task wait time in hours

**Intuition:** Longer tasks are more valuable, but utility decays exponentially as tasks wait longer.

## Configuration

### Config Parameters (`config.py`)

```python
"fatp_ann": {
    "mu": 0.1,              # Decay factor (assumes time in hours)
    "alpha_scale": 1.0,     # Scaling factor for base utility
    "use_k_nearest": False, # Toggle k-NN optimization
    "k": 15,                # Number of nearest workers (if use_k_nearest=True)
}
```

### Parameter Guidance

| Parameter | Default | Description | Tuning Guide |
|-----------|---------|-------------|--------------|
| `mu` | 0.1 | Decay rate for task utility | Higher = more aggressive decay (prioritize fresh tasks) |
| `alpha_scale` | 1.0 | Base utility multiplier | Increase if utility values too small; decrease if too large |
| `use_k_nearest` | False | k-NN optimization | Enable if simulations are too slow |
| `k` | 15 | Nearest workers to consider | Only used if `use_k_nearest=True` |

## Usage

### Basic Simulation

```python
from simulator.simulation import run_simulation
from data.loader import load_workers_tasks

# Load data
workers, tasks = load_workers_tasks('didi', 'data/didi')

# Configure FATP-ANN
config = {
    'assignment_strategy': 'fatp_ann',
    'mu': 0.1,
    'alpha_scale': 1.0,
    'use_k_nearest': False,
    'k': 15
}

# Run simulation
summary = run_simulation(workers, tasks, sim_config=config)

print(f"Completed: {summary['completed_tasks']}")
print(f"JFI: {summary['final_jains_fairness_index']:.4f}")
print(f"Avg wait: {summary['avg_wait_time_minutes']:.2f} min")
```

### Experiment Comparison

```python
# Compare FATP-ANN vs Greedy vs Composite
strategies = ['fatp_ann', 'greedy', 'composite']
results = {}

for strategy in strategies:
    config = {'assignment_strategy': strategy}
    if strategy == 'fatp_ann':
        config.update({'mu': 0.1, 'alpha_scale': 1.0})
    
    summary = run_simulation(workers.copy(), tasks.copy(), sim_config=config)
    results[strategy] = summary

# Analyze trade-offs
for name, summary in results.items():
    print(f"{name:12} | JFI: {summary['final_jains_fairness_index']:.3f} | "
          f"TAR: {summary['task_assignment_ratio']:.1f}% | "
          f"Wait: {summary['avg_wait_time_minutes']:.1f} min")
```

## Implementation Details

### Files Modified/Created

1. **`models/task.py`**: Added `base_utility` attribute (calculated from task distance)
2. **`config.py`**: Added `fatp_ann` configuration section
3. **`simulator/strategies/fatp_ann.py`**: Main implementation (~410 lines)
   - `FairnessCapTracker` class
   - `assign_new_tasks_fatp_ann()` (TP handler)
   - `match_worker_fatp_ann()` (WP handler)
   - Helper functions: `_calculate_utility()`, `_is_valid_assignment()`, etc.
4. **`simulator/strategies/__init__.py`**: Registered strategy
5. **`simulator/simulation.py`**: Added FairnessCapTracker initialization for fatp_ann

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| TP (Task-Process) | O(\|W\|) per task | Scans all workers; O(k) if k-NN enabled |
| WP (Worker-Process) | O(\|R_pool\| × tasks_assigned) | Scans task pool for each assignment |
| Cap Update | O(1) | Incremental calculation |

**Performance Note:** FATP-ANN is significantly slower than k-NN strategies (Greedy, Composite) due to full scans. Enable `use_k_nearest=True` for large-scale experiments.

## Research Questions Addressed

This strategy supports:
- **RQ4.2**: Baseline comparison (FATP-ANN as published algorithm)
- **RQ6**: Alternative fairness mechanisms (fairness cap vs. EWMA)
- **RQ11**: Algorithm comparison and trade-off analysis

## Validation

Run the test suite to verify implementation:

```bash
python test_fatp_ann.py
```

**Expected Output:**
- ✅ Fairness cap calculation (O(1) updates)
- ✅ Utility calculation (exponential decay)
- ✅ End-to-end simulation (FATP-ANN completes successfully)

## Known Limitations

1. **No GMM Idle Movement**: Original paper uses Gaussian Mixture Models to direct idle workers to task hotspots. This implementation skips this feature (workers remain idle instead).

2. **Single-Task Events**: The simulation uses single-task event returns, but WP assigns multiple tasks internally. Only the first task is returned to the event queue; others are committed immediately.

3. **Performance**: Full worker/task scans are expensive. Enable k-NN optimization for datasets > 5K workers.

## References

Paper: "Fair Task Allocation in Crowdsourced Delivery" (Algorithm 1 & 4)
- Task-Process: Algorithm 4
- Worker-Process: Algorithm 1
- Fairness Cap: Equation 8

## Future Enhancements

1. **GMM Integration**: Implement idle worker repositioning using clustering
2. **Adaptive k-NN**: Dynamically adjust k based on worker density
3. **Fairness Cap Variants**: Explore alternative cap formulations (e.g., percentile-based)
4. **Parallel Processing**: Optimize TP/WP with spatial indexing for large-scale datasets



