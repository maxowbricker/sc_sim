# Spatial Crowdsourcing Assignment Strategies

This directory contains the implementation of various task assignment strategies for the spatial crowdsourcing simulator.

## Available Strategies

### 1. Greedy Strategy (`greedy.py`)
**Pure efficiency-based assignment**

Assigns tasks to the nearest available worker based on Manhattan distance. This serves as the baseline for comparing fairness-aware strategies.

**Usage:**
```python
config = {
    'assignment_strategy': 'greedy'
}
```

**Characteristics:**
- Fast assignment decisions
- Minimizes travel distances
- No fairness considerations
- Baseline for comparison

---

### 2. Composite Strategy (`composite.py`)
**Fairness-aware assignment with multi-objective optimization**

The main contribution of this research. Combines fairness, starvation prevention, and spatial utility into a single composite score for assignment decisions.

#### Basic Configuration

```python
config = {
    'assignment_strategy': 'composite',
    'strategy_params': {
        'fairness_weight': 1.0,      # λ₁: Weight for fairness component
        'starvation_weight': 1.0,    # λ₂: Weight for starvation component  
        'utility_weight': 0.5,       # λ₃: Weight for utility component
        'gamma': 0.3,                # EWMA smoothing factor
        'k': 15,                     # Number of nearest workers to consider
        'soft_threshold': 0.5,       # Minimum score for immediate assignment
        'fairness_metric': 'ewma'    # Fairness calculation method
    }
}
```

#### Score Components

The composite score is calculated as:
```
Score = λ₁·Fairness + λ₂·Starvation + λ₃·Utility
```

Where:
- **Fairness**: EWMA of worker idle time (higher = more underserved)
- **Starvation**: log(1 + task_age) (prevents tasks from being ignored)
- **Utility**: 1/(1 + distance) (spatial efficiency)

#### Fairness Metrics

**EWMA (Default):**
```
Fairness(w) = (1 - γ) · T_idle(w) + γ · Previous_EWMA
```
- Exponentially weighted moving average of idle time
- `γ ∈ [0,1]`: 0.1 = responsive, 0.9 = smooth
- Balances recent and historical fairness

**Other metrics:**
- `'idle_time'`: Direct idle time (no smoothing)
- `'task_count'`: Inverse of completed tasks

#### Assignment Mechanism

1. **Candidate Selection**: Find k nearest workers (k=15 by default)
2. **Scoring**: Calculate composite score for each candidate
3. **Threshold Check**: Assign if `score >= soft_threshold`
4. **Deferral**: If no candidate meets threshold, defer task for later

---

### 3. Experiment 008 Extensions

#### Score Normalization

**Purpose**: Prevent one component from dominating due to scale mismatch.

**Enable normalization:**
```python
config = {
    'assignment_strategy': 'composite',
    'strategy_params': {
        # ... other params ...
        'normalize_scores': True  # ⚠️ EXPERIMENTAL (Experiment 008)
    }
}
```

**How it works:**
- For each assignment decision, collect F, S, U values for all k candidates
- Apply min-max normalization: `norm = (val - min) / (max - min)`
- Compute composite score with normalized values
- Ensures all components contribute proportionally to their weights

**When to use:**
- Testing hypothesis that Fairness component dominates due to scale
- Diagnosing worker idle time paradox
- Comparing fairness-efficiency trade-offs

---

#### Soft Threshold Ablation

**Purpose**: Test if threshold-based deferral creates artificial task shortages.

**Disable soft threshold:**
```python
config = {
    'assignment_strategy': 'composite',
    'strategy_params': {
        # ... other params ...
        'disable_soft_threshold': True  # ⚠️ EXPERIMENTAL (Experiment 008)
    }
}
```

**How it works:**
- Bypasses the `score >= threshold` check
- Always assigns task to best candidate (if any exist)
- No task deferrals due to low scores

**When to use:**
- Testing hypothesis that threshold delays cause idle time paradox
- Ablation studies
- Understanding impact of deferral mechanism

**⚠️ Warning:** Disabling threshold may reduce fairness by forcing assignments even when all candidates have poor fairness scores.

---

### 4. Diagnostic Tracking (Experiment 008)

**Purpose**: Collect detailed data about assignment decisions for analysis and debugging.

**Enable diagnostic tracking:**
```python
config = {
    'assignment_strategy': 'composite',
    'strategy_params': {
        # ... other params ...
        'enable_diagnostics': True  # ⚠️ PERFORMANCE IMPACT
    }
}
```

**What it records:**
- Raw and normalized component values (F, S, U) for each assignment
- Which component dominates each assignment decision
- Dominance ratio (max component / sum of others)
- Task deferral events and reasons
- Score statistics and distributions

**Performance considerations:**
- **Default: OFF** - Diagnostics disabled by default for performance
- **When enabled**: Forces SLOW PATH in assignment algorithm
- **Performance impact**: 2-3x slower (e.g., 3 hours → 6-9 hours)
- **Only enable when needed** for experimental analysis

**Accessing diagnostic data:**
```python
from simulator.simulation import run_simulation

summary = run_simulation(workers, tasks, sim_config=config)

if 'diagnostic_tracker' in summary:
    tracker = summary['diagnostic_tracker']
    
    # Get summary statistics
    stats = tracker.get_summary_stats()
    print(f"Fairness dominated {stats['dominance_percentages']['fairness']:.1f}% of assignments")
    print(f"Deferral rate: {stats['deferral_rate']*100:.1f}%")
    
    # Export detailed records for analysis
    assignments_df = tracker.to_dataframe('assignments')
    deferrals_df = tracker.to_dataframe('deferrals')
```

**Use cases:**
- Understanding why certain components dominate
- Measuring impact of normalization
- Analyzing deferral patterns
- Diagnosing worker idle time paradox

---

## Strategy Selection Guidelines

### Use Greedy when:
- Pure efficiency is the goal
- Establishing baseline performance
- No fairness requirements

### Use Composite when:
- Fairness is important alongside efficiency
- Preventing worker starvation
- Research on fairness-efficiency trade-offs

### Use Composite + Normalization when:
- Component scale mismatch suspected
- One component consistently dominates
- Testing Hypothesis 1 from Experiment 008

### Use Composite + No Threshold when:
- Investigating impact of task deferrals
- Threshold mechanism suspected of causing issues
- Testing Hypothesis 2 from Experiment 008

---

## Parameter Tuning Recommendations

### Optimal Configuration (from Experiment 006)
"Sweet spot" parameters that balance fairness and efficiency:

```python
{
    'fairness_weight': 0.5,
    'starvation_weight': 0.8,
    'utility_weight': 0.8,
    'soft_threshold': 0.5,
    'gamma': 0.5
}
```

**Results:**
- JFI: 0.294 (9.3% improvement over Greedy)
- TAR: 86.2% (maintained throughput)
- Trade-off: +72% wait time and pickup distance

### Tuning Guidelines

**For higher fairness (↑ JFI):**
- Increase `fairness_weight` (try 1.0-2.0)
- Decrease `utility_weight` (try 0.3-0.5)
- Increase `gamma` for smoother EWMA (try 0.5-0.7)

**For higher efficiency (↓ wait time, ↓ distance):**
- Increase `utility_weight` (try 1.0-1.5)
- Decrease `fairness_weight` (try 0.3-0.5)
- Lower `soft_threshold` (try 0.2-0.4)

**To reduce worker idle times:**
- Enable `normalize_scores=True` (Experiment 008 finding)
- Consider `disable_soft_threshold=True` (test carefully)
- Increase `utility_weight` relative to `fairness_weight`

---

## Implementation Details

### Spatial Optimization
All strategies use **k-nearest neighbors** approach:
- Instead of checking all workers O(|W|)
- Check only k=15 nearest workers O(k log k)
- Massive performance improvement: 38,000 → 15 workers per task

### EWMA Fairness Calculation
Occurs at assignment time, not during idle time updates:
- Prevents competing updates from different code paths
- Ensures consistency with research methodology
- Updated in `calculate_fairness_signal()` function

---

## Adding New Strategies

To add a new assignment strategy:

1. Create `your_strategy.py` in this directory
2. Implement two handler functions:
   ```python
   def assign_new_tasks_your_strategy(state, now, tasks_to_assign, **params):
       # Handle new task arrivals
       return assignments
   
   def match_worker_your_strategy(state, now, worker, **params):
       # Handle worker becoming free
       return assignment or None
   ```

3. Register the strategy:
   ```python
   from simulator.strategies import register
   
   @register("your_strategy")
   def get_your_strategy_handlers():
       return {
           "NEW_TASK": assign_new_tasks_your_strategy,
           "FREE_WORKER": match_worker_your_strategy
       }
   ```

4. Add parameters to `config.py`:
   ```python
   STRATEGY_PARAMS = {
       "your_strategy": {
           # your parameters here
       }
   }
   ```

---

## References

- **Composite Strategy Research**: `../docs/research_proposal.md`
- **Experiment 006**: Comprehensive parameter sweep and paradox discovery
- **Experiment 008**: Score normalization and threshold ablation
- **Configuration**: `../../config.py`
- **Diagnostic Tracker**: `../../metrics/diagnostic_tracker.py`

