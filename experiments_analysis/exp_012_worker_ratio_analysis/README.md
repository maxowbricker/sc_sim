# Experiment 012: Worker-to-Task Ratio Analysis

**Goal**: Find optimal worker count for 20K task workload

## Quick Reference

- **Worker Counts**: 2K, 3K, 4K, 5K, 6K, 7K, 8K, 9K, 10K, 12K, 15K (11 total)
- **Fixed Tasks**: 20,000 (stratified across 12 temporal bins)
- **Configuration**: λ₁=2.0, λ₂=0.5, λ₃=1.0, θ=0.0
- **Duration**: ~2.5-3 hours
- **Innovation**: Stratified temporal sampling (fixes Exp 011 failures)

## Run Experiment

```bash
cd /Users/maxapple/Documents/GitHub/sc_sim/experiments_analysis/exp_012_worker_ratio_analysis

# Foreground (watch progress)
../../venv/bin/python run_experiment.py

# Background (run unattended)
nohup ../../venv/bin/python run_experiment.py > experiment_012_run.log 2>&1 &

# Monitor progress
tail -f experiment_012_run.log
```

## Key Innovation: Stratified Temporal Sampling

### Problem (Exp 011)
- Random sampling → temporal misalignment
- Tasks arrived early, workers available late
- Early tasks starved → 0% TAR for 4K+ workers

### Solution (Exp 012)
- **Stratified sampling** across 12 temporal bins (15-min each)
- Tasks sampled proportionally from each bin
- Workers sampled to match task arrival distribution
- Result: 8-9% early availability (vs 5% random), 94.3% TAR validated

## Expected Outcomes

1. **All experiments achieve >85% TAR** (validates sampling)
2. **"Knee" at 6K-8K workers** (optimal balance)
3. **TAR plateaus** above 6K workers
4. **Utilization decreases linearly** with worker count
5. **Gini lowest** at 6K-8K workers (best distribution)

## Metrics Collected (78+)

- **Core**: TAR, JFI, completed tasks
- **Tier 1**: Task wait (mean, std, p90, p95, max, CV), Worker idle (mean, std, p90, max, CV)
- **Tier 2**: Worker task distribution (Gini, CV, percentiles), Utilization, Pickup distances, Deferrals, Assignment timing
- **System**: Travel km, empty km, backlog, EWMA CV

## Analysis

See `ANALYSIS_PLAN.md` (to be created) for visualization strategy.

Key plots:
1. TAR vs Worker Count (identify knee)
2. Tasks/Worker vs Worker Count (trade-off)
3. Gini vs Worker Count (fairness sweet spot)
4. Wait Time vs Worker Count (efficiency)
5. Utilization vs Worker Count (resource efficiency)

## Files

- `setup.md` - Detailed experimental design and hypotheses
- `run_experiment.py` - Main experiment script
- `data/` - Results directory
- `experiment_012_aggregate_results.csv` - Summary CSV

## Status

- [x] Experimental design complete
- [x] Stratified sampler implemented (`data/stratified_sampler.py`)
- [x] Validation test passed (94.3% TAR with 4K workers)
- [ ] Experiment running
- [ ] Results analysis
- [ ] Findings documentation

## Connection to Research

**Research Question**: What worker-to-task ratio maximizes both efficiency (TAR) and fairness tracking accuracy?

**Hypothesis**: 6K-8K workers optimal (2.5-3.3 tasks/worker)

**Impact**: Informs deployment recommendations and future experiment design




