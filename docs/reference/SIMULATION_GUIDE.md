# Simulation Guide

Practical reference for running simulations, configuring strategies, and interpreting outputs. For the metric definitions see `DATA_DICTIONARY.md`; for the training run file layout see `docs/README.md` or `rl_logs_sb3/EXPERIMENTATION_PROCESS.md`.

---

## 1. Prerequisites

```bash
conda activate sc          # or source venv/bin/activate
cd /path/to/sc_sim
```

Data must be in place:

```
data/didi/full_didi_gaia/<day_folder>/order.txt
data/didi/full_didi_gaia/<day_folder>/gps.txt
```

---

## 2. Running a static simulation

### Via `main.py` (recommended)

```bash
# Uses all settings from config.py
python main.py

# Override specific fields on the command line
python main.py --strategy greedy
python main.py --strategy composite --root data/didi/full_didi_gaia/496528674@qq.com_20161128
```

**Smoke test**: To quickly verify the simulator works after code changes, run:

```bash
python scripts/test_simulation_simple.py
```

This runs two basic simulation configs and the `EventSimulator` interface (used by RL) on the full dataset. Helps catch breakage before running expensive RL training.

---

## 3. Configuration

All settings live in **`config.py`**. Never hard-code values in experiment scripts — override via `create_composite_config(**overrides)` instead. It raises a `ValueError` on unknown keys to catch typos.

### Core simulation

```python
SIMULATION_CONFIG = {
    "dataset": "didi",                  # "didi" | "synthetic"
    "data_root_path": None,             # Override path (None = use DATA_SAMPLING default)
    "assignment_strategy": "composite", # "composite" | "greedy" | "fatp_ann" | "ewma_only" | "random_assign"
}
```

### Data sampling

```python
DATA_SAMPLING = {
    "use_stratified_sampling": True,  # Sample a subset (faster, consistent scale)
    "target_tasks": 40000,            # Target task count after sampling
    "target_workers": 10000,          # Target worker count after sampling
    "stratified_sampling_bins": 288,  # Temporal bins (288 = 5-min bins per 24h)
    "random_state": 42,               # Reproducibility seed
}
```

Turn off `use_stratified_sampling` when you need the full raw day (~220k tasks, ~38k workers).

### Composite strategy parameters

```python
STRATEGY_PARAMS["composite"] = {
    "fairness_weight": 1.0,      # λ1 — weight on EWMA fairness score F
    "starvation_weight": 0.2,    # λ2 — weight on idle-time starvation score S
    "utility_weight": 1.0,       # λ3 — HARDCODED, not tuned (anchors action space)
    "gamma": 0.1,                # EWMA decay (0.1 = responsive to recent history)
    "k": 15,                     # k nearest workers considered per task
    "soft_threshold": 0.05,      # Min score to assign immediately (0.0 = disabled)
    "enable_diagnostics": False, # Heavy IOR/eligibility fairness tracking (slow)
    "enable_deferral_tracking": False,
}
```

Scoring function per (task, worker) candidate:

```
Score = λ1 × F  +  λ2 × S  +  λ3 × U

F = worker.fairness_ewma              (EWMA-smoothed historical fairness)
S = worker.total_idle_time / elapsed  (idle fraction)
U = 1 / (1 + d_pick_km)              (proximity efficiency)
```

The worker with the highest score among the k nearest gets the task.

### Platform revenue

```python
PLATFORM_REVENUE = {
    "base_fare": 2.00,      # Fixed component ($)
    "per_km_rate": 1.50,    # Per-km rate on pickup→dropoff distance ($/km)
}
# task.revenue = base_fare + per_km_rate × core_movement_cost_km
```

---

## 4. Available strategies

| Strategy key | Description |
|-------------|-------------|
| `composite` | Research strategy — λ-weighted fairness + starvation + utility score |
| `greedy` | Nearest available worker (proximity only) |
| `fatp_ann` | First Available Time Priority with ANN proximity |
| `ewma_only` | Score on EWMA fairness only |
| `random_assign` | Random from k nearest workers |
| `mmd_batch` | Batch assignment using MMD |

For strategy implementation details see `docs/reference/strategies/README.md`.

---

## 5. RL training

### Starting a training run

```bash
# Full training run (Optuna-tuned hyperparameters, 300k timesteps)
python rl/train_sb3.py --timesteps 300000 --hyperparams best_hyperparameters.json

# Quick smoke test (confirm env/data is working)
python rl/train_sb3.py --timesteps 1000

# Resume from checkpoint
python rl/train_sb3.py \
  --resume rl_logs_sb3/run_YYYYMMDD_HHMMSS/ppo_sc_model_50000_steps.zip \
  --timesteps 300000
```

### How the RL environment works

The RL environment (`rl/gym_environment.py`, class `AdaptiveSpatialCrowdsourcingEnv`) wraps `EventSimulator`:

- **Episode**: One day of simulation, entered at a random drop-in time after a 30-minute greedy warmup.
- **Step**: 5 minutes of simulation time.
- **Observation** (15-dim Box): Current JFI, delta-JFI, backlog, wait stats, EWMA CV, arrival delta, worker count fractions, etc.
- **Action** (2-dim Box): `[λ1, λ2]` — fairness and starvation weights. λ3 is fixed at 1.0.
  - λ1 ∈ [0, 2], λ2 ∈ [0, 0.5]
- **Reward**: Weighted combination of JFI improvement, wait-time reduction, and backlog penalty.

The exact reward formula is in `rl/gym_environment.py` (`_calculate_reward`). Each run folder stores a frozen copy as `gym_environment_snapshot.py` so older runs remain reproducible even when the live file changes.

### Hyperparameters
Current best hyperparameters (`best_hyperparameters.json`):

```json
{
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 256,
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "ent_coef": 0.04,
    "vf_coef": 0.55,
    "max_grad_norm": 1.0,
    "net_arch_type": "large"
}
```

These were found just through trial and error and using close to suggested PPO defaults

---

## 6. Reading training run outputs

Each run in `rl_logs_sb3/run_YYYYMMDD_HHMMSS/` produces:

### Headline results

**`baseline_best_model_metrics.txt`** — the primary result table:

```
Metric               | Static Baseline | RL Agent  | Improvement
------------------------------------------------------------
JFI (Fairness)       | 0.5652          | 0.5501    | -0.0151
Peak Backlog         | 103             | 103       | +0
Avg Wait Time (m)    | 2.66            | 2.65      | -0.02
```

The **static baseline** is: 30-min greedy warmup → 8 hours of composite with the fixed λ from `config.py`. The **RL agent** is: same warmup → 8 hours with policy-chosen λ every 5 minutes. This isolates the value of dynamic λ adaptation.

**`baseline_best_model_metrics.txt`** uses the **best eval checkpoint** (`best_model/best_model.zip`). `baseline_final_model_metrics.txt` uses the final checkpoint after the last update.

### Policy weight trace

**`eval_weights_best_steps.txt`** — the λ the policy chose at each step:

```
Step 1: λ1=0.043, λ2=0.012, λ3=1.000 (reward=2.70, sim_time=...)
Step 2: λ1=0.160, λ2=0.040, λ3=1.000 (reward=2.48, sim_time=...)
...
```

Use this to understand how the policy adapted — e.g. whether it converged to near-static weights or genuinely varied them. `eval_weights_best.csv` has the same data in a machine-readable form.

### Reproducibility files

| File | Purpose |
|------|---------|
| `gym_environment_snapshot.py` | Frozen copy of the env at train time (reward formula, obs space) |
| `hyperparams_snapshot.json` | Frozen PPO hyperparameters used |
| `environment_spec.json` | Observation/action space + reward config in JSON |
| `run_manifest.json` | Full audit: command, git hash, eval day, train/test split, return codes |

---

## 7. Manual baseline comparison

Run the eval script directly against any model:

```bash
python scripts/compare_model_to_baseline.py \
  --model rl_logs_sb3/run_YYYYMMDD_HHMMSS/ppo_sc_final \
  --day 496528674@qq.com_20161128 \
  --data-root data/didi/full_didi_gaia \
  --eval-seed 42 \
  --log-weights my_weight_trace.txt \
  --metrics-out my_metrics.txt
```

Or the shell wrapper (evaluates both final and best checkpoints for a run folder):

```bash
./scripts/compare_run_checkpoints_to_baseline.sh rl_logs_sb3/run_YYYYMMDD_HHMMSS
```

---

## 8. Multi-day evaluation

```bash
python scripts/run_6day_evaluation.py
```

Evaluates across 6 held-out days and produces a summary CSV. See `docs/experiments/6DAY_EVAL_RESULTS_MECHANICS.md` for protocol and result interpretation.

---

## 9. Diagnostics mode

When `enable_diagnostics=True` in the composite strategy params, the simulator also computes:

- **`eligibility_utility_difference`** — UD over workers who were within 10 km of released tasks
- **`eligibility_fairness_loss`** — FL from IOR-weighted ideal shares
- **IOR (input/output ratio)** — `actual_tasks / eligible_tasks` per worker

These are expensive (O(|W|) spatial scan per task release) so they are **disabled by default and must never be enabled during RL training**. Enable them only for mechanism analysis on a finished simulation.

---

## 10. Common issues

| Symptom | Fix |
|---------|-----|
| `ValueError: Unknown parameter 'foo'` | `create_composite_config()` caught a typo in your overrides |
| `FileNotFoundError: No day folders found` | Check `data/didi/full_didi_gaia/` exists and contains day folders |
| RL env very slow | `enable_diagnostics` is probably on — set it to `False` |
| JFI = 1.0 suspiciously | Very small dataset — all workers got identical task counts |
| Model not improving after many steps | Check `eval_logs/evaluations.npz` reward trend; likely needs reward re-tuning |
