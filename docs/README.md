# sc_sim — Spatial Crowdsourcing Simulator

A discrete-event simulation engine for studying fair task allocation in spatial crowdsourcing markets, with a PPO-based deep reinforcement learning agent for dynamic weight adaptation.

Research paper (honours thesis): https://drive.google.com/file/d/15yfEliaieEv4Ulx9Z_bCm_OOsYGUIZOo/view?usp=sharing

**Branches:** see [`docs/BRANCHES.md`](BRANCHES.md) for which branch to checkout (start on **`main`** — formerly `conference-ready`).

---

## Repository layout

```
sc_sim/
├── main.py                      # Entry point — static simulation
├── config.py                    # All configuration (read this first)
├── best_hyperparameters.json    # Tuned PPO hyperparameters (used by train_sb3.py)
│
├── models/                      # Domain objects
│   ├── worker.py                # Worker: id, position, deadline, earnings, EWMA state
│   └── task.py                  # Task: pickup/dropoff, release/expire, revenue
│
├── simulator/                   # Core simulation engine
│   ├── simulation.py            # EventSimulator + run_simulation() — start here
│   ├── state.py                 # Worker/task pools, fast lookups
│   ├── spatial_index.py         # Manhattan-km distance, GridSpatialIndex, k-NN
│   ├── behavior.py              # Worker stochastic acceptance (Basık cascade, off by default)
│   └── strategies/              # Assignment strategies
│       ├── composite.py         # Primary strategy (fairness + starvation + utility score)
│       ├── greedy.py            # Nearest-worker baseline
│       ├── fatp_ann.py          # FATP-ANN baseline
│       ├── ewma_only.py         # EWMA-only baseline
│       ├── random_assign.py     # Random-from-k-nearest baseline
│       └── mmd_batch.py         # MMD batch strategy
│
├── metrics/                     # Metric computation
│   ├── manager.py               # Central hub — all metrics flow through here
│   ├── fairness.py              # JFI, UD, FL, Gini, earnings fairness
│   ├── tracker.py               # Per-tick snapshots (diagnostics mode only)
│   └── deferral_tracker.py      # Task deferral lifecycle (optional)
│
├── data/                        # Data loading
│   ├── loader.py                # load_workers_tasks() — main loading entry point
│   ├── stratified_sampler.py    # Temporal stratified sampling (config DATA_SAMPLING)
│   └── didi/didi.py             # DiDi GAIA dataset adapter
│
├── rl/                          # Reinforcement learning
│   ├── gym_environment.py       # Gymnasium env (AdaptiveSpatialCrowdsourcingEnv)
│   ├── train_sb3.py             # PPO training entry point
│   └── tune_sb3.py              # Optuna hyperparameter search
│
├── scripts/                     # Standalone utilities (see scripts/README below)
│   ├── compare_model_to_baseline.py        # Core eval: RL vs static composite
│   ├── compare_run_checkpoints_to_baseline.sh  # Wrapper for final + best checkpoints
│   └── run_6day_evaluation.py              # Multi-day eval across held-out days
│
├── docs/
│   ├── README.md                           # (this file) Start here
│   ├── reference/
│   │   ├── SIMULATION_GUIDE.md            # How to run simulations and training
│   │   ├── DATA_DICTIONARY.md             # Complete metric and config reference
│   │   ├── METRICS_OUTLINE.md             # Metric architecture and code map
│   │   └── dataset_and_simulation_details.md  # DiDi GAIA dataset details
│   └── experiments/                        # Experiment logs and findings
│
└── rl_logs_sb3/                 # Training run outputs — gitignored, copy manually
    └── run_YYYYMMDD_HHMMSS/     # One folder per training run (see below)
```

---

## Data setup (DiDi GAIA)

The simulator is built around the **DiDi GAIA** Chengdu dataset (~38k drivers, ~220k orders per day). You need to obtain and place this data manually — it is not included in the repo.

Expected layout:

```
data/didi/full_didi_gaia/
└── <day_folder>/          # e.g. 496528674@qq.com_20161128
    ├── order.txt          # orderID,startBillingTime,endBillableTime,pickupLon,pickupLat,dropoffLon,dropoffLat
    └── gps.txt            # driverID,orderID,timestamp,lon,lat
```

Timestamps are Unix seconds. Multiple day folders live under `full_didi_gaia/`; the training/evaluation code iterates over them.

---

## Installation

```bash
# 1. Create and activate environment (not necessary but if you want to encapsulate your dependcy downloads I recomend this, you will just have to do 'conda activate sc' at the start of each terminal session)
conda create -n sc python=3.10
conda activate sc

# 2. Install dependencies (probably missing somehere)
pip install numpy pandas stable-baselines3 gymnasium optuna scikit-learn

# 3. Verify
python main.py --help
```

---

## Data setup (from zip)

The data is distributed as a zip containing one `.tar.gz` per day. Once you've unzipped it, run the setup script to extract and rename everything into the format the simulator expects:

```bash
# Unzip the archive you received
unzip "滴滴 gaiya.zip" -d didi_raw

# Extract all 30 days into data/didi/full_didi_gaia/
python3 scripts/setup_didi_data.py --source "didi_raw/滴滴 gaiya"

# Or just extract a single day to get started quickly
python3 scripts/setup_didi_data.py --source "didi_raw/滴滴 gaiya" --day 20161128
```

The script creates one folder per day under `data/didi/full_didi_gaia/`, each containing `order.txt` and `gps.txt`. Already-extracted days are skipped automatically so it's safe to re-run.

---

## Running a static simulation

```bash
# Uses config.py settings (strategy = composite, dataset = didi, stratified sampling)
python main.py

# Override strategy or data path on the fly
python main.py --strategy greedy
python main.py --strategy composite --root data/didi/full_didi_gaia/496528674@qq.com_20161128
```

The key knobs are in `config.py`:

| Setting | Where | What it does |
|---------|-------|-------------|
| `assignment_strategy` | `SIMULATION_CONFIG` | Which strategy to use |
| `use_stratified_sampling` | `DATA_SAMPLING` | Sample ~40k tasks / 10k workers from full day |
| `fairness_weight` (λ1) | `STRATEGY_PARAMS["composite"]` | Composite strategy fairness weight |
| `starvation_weight` (λ2) | `STRATEGY_PARAMS["composite"]` | Starvation penalty weight |
| `gamma` | `STRATEGY_PARAMS["composite"]` | EWMA decay (0.1 = responsive) |
| `k` | `STRATEGY_PARAMS["composite"]` | k-nearest workers considered per task |
| `enable_diagnostics` | `STRATEGY_PARAMS["composite"]` | Enable heavy fairness tracking (slow, off for RL) |

See `docs/reference/DATA_DICTIONARY.md` for the full config reference.

---

## RL training

```bash
# Standard training run (Optuna-tuned hyperparams, 300k steps)
python rl/train_sb3.py --timesteps 300000 --hyperparams best_hyperparameters.json

# Quick smoke test
python rl/train_sb3.py --timesteps 1000

# Resume from a checkpoint
python rl/train_sb3.py --resume rl_logs_sb3/run_YYYYMMDD_HHMMSS/ppo_sc_model_50000_steps.zip --timesteps 300000
```

Each run creates `rl_logs_sb3/run_YYYYMMDD_HHMMSS/`. See the **Training run folder** section below for what every file in that folder means.

To tune hyperparameters with Optuna first:

```bash
python rl/tune_sb3.py --trials 50
# Results written to best_hyperparameters.json
```

---

## Training run folder — file-by-file guide

Every `rl_logs_sb3/run_YYYYMMDD_HHMMSS/` folder is self-contained and reproducible:

```
run_YYYYMMDD_HHMMSS/
├── gym_environment_snapshot.py     # Snapshot of rl/gym_environment.py at train time
├── hyperparams_snapshot.json       # Copy of best_hyperparameters.json used
├── environment_spec.json           # Observation/action space + reward config (JSON)
├── run_manifest.json               # Full audit log: command, hashes, eval day, return codes
│
├── ppo_sc_final.zip                # Final model after training completes
├── best_model/best_model.zip       # Best checkpoint by eval reward (EvalCallback)
├── ppo_sc_model_N_steps.zip        # Intermediate checkpoints at N steps
│
├── eval_logs/evaluations.npz       # SB3 EvalCallback reward history (numpy)
├── eval.monitor.csv                # Per-episode r/l/t from Monitor wrapper
│
├── eval_weights_final.csv          # Per-step λ trace for final model (CSV)
├── eval_weights_final_steps.txt    # Same, human-readable (Step N: λ1=..., λ2=..., reward=...)
├── eval_weights_best.csv           # Per-step λ trace for best checkpoint (CSV)
├── eval_weights_best_steps.txt     # Same, human-readable
│
├── baseline_final_model_metrics.txt   # Static composite vs RL — summary table (final)
├── baseline_best_model_metrics.txt    # Static composite vs RL — summary table (best)
├── baseline_final_model_weight_outputs.txt  # Alias / companion to eval_weights_final_steps.txt
├── baseline_best_model_weight_outputs.txt   # Alias / companion to eval_weights_best_steps.txt
├── baseline_eval_final.txt         # Full stdout from the final compare run
└── baseline_eval_best.txt          # Full stdout from the best compare run
```

### What to look at first after a training run

1. **`baseline_best_model_metrics.txt`** — the headline result: JFI, peak backlog, and avg wait for *static composite* vs *RL agent* on the held-out eval day.

2. **`eval_weights_best_steps.txt`** — the RL policy's λ1 / λ2 trace per 5-minute step. Shows how the agent adapted weights over the day.

3. **`environment_spec.json`** — confirms the observation/action space and reward config the model was trained under (critical for reproducing results).

4. **`run_manifest.json`** — full audit trail: which eval day was used, train/test split, SHA-256 hashes of snapshots, and return codes for all post-eval runs.

### Evaluation protocol (what the metrics table compares)

The comparison is **not** "greedy vs RL." It is:

- **Both arms** run a **shared 30-minute greedy warmup** first (identical seed, same day).
- **Static baseline** then runs composite with **fixed λ from `config.py`** for 8 hours.
- **RL agent** then runs composite with **policy-chosen λ** (new λ every 5 minutes).

So the table isolates the effect of *learned dynamic λ adaptation* vs *best static fixed λ*.

---

## Key scripts

| Script | When to use |
|--------|------------|
| `scripts/compare_model_to_baseline.py` | Manual eval of any `.zip` model against the static baseline |
| `scripts/compare_run_checkpoints_to_baseline.sh` | Quick wrapper: eval final + best for a run folder |
| `scripts/run_6day_evaluation.py` | Evaluate a model across 6 held-out days; the results are in `docs/experiments/6DAY_EVAL_RESULTS_MECHANICS.md` |

---

## Key reference docs

| Document | Contents |
|----------|----------|
| `docs/reference/SIMULATION_GUIDE.md` | Detailed how-to: config knobs, calling the sim API, running eval |
| `docs/reference/DATA_DICTIONARY.md` | Every metric, config flag, and result key defined |
| `docs/reference/METRICS_OUTLINE.md` | How metrics flow through the code (architecture) |
| `docs/reference/dataset_and_simulation_details.md` | DiDi GAIA dataset details and simulation design choices |
| `rl_logs_sb3/EXPERIMENTATION_PROCESS.md` | Deep dive: training/eval protocol, backfilling old runs |
| `docs/experiments/` | Findings from individual experiments and training runs |
