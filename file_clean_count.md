# File Clean & Optimize Tracker

Track which files have been cleaned/optimized during the conference-ready refactor.  
Order: from deepest architecture up to wrappers and documentation.

---

## CLEANED

| File | Notes |
|------|-------|
| `models/task.py` | Lazy-loaded base_utility, lean domain model |
| `models/worker.py` | Removed gamma, revenue; lean domain model |
| `simulator/spatial_index.py` | Removed dead SpatialIndex class and broken benchmark; GridSpatialIndex only |
| `simulator/state.py` | Worker/Task pool management, fast lookups |
| `simulator/simulation.py` | Main event loop |
| `data/loader.py` | Data loading, Flat Earth setup, load_day_data |
| `simulator/__init__.py` | Package re-exports (Simulation, run_simulation, StateManager) |
| `simulator/strategies/composite.py` | Primary strategy; unified ANN loop, threshold resolution |
| `simulator/strategies/__init__.py` | Strategy registry (get_strategy, register) |
| `metrics/manager.py` | Central hub; orchestrates trackers, RL step stats |
| `metrics/fairness.py` | JFI, UD, FL, FairnessMetricsTracker |
| `metrics/tracker.py` | Per-tick snapshots, historical time-series |
| `metrics/deferral_tracker.py` | Deferral lifecycle (optional) |
| `data/stratified_sampler.py` | |
| `data/didi/didi.py` | Simplified adapter |
| `data/didi/didi_optimized.py` | |
| `config.py` | |
| `main.py` | |

---

## TO CLEAN (Priority Order)

# DEFER UNTIL PROMISING RESULTS WITH DRL:


### 2. Baseline Strategies

Ensure they don’t break after worker.py changes (gamma, revenue removed). Share optimization tricks from composite.py.

| # | File | Notes |
|---|------|-------|
| 11 | `simulator/strategies/fatp_ann.py` | |
| 12 | `simulator/strategies/fatp.py` | |
| 13 | `simulator/strategies/ewma_only.py` | |
| 14 | `simulator/strategies/laf.py` | |
| 15 | `simulator/strategies/greedy.py` | |
| 16 | `simulator/strategies/random_assign.py` | |

### 3. Reinforcement Learning (Deferred)

Queue for later. Ensure observation space relies on cleaned MetricsManager.

| # | File | Notes |
|---|------|-------|
| 27 | `rl/gym_environment.py` | |
| 28 | `rl/test_env_quick.py` | |
| 29 | `rl/train_sb3.py` | |
| 30 | `rl/tune_sb3.py` | |

### 4. Documentation (The Polish)

Once code is locked in, ensure manuals match the engine.

| # | File | Notes |
|---|------|-------|
| 31 | `docs/DATA_DICTIONARY.md` | Update after dropped variables |
| 32 | `docs/METRICS_OUTLINE.md` | |
| 33 | `docs/SIMULATION_GUIDE.md` | |
| 34 | `docs/strategies/README.md` | |
| 35 | `docs/strategies/FATP_ANN_README.md` | |
| 36 | `docs/README.md` | |
