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

---

## TO CLEAN (Priority Order)

### 1. Core Engine (Deepest Architecture)

If these are slow or bloated, the entire simulation suffers. Ensure highly optimized and memory-efficient.

| # | File | Notes |
|---|------|-------|
| 1 | `simulator/simulation.py` | Main event loop |
| 2 | `simulator/__init__.py` | |

### 2. Primary Strategy

Core contribution. Final pass to ensure it interacts perfectly with cleaned models and state.

| # | File | Notes |
|---|------|-------|
| 5 | `simulator/strategies/composite.py` | |
| 6 | `simulator/strategies/__init__.py` | |

### 3. Metrics & Accounting

Single sources of truth. Ensure no other files do manual math that belongs here.

| # | File | Notes |
|---|------|-------|
| 7 | `metrics/manager.py` | |
| 8 | `metrics/fairness.py` | |
| 9 | `metrics/tracker.py` | |
| 10 | `metrics/deferral_tracker.py` | |

### 4. Baseline Strategies

Ensure they don’t break after worker.py changes (gamma, revenue removed). Share optimization tricks from composite.py.

| # | File | Notes |
|---|------|-------|
| 11 | `simulator/strategies/fatp_ann.py` | |
| 12 | `simulator/strategies/fatp.py` | |
| 13 | `simulator/strategies/ewma_only.py` | |
| 14 | `simulator/strategies/laf.py` | |
| 15 | `simulator/strategies/greedy.py` | |
| 16 | `simulator/strategies/random_assign.py` | |

### 5. Data Layer

Ensure data is loaded cleanly and efficiently into the models.

| # | File | Notes |
|---|------|-------|
| 17 | `data/loader.py` | |
| 18 | `data/stratified_sampler.py` | |
| 19 | `data/didi/didi.py` | |
| 20 | `data/didi/didi_optimized.py` | |

### 6. Configuration & Standard Entry Points

Standardize how a run is defined and executed.

| # | File | Notes |
|---|------|-------|
| 21 | `config.py` | |
| 22 | `main.py` | |
| 23 | `scripts/test_simulation_simple.py` | Canonical physics test |
| 24 | `scripts/test_simulation_timed.py` | |
| 25 | `scripts/validate_dynamic.py` | |
| 26 | `scripts/tune_physics_full.py` | |

### 7. Reinforcement Learning (Deferred)

Queue for later. Ensure observation space relies on cleaned MetricsManager.

| # | File | Notes |
|---|------|-------|
| 27 | `rl/gym_environment.py` | |
| 28 | `rl/test_env_quick.py` | |
| 29 | `rl/train_sb3.py` | |
| 30 | `rl/tune_sb3.py` | |

### 8. Documentation (The Polish)

Once code is locked in, ensure manuals match the engine.

| # | File | Notes |
|---|------|-------|
| 31 | `docs/DATA_DICTIONARY.md` | Update after dropped variables |
| 32 | `docs/METRICS_OUTLINE.md` | |
| 33 | `docs/SIMULATION_GUIDE.md` | |
| 34 | `docs/strategies/README.md` | |
| 35 | `docs/strategies/FATP_ANN_README.md` | |
| 36 | `docs/README.md` | |
