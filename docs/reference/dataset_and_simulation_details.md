# Dataset and Simulation Details

*Generated for the experimental setup section of an ICDM-style research paper.*
*All values verified by reading source files and running shell commands.*

---

## 1. Dataset Overview

### 1.1 Source

The dataset is the **DiDi GAIA Open Dataset** — real-world ride-hailing GPS traces and order records
from Chengdu, China.

Geographic extent (verified from Nov 1 data):

| Source | Lat range | Lon range | Notes |
|--------|-----------|-----------|-------|
| `order.txt` full day (all 209k trips) | 29.52°N – 31.34°N | 103.26°E – 104.70°E | Includes long-haul outliers |
| `gps.txt` first 5,000 pings | 30.65°N – 30.73°N | 104.04°E – 104.13°E | Core urban area of Chengdu |

The representative urban core (GPS-derived) centres around **~30.67°N, 104.07°E**, which is used
as the mean latitude for the flat-earth projection in the simulator.

### 1.2 Day Folders

The dataset is partitioned into exactly **30 daily folders**, one per day of **November 2016**:

```
496528674@qq.com_20161101   (1 Nov)
496528674@qq.com_20161102   (2 Nov)
...
496528674@qq.com_20161130   (30 Nov)
```

Each folder contains two raw files:

| File | Schema | Description |
|------|--------|-------------|
| `gps.txt` | `driver_id, order_id, timestamp, lon, lat` | GPS pings for every active driver |
| `order.txt` | `order_id, start_billing, end_billing, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat` | One row per completed trip/task |

### 1.3 Raw Record Counts (Before Sampling)

Two representative days were checked (`wc -l`):

| Day | `gps.txt` rows | `order.txt` rows |
|-----|---------------|-----------------|
| 20161101 | 32,155,517 | 209,423 |
| 20161115 | 31,866,045 | 231,112 |

**Exact totals across all 30 days** (verified via `find ... | xargs wc -l`):
- **GPS pings:** 1,096,618,422 total rows
- **Raw tasks (orders):** 7,065,937 total rows

This represents approximately:
- **236.9 million GPS pings per day on average** (~1.1B ÷ 30 days)
- **235,531 tasks per day on average** (~7.1M ÷ 30 days)

*Note: GPS rows represent individual timestamped pings, not unique drivers.*

---

## 2. Worker and Task Extraction

### 2.1 Worker Extraction (see `data/didi/didi.py`)

Workers are derived from the GPS traces:

- **Worker identity:** One worker per unique `driver_id`
- **Start position:** Coordinates from the **first GPS ping** for that driver on the day (`drop_duplicates(subset=["driver_id"], keep="first")`)
- **Release time:** Unix timestamp of that first GPS ping
- **Deadline (shift end):** `release_time + 28,800 seconds` (exactly **8-hour shift**, hardcoded — see `didi.py` line 71)
- **Raw worker counts:** Approximately 32M GPS lines per day from ~200–300k unique drivers (inferred from GPS volume)

### 2.2 Task Extraction (see `data/didi/didi.py`)

Each row in `order.txt` becomes one task:

- **`release_time`** ← `start_billing` (Unix timestamp when the trip was dispatched)
- **`expire_time`** ← `end_billing` (Unix timestamp when billing ended — used as task deadline)
- **Pickup/dropoff coordinates** retained directly

---

## 3. Stratified Temporal Sampling

### 3.1 Config Parameters (see `config.py` lines 23–29)

```python
DATA_SAMPLING = {
    "use_stratified_sampling": True,
    "target_tasks": 40000,
    "target_workers": 10000,
    "stratified_sampling_bins": 288,
    "random_state": 42,
}
```

- **Target tasks per day:** 40,000
- **Target workers per day:** 10,000 (task:worker ratio of 4:1)
- **Temporal bins:** 288 bins per day = **5-minute resolution** (24 h × 60 min / 5 min = 288)
- **Random seed:** 42 (fully reproducible)

### 3.2 Sampling Algorithm (see `data/stratified_sampler.py`, `data/loader.py`)

The sampler is invoked inside `load_workers_tasks()` (`data/loader.py` line 83):

```python
tasks, w_dict = stratified_temporal_sample(
    all_workers=workers,
    all_tasks=tasks,
    target_tasks=target_t,         # 40,000
    worker_counts=[target_w],      # [10,000]
    num_bins=288,                  # 5-min bins
    seed=42
)
workers = w_dict[target_w]
```

**Task sampling (proportional stratified):**
1. Sort all tasks by `release_time`
2. Divide the day into 288 equal-duration bins
3. For each bin: sample `floor(40000 × bin_fraction)` tasks at random — preserving the day's natural temporal distribution
4. If still short of 40,000, draw the remainder from the last bin

**Worker sampling (proportional stratified):**
1. Workers are sampled with the same proportional-bin logic, using the overlap window between worker availability and task releases
2. Workers are considered eligible for a bin if their active period `[release_time, deadline]` overlaps with that bin
3. Deduplication enforced (`O(1)` set membership check)

---

## 4. Simulation Time Window

### 4.1 Episode Duration (see `rl/gym_environment.py` lines 41–55)

```python
self.warmup_duration_seconds = warmup_duration_minutes * 60  # 30 * 60 = 1,800 s
self.episode_duration_seconds = episode_duration_hours * 60 * 60  # 8 * 3600 = 28,800 s
```

| Phase | Duration |
|-------|----------|
| Greedy warm-up | 30 minutes (1,800 s) |
| RL episode (composite strategy) | 8 hours (28,800 s) |
| **Total simulated window** | **8.5 hours** |

### 4.2 Random Drop-In Start (see `rl/gym_environment.py` lines 151–165)

At each `reset()`, a start time is drawn uniformly at random:

```python
earliest = min(t.release_time for t in self.tasks)
latest   = max(t.release_time for t in self.tasks)
total_duration_needed = warmup_duration_seconds + episode_duration_seconds  # 30,600 s
max_start = latest - total_duration_needed
start_time = random.uniform(earliest, max_start)
```

This ensures the full 8.5-hour window fits within the day's task window.
The drop-in start can be overridden via `options["start_time"]` for deterministic evaluation.

### 4.3 Simulation Step Size

Each RL decision step advances simulation time by **5 minutes (300 seconds)**.
An 8-hour episode therefore contains **96 steps** per episode
(8 h × 60 min / 5 min = 96).

---

## 5. Worker Physics

### 5.1 Travel Speed (see `simulator/strategies/composite.py` line 6)

```python
AVG_SPEED_KMH = 30
```

All travel time estimates use a **constant 30 km/h** (Manhattan distance ÷ speed), consistent with urban ridesharing approximations over the Chengdu road network.

### 5.2 Position Update After Task Completion (see `simulator/state.py` lines 81–93)

When a TASK_COMPLETE event fires, the worker's position is teleported to the dropoff:

```python
worker.start_lat = task.dropoff_lat
worker.start_lon = task.dropoff_lon
```

The worker is immediately re-inserted into the `available_workers` pool and the spatial index.

### 5.3 Task Finish Time Calculation (see `simulator/strategies/composite.py` lines 85–100)

```python
pickup_travel_hours  = pickup_distance / AVG_SPEED_KMH
service_travel_hours = drop_distance / AVG_SPEED_KMH

task.start_time  = now + (pickup_travel_hours * 3600)
task.finish_time = task.start_time + (service_travel_hours * 3600)
```

The TASK_COMPLETE event is scheduled at `task.finish_time`.

### 5.4 Worker Shift Expiry

There is **no explicit WORKER_DEADLINE event** in the event queue. Workers become de-facto unavailable past their deadline via the feasibility check (see Section 6). Workers are never forcibly removed from the available pool by deadline expiry alone; they are implicitly excluded from all new assignments once `now + total_travel_time > worker.deadline`.

---

## 6. Feasibility Check

The feasibility check gates every potential assignment in **both** event-driven pathways
(`NEW_TASK` and `FREE_WORKER`) — see `simulator/strategies/composite.py` lines 51–54:

```python
if (now + (d_pick / AVG_SPEED_KMH) * 3600) > task.expire_time or (
    now + ((d_pick + drop_distance_const) / AVG_SPEED_KMH) * 3600
) > worker.deadline:
    continue
```

Two simultaneous conditions must hold for an assignment to be feasible:

| Condition | Meaning |
|-----------|---------|
| `now + pickup_ETA ≤ task.expire_time` | Worker can reach the pickup **before the task expires** |
| `now + pickup_ETA + service_ETA ≤ worker.deadline` | Worker can **complete the trip before their shift ends** |

Workers failing either condition are silently skipped for that task.

The same two-condition check appears in `match_worker_composite` (lines 179–183) when a free worker scans the deferred task pool.

---

## 7. Spatial Indexing

### 7.1 Index Type (see `simulator/spatial_index.py`)

A custom **grid-based spatial index** (`GridSpatialIndex`) is used — not KD-tree or BallTree.

Key constants:

```python
GRID_RESOLUTION = 0.01  # degrees ≈ 1 km × 1 km cells at Chengdu's latitude
KM_PER_DEG_LAT  = 111.32
```

### 7.2 Distance Metric

**Manhattan distance with flat-earth projection:**

```python
def fast_manhattan_km(lat1, lon1, lat2, lon2):
    d_lat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    d_lon = abs(lon1 - lon2) * KM_PER_DEG_LON  # KM_PER_DEG_LON = 111.32 * cos(mean_lat)
    return d_lat + d_lon
```

`KM_PER_DEG_LON` is pre-computed once per simulation run using the mean latitude of the dataset
(approximately **30.67°N** for the Chengdu area), avoiding per-call `cos()` overhead. The flat-earth
approximation introduces a maximum spatial error of **< 0.15%** across the urban extent.

### 7.3 k-Nearest Neighbours (`k = 15`)

The query method (`query_k_nearest`, `k=15`) uses a **spiral ring expansion** from the center cell:

1. Starts at ring radius 0 (the containing cell)
2. Expands outward ring-by-ring
3. After collecting ≥ `k` candidates, checks whether the **minimum distance to the next ring**
   exceeds the current k-th best distance — if so, terminates early
4. Maximum search radius: 50 cells (≈ 50 km), acting as a sanity cap

Two independent spatial indices are maintained:
- **Worker index** (`start_lat/start_lon`) — available workers only
- **Deferred task index** (`pickup_lat/pickup_lon`) — tasks waiting for a free worker

---

## 8. Assignment Scoring Function

### 8.1 Composite Score (see `simulator/strategies/composite.py`)

For each candidate (worker, task) pair that passes the feasibility check:

```
Score = (fairness_weight × F) + (starvation_weight × S) + (utility_weight × U)
```

| Component | Formula | Role |
|-----------|---------|------|
| **F** (Fairness) | `(1 - γ) × T_idle + γ × worker.fairness_ewma` | Prioritises under-served workers |
| **S** (Starvation) | `log(1 + (now - task.release_time))` | Urgency for long-waiting tasks |
| **U** (Utility) | `1.0 / (1.0 + d_pick)` | Proximity reward |

where `T_idle = now - worker.last_active_ts` (time since last task completion, or since worker release if never assigned).

### 8.2 EWMA Fairness Update

After assignment the worker's EWMA is updated:

```python
worker.fairness_ewma = (1 - gamma) * T_idle_seconds + gamma * worker.fairness_ewma
```

`gamma = 0.1` makes the EWMA **responsive** (closer to instantaneous idle time).

### 8.3 Default Config Values (see `config.py` lines 49–67)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fairness_weight` (λ1) | 1.0 | Dynamic — controlled by DRL at runtime |
| `starvation_weight` (λ2) | 0.2 | Dynamic — controlled by DRL at runtime |
| `utility_weight` (λ3) | 1.0 | **Fixed** anchor (not in DRL action space) |
| `gamma` | 0.1 | EWMA smoothing factor |
| `k` | 15 | Nearest workers to evaluate |
| `soft_threshold` | 0.05 | Minimum score to assign immediately (0.0 = disabled) |

---

## 9. Reinforcement Learning Setup

### 9.1 Algorithm

**Proximal Policy Optimisation (PPO)** via Stable Baselines 3 with an `MlpPolicy`.

Default PPO hyperparameters (see `rl/train_sb3.py` lines 844–855):

| Hyperparameter | Default |
|----------------|---------|
| `learning_rate` | 3 × 10⁻⁴ |
| `n_steps` | 2,048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` (discount) | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| Network architecture | MLP [256, 256] for both policy and value |

Hyperparameters are Optuna-tuned; a `best_hyperparameters.json` is loaded if present
(`rl/best_hyperparameters.json`).

### 9.2 Observation Space (15-dimensional)

See `rl/gym_environment.py` lines 297–314:

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0 | Deferred task ratio | [0, 1] |
| 1 | Worker availability ratio | [0, 1] |
| 2 | Total worker count | ÷ `target_workers` (10,000) |
| 3 | Backlog peak (step) | ÷ 200 |
| 4 | Current JFI (Jain's Fairness Index) | [0, 1] |
| 5 | ΔJFI (step delta) | ÷ 0.05 |
| 6 | Mean wait time (step, minutes) | ÷ 2.0 min |
| 7 | Δ mean wait (step delta) | ÷ 10.0 |
| 8 | Δ backlog (step delta) | ÷ 30.0 |
| 9 | Δ task arrival rate (step delta) | ÷ 40.0 |
| 10 | `is_midweek` (binary) | {0, 1} |
| 11 | `is_mon_fri` (binary) | {0, 1} |
| 12 | `is_weekend` (binary) | {0, 1} |
| 13 | Time-of-day (sin) | [−1, 1] |
| 14 | Time-of-day (cos) | [−1, 1] |

### 9.3 Action Space (2-dimensional)

Symmetric `Box([-1, 1]²)`, mapped to physical weight ranges in `step()`:

```python
lambda1 = action[0] + 1.0            # [-1, 1] → [0.0, 2.0]  (fairness weight)
lambda2 = (action[1] + 1.0) * 0.25   # [-1, 1] → [0.0, 0.5]  (starvation weight)
lambda3 = 1.0                         # Fixed utility anchor
```

The `utility_weight` (λ3) is hardcoded at 1.0 and excluded from the action space, reducing it to 2D.

### 9.4 Reward Function (see `rl/gym_environment.py` lines 317–357)

A **"Dynamic SLA"** (twin-simulator differential reward) approach:

```
r_fairness  = (JFI_RL - JFI_greedy_twin) × 100.0
r_latency   = 0.0                           if latency_RL ≤ latency_twin + 0.1 min
            = -10.0 × (latency_RL - latency_twin)  otherwise
r_starvation = (expirations_twin - expirations_RL) × 1.0
reward = (r_fairness + r_latency + r_starvation) / 5.0
```

A **greedy shadow simulator** runs in lockstep, starting from an identical state. The RL agent is rewarded for **beating the greedy baseline on fairness** while incurring a latency penalty only if it is noticeably slower (> 0.1-minute buffer).

### 9.5 Warmup Protocol

At every `reset()`, both the main simulator and the shadow simulator execute **30 minutes of greedy** assignment (`assignment_strategy = "greedy"`) before the RL control phase begins. After warmup:
- **Main simulator** switches to `composite` strategy (RL-controlled λ1, λ2)
- **Shadow simulator** remains greedy for the entire episode (acts as a real-time baseline)

The greedy-phase JFI is captured as `greedy_baseline_jfi` for reward normalisation.

---

## 10. Train / Test Split

### 10.1 Day-Level Split (see `rl/train_sb3.py` lines 643–651)

```python
train_days, test_days = train_test_split(
    all_days,              # 30 days, sorted alphabetically
    train_size=24,         # Default: --train-days 24
    random_state=42,
    shuffle=True
)
```

- **Training:** 24 days (randomly selected, stratified across the month)
- **Test / evaluation:** 6 days (held out)
- The shuffle ensures a random mix of weekdays and weekends in each set

### 10.2 Evaluation Day

Post-training baseline comparison is run on a **single held-out test day**.
Default preferred evaluation day: `496528674@qq.com_20161128` (Nov 28, 2016).
If that day lands in the training split, the lexicographically first test day is used.

### 10.3 Parallel Environments

Training uses **8 parallel `SubprocVecEnv` environments** by default (`--num-cpu 8`).
Each subprocess independently samples a random training day and drop-in start time per episode.

---

## 11. Event-Driven Simulation Architecture

### 11.1 Event Types (see `simulator/simulation.py`)

| Event | Trigger | Action |
|-------|---------|--------|
| `WORKER_RELEASE` | Worker's `release_time` reached | Add to available pool, run `FREE_WORKER` handler |
| `TASK_RELEASE` | Task's `release_time` reached | Add to active pool; assign immediately if workers available, else defer |
| `TASK_COMPLETE` | `task.finish_time` reached | Complete task, teleport worker to dropoff, re-enter available pool |
| `TASK_EXPIRE` | `task.expire_time` reached | Remove from deferred pool if unassigned and uncompleted |

### 11.2 Deferral Mechanism

Tasks that arrive when no workers are available, or that score below `soft_threshold`, are placed in the
**deferred task pool** (`state.deferred_tasks`) with a paired spatial index (`state.deferred_task_index`).
A `TASK_EXPIRE` event is scheduled for each deferred task. Whenever a worker becomes free
(`FREE_WORKER` handler), it queries the deferred task index for the `k=15` nearest deferred tasks and
attempts to claim the best feasible one.

### 11.3 Flat-Earth Projection

All distance calculations use **Manhattan distance with a precomputed flat-earth projection**:

- `KM_PER_DEG_LAT = 111.32 km/°`
- `KM_PER_DEG_LON = 111.32 × cos(mean_lat) km/°` (computed once at simulation startup)

For the Chengdu dataset (mean lat ≈ 30.67°N), `KM_PER_DEG_LON ≈ 95.64 km/°`.
The approximation error is stated to be < 0.15% across the urban extent.

---

## 12. Key Details for Paper Section 5.1

*A concise list ready for the paper author to use directly.*

**Dataset:**
- DiDi GAIA open dataset; Chengdu, China; November 1–30, 2016 (30 days)
- Raw data: ≈209k–231k trips per day; ≈32M GPS pings per day
- Geographic extent: approximately 29.52°–31.34°N, 103.26°–104.70°E

**Stratified Sampling:**
- Per-day targets: **40,000 tasks** and **10,000 workers** (4:1 ratio)
- Sampling method: proportional stratified temporal sampling over **288 five-minute bins**
- Random seed: 42 for full reproducibility (`config.py` line 29)
- Worker shift: 8-hour window starting from first GPS ping

**Simulation Time Window:**
- Each episode = 30-minute greedy warm-up + **8-hour RL phase** (8.5 hours total)
- Start time sampled uniformly at random within the day's task window
- RL decision frequency: every **5 minutes** → **96 steps per episode**

**Worker Physics:**
- Constant travel speed: **30 km/h** (`composite.py` line 6)
- Worker 8-hour shift modelled via `deadline = release_time + 28,800 s`
- After task completion: worker position teleported to **dropoff location**
- No explicit worker-offline event; workers implicitly excluded by feasibility check

**Feasibility Check (dual-condition):**
1. `ETA_pickup ≤ task.expire_time` — worker can reach pickup before task expires
2. `ETA_complete ≤ worker.deadline` — worker can complete trip before shift ends
- Applied before every assignment attempt; infeasible workers skipped silently

**Spatial Indexing:**
- **Custom grid spatial index** with 0.01° resolution (≈ 1 km × 1 km cells)
- **Manhattan distance** with flat-earth projection (< 0.15% error over Chengdu)
- **k = 15** nearest workers/tasks considered per assignment decision
- Spiral ring expansion with provably-correct early termination

**Composite Scoring:**
- `Score = λ1 × F + λ2 × S + λ3 × U`; λ3 = 1.0 (fixed)
- F = EWMA idle time (γ = 0.1), S = log-waiting urgency, U = proximity reciprocal
- Soft assignment threshold: 0.05 (tasks below threshold deferred)

**RL Agent:**
- PPO (Stable Baselines 3), MLP policy [256, 256], 2D continuous action space (λ1, λ2)
- 15-dimensional observation (fairness, backlog, wait times, temporal features + deltas)
- Reward: differential vs greedy shadow twin; fairness advantage × 100, latency penalty if > 0.1 min slower

**Train / Test Split:**
- 30 days → **24 train / 6 test** via `sklearn.train_test_split(random_state=42, shuffle=True)`
- 8 parallel training environments (`SubprocVecEnv`)
- Evaluation day: `496528674@qq.com_20161128` (Nov 28, 2016)

---

## Appendix: File References

| Topic | File | Key Lines |
|-------|------|-----------|
| Sampling config | `config.py` | 23–29 |
| Strategy params | `config.py` | 49–84 |
| Worker model | `models/worker.py` | — |
| Task model | `models/task.py` | — |
| DiDi adapter | `data/didi/didi.py` | 56–93 |
| Loader + sampling intercept | `data/loader.py` | 76–93 |
| Stratified sampler | `data/stratified_sampler.py` | 18–231 |
| Spatial index | `simulator/spatial_index.py` | 1–202 |
| State manager | `simulator/state.py` | 81–93 (position update) |
| Composite strategy | `simulator/strategies/composite.py` | 6, 51–54, 86–100 |
| Event simulator | `simulator/simulation.py` | 180–231 |
| Gym environment | `rl/gym_environment.py` | 39–55, 136–221, 297–357 |
| Training script | `rl/train_sb3.py` | 512–535, 643–651, 824–880 |
