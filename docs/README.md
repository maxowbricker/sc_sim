# sc_sim — Spatial Crowdsourcing Simulator

A discrete-event simulation engine for studying fair task allocation in spatial crowdsourcing markets. The simulator supports multiple assignment strategies and is the experimental platform for the accompanying paper.

---

## Repository layout

```
sc_sim/
├── main.py                      # Entry point — run a single-strategy simulation
├── config.py                    # All configuration (read this first)
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
│   └── strategies/              # Assignment strategies (see docs/strategies.md)
│       ├── composite.py         # Composite: fairness + starvation + utility score
│       ├── greedy.py            # Greedy: nearest available worker
│       ├── knlf.py              # k-NLF: k-nearest, fewest-tasks-first
│       ├── laf.py               # LAF: least-allocated-first (global)
│       ├── fatp_ann.py          # FATP-ANN baseline (Tong et al.)
│       ├── aveklouris_lp.py     # Discrete Review LP (Aveklouris et al.)
│       ├── onrta_rt.py          # ONRTA-RT baseline
│       ├── biranking.py         # BiRanking baseline
│       └── ...                  # Additional baselines
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
│   ├── didi/didi.py             # DiDi GAIA dataset adapter
│   └── gowalla/gowalla.py       # Gowalla LBSN dataset adapter
│
├── scripts/
│   ├── experiments/             # Paper experiment scripts (§5.2–§5.4)
│   │   ├── s52_main_results/    # §5.2 main strategy comparison
│   │   ├── s53_scalability/     # §5.3 scalability sweeps
│   │   └── s54_ablation/        # §5.4 ablation studies
│   ├── plots/                   # Figure generation scripts
│   └── setup_didi_data.py       # One-time DiDi data extraction helper
│
├── results/                     # Experiment outputs — gitignored
│   ├── s52_main_results/        # §5.2 CSVs
│   ├── s53_scalability/         # §5.3 CSVs
│   ├── s54_ablation/            # §5.4 CSVs
│   └── figures/                 # Generated PDF/PNG figures
│
└── docs/
    ├── README.md                            # (this file)
    ├── strategies.md                        # Strategy reference (all algorithms)
    └── reference/
        ├── SIMULATION_GUIDE.md              # How to run simulations
        ├── DATA_DICTIONARY.md               # Complete metric and config reference
        ├── METRICS_OUTLINE.md               # Metric architecture and code map
        └── dataset_and_simulation_details.md  # DiDi GAIA dataset details
```

---

## Data setup (DiDi GAIA)

The simulator is built around the **DiDi GAIA** Chengdu dataset (~38k drivers, ~220k orders per day). The data is not included in the repo and must be obtained separately.

Expected layout:

```
data/didi/full_didi_gaia/
└── <day_folder>/          # e.g. 496528674@qq.com_20161128
    ├── order.txt          # orderID,startBillingTime,endBillableTime,pickupLon,pickupLat,dropoffLon,dropoffLat
    └── gps.txt            # driverID,orderID,timestamp,lon,lat
```

Timestamps are Unix seconds. Multiple day folders live under `full_didi_gaia/`.

### Extracting from zip

The data is distributed as a zip containing one `.tar.gz` per day:

```bash
unzip "滴滴 gaiya.zip" -d didi_raw
python3 scripts/setup_didi_data.py --source "didi_raw/滴滴 gaiya"

# Or extract a single day to get started quickly
python3 scripts/setup_didi_data.py --source "didi_raw/滴滴 gaiya" --day 20161128
```

---

## Installation

```bash
conda create -n sc python=3.10
conda activate sc
pip install numpy pandas scikit-learn scipy
```

---

## Running a single simulation

```bash
# Uses config.py defaults (strategy = composite, dataset = didi)
python main.py

# Override strategy or data path
python main.py --strategy greedy
python main.py --strategy knlf --root data/didi/full_didi_gaia/496528674@qq.com_20161128
```

Key configuration knobs in `config.py`:

| Setting | Where | What it does |
|---------|-------|-------------|
| `assignment_strategy` | `SIMULATION_CONFIG` | Which strategy to use |
| `use_stratified_sampling` | `DATA_SAMPLING` | Sample ~40k tasks / 10k workers from full day |
| `fairness_weight` (λ_f) | `STRATEGY_PARAMS["composite"]` | Composite fairness weight |
| `starvation_weight` (λ_s) | `STRATEGY_PARAMS["composite"]` | Composite starvation weight |
| `gamma` | `STRATEGY_PARAMS["composite"]` | EWMA decay (0.1 = responsive) |
| `k` | `STRATEGY_PARAMS["composite"]` | k-nearest workers considered per task |

See `docs/reference/DATA_DICTIONARY.md` for the full config reference.

---

## Running paper experiments

All paper experiment scripts live under `scripts/experiments/`. Each subdirectory has its own README:

| Section | Script directory | Output directory |
|---------|-----------------|-----------------|
| §5.2 Main comparison | `scripts/experiments/s52_main_results/` | `results/s52_main_results/` |
| §5.3 Scalability | `scripts/experiments/s53_scalability/` | `results/s53_scalability/` |
| §5.4 Ablation | `scripts/experiments/s54_ablation/` | `results/s54_ablation/` |

Example:

```bash
# §5.2 DiDi strategy comparison
python3 scripts/experiments/s52_main_results/run_strategy_comparison.py --day 20161109

# §5.2 Gowalla comparison
python3 scripts/experiments/s52_main_results/run_gowalla_comparison.py
```

See `PAPER_PROVENANCE.md` at the repo root for a complete mapping of paper tables and figures to their source CSV files.

---

## Key reference docs

| Document | Contents |
|----------|----------|
| `docs/strategies.md` | All strategies: algorithm summary, config params, complexity |
| `docs/reference/SIMULATION_GUIDE.md` | Detailed how-to: config knobs, calling the sim API |
| `docs/reference/DATA_DICTIONARY.md` | Every metric, config flag, and result key defined |
| `docs/reference/METRICS_OUTLINE.md` | How metrics flow through the code |
| `docs/reference/dataset_and_simulation_details.md` | DiDi GAIA dataset details and design choices |
| `PAPER_PROVENANCE.md` | Per-table CSV provenance map and result file inventory |
