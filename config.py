# ============================================================================
# SPATIAL CROWDSOURCING SIMULATOR - CORE CONFIGURATION
# ============================================================================
"""
Central configuration file for the simulation engine.
Strictly contains active parameters used by the physics engine and strategies.
"""

# ============================================================================
# CORE SIMULATION SETTINGS
# ============================================================================

SIMULATION_CONFIG = {
    "dataset": "didi",                          # "didi" | "nyc_taxi" | "gowalla" | "synthetic"
    "data_root_path": None,                     # Override data directory. Example: "data/didi/full_didi_gaia/496528674@qq.com_20161128". None = use default from DATA_SAMPLING
    "assignment_strategy": "composite",         # "greedy" | "composite" | "fatp_ann" | "ewma_only" | "random_assign" | "mmd_batch" | "cost_balancing" | "tsgf" | "discrete_review_lp" | "onrta_op" | "onrta_rt" | "biranking"
}

# ============================================================================
# DATA SAMPLING PARAMETERS
# ============================================================================

DATA_SAMPLING = {
    "use_stratified_sampling": False,
    "target_tasks": 40000,
    "target_workers": 10000,
    "stratified_sampling_bins": 288,         
    "random_state": 42,                     
}

# ============================================================================
# NYC TAXI DATASET PARAMETERS
# ============================================================================

NYC_TAXI_CONFIG = {
    # ISO date string to load a single day (e.g. "2012-05-01").
    # None loads the entire parquet file (full month).
    "date": None,

    # Worker bootstrapping mode.
    # True  -> fleet size = round(num_tasks * workers_per_task_ratio)
    # False -> fixed fleet of num_workers, regardless of task count
    "use_proportional_workers": True,
    "workers_per_task_ratio": 0.2,          # ~1 worker per 5 tasks
    "num_workers": 5000,                    # used when use_proportional_workers=False

    "random_state": 42,
}

# ============================================================================
# GOWALLA DATASET PARAMETERS
# ============================================================================

GOWALLA_CONFIG = {
    # Named city preset: "austin" (default, densest cluster) | "san_francisco"
    # Set to None and supply a custom bbox tuple to override.
    "region": "austin",

    # Custom bounding box (lat_min, lat_max, lon_min, lon_max). Overrides region.
    "bbox": None,

    # ISO date range filter. None = use all available data.
    "date_start": None,
    "date_end":   None,

    # Task generation mode:
    #   "checkin"       - every check-in = 1 task (recommended, realistic ratio)
    #   "location_pair" - paper-faithful: same-location pairs per day (inverted ratio)
    "task_mode": "checkin",

    # Expiry window for "checkin" mode tasks (ignored in "location_pair" mode).
    # Set to match Didi's median trip duration (~19.5 min) / p90 (~38 min).
    # 0.5 hours (30 min) sits between these, giving a comparable task availability
    # window to the real trip-end expiry used in the Didi and NYC adapters.
    "task_window_hours": 0.5,

    # Worker shift duration in hours (deadline = release_time + shift_hours * 3600)
    "shift_hours": 8.0,

    # Std-dev radius (km) for synthetic dropoff displacement from pickup
    "dropoff_noise_km": 2.0,

    # Workers = round(n_tasks * ratio). Set to None to keep all (user, day) workers.
    "workers_per_task_ratio": 0.2,    # 1:5 ratio — tune between 0.1 (1:10) and 0.33 (1:3)

    # Compress all check-ins from the selected date range onto a single 24-hour
    # reference day (strips calendar date, keeps HH:MM:SS). This is strongly
    # recommended when date_start/date_end span multiple days: LBSN check-in
    # rates are ~150x lower than ride-hailing, so without compression only
    # ~31 tasks are active at any given moment — too sparse for meaningful
    # strategy differentiation.  With compression a 1-month window produces
    # ~900 concurrent tasks, comparable to a stratified-sampled Didi day.
    "compress_to_day": True,

    "random_state": 42,
}


# These values are for the scaling for a variety of reward functions that I have tried, potentially not useful anymore.
OBSERVATION_STATIC_SCALING = {
    "ref_wait_minutes": 2.0,
    "ref_backlog": 200,
    "max_abs_jfi_delta": 0.05,
    "max_abs_arrival_delta": 40.0,
    "max_abs_wait_delta": 10.0,
    "max_abs_backlog_delta": 30.0,
}

# Platform task revenue (Basik et al.): t_j.m = base_fare + per_km_rate * alpha
# alpha = pickup->dropoff distance (km). Independent of composite-strategy U = 1/(1+d_pick).
PLATFORM_REVENUE = {
    "base_fare": 2.00,
    "per_km_rate": 1.50,
}

# Worker stochastic acceptance (Basik et al.): P(accept) = exp(-d_pick) * c
# I implemented this to simulate worker task rejection however, this didn't work well but potentially is needed to improve the strength of the paper.
# The function is very severe, I found with the Didi dataset the task rejection rate was too high (95%), the paper which used this function used a smaller scale for their dataset.
WORKER_ACCEPTANCE = {
    "enabled": False,       # Off by default - RL training unchanged
    "c_willingness": 0.6,   # Basik willingness constant
    "seed": 42,               # Dedicated RNG seed for reproducible acceptance rolls
}

# ============================================================================
# STRATEGY-SPECIFIC PARAMETERS
# ============================================================================

STRATEGY_PARAMS = {
    # === COMPOSITE STRATEGY (DRL Target) ===
    "composite": {
        # Weights for scoring function: Score = (fairness_weight * F) + (starvation_weight * S) + (1.0 * U)
        # fw=1.6, sw=0.0 confirmed as Pareto-optimal for pure-fairness operation on Didi 20161109
        # (best JFI among all sw=0.0 configs; Pareto-dominates sw>0 on wait time at similar JFI).
        "fairness_weight": 1.6,                 # Static weight for paper experiments (Pareto sweep confirmed)
        "starvation_weight": 0.0,               # Disabled: ablation showed sw hurts idle-time equity on Didi
        "utility_weight": 1.0,                  # HARDCODED: Anchors the DRL action space

        # EWMA fairness calculation
        "gamma": 0.1,                           # EWMA smoothing factor (0.1=responsive, 0.9=smooth), (0.1/0.15 was found to be optimal for a single days simulation)
        
        # Assignment mechanism
        "k": 15,                                # Number of nearest workers to consider
        "soft_threshold": 0.0,                   # Disabled: sensitivity test showed negligible effect; cleaner JFI
        
        # Diagnostic Trackers
        "enable_diagnostics": False,            # Enable heavy evaluation metrics (IOR, Fairness Loss) - DISABLE FOR RL
        "enable_deferral_tracking": True,       # Track O(1) task deferral statistics

        # Stochastic worker acceptance (Basik cascade dispatch)
        "worker_acceptance": dict(WORKER_ACCEPTANCE),
    },

    # === GREEDY BASELINE ===
    "greedy": {
        "k": 10,                                # k nearest workers queried via spatial index
        "worker_acceptance": dict(WORKER_ACCEPTANCE),
    },
    
    # === BASELINE STRATEGIES ===
    "ewma_only": {
        "gamma": 0.2,                           # EWMA smoothing factor 
    },
    
    "random_assign": {
        "k": 15,                                # Number of nearest workers to consider for random selection
    },
    
    "fatp_ann": {
        # mu calibrated so utility retains ~50% at the dataset mean task completion time.
        # Formula: mu = ln(2) / T_mean_hours.
        # Didi  (T_mean ≈ 23.6 min = 0.39h) → mu ≈ 1.76
        # Gowalla (T_mean ≈ 15.5 min = 0.26h) → mu ≈ 2.68
        # Shared value mu=1.5 (50% at ~27.7 min) used across both datasets for
        # cross-dataset generalisation, consistent with Composite's fixed-weight policy.
        # With the previous mu=0.5, the utility range across all feasible tasks was
        # only ~16% (flat); mu=1.5 gives a ~28% spread, providing real discriminating power.
        "mu": 1.5,
        "alpha_scale": 0.5,                     # Scaling factor for base utility (task distance)
        "use_k_nearest": False,                 # Use full worker scan (k-NN optimization disabled)
        "k": 15,                                # Number of nearest workers (only used if use_k_nearest=True)
    },

    # === COST-BALANCING BASELINE ===
    "cost_balancing": {
        "alpha": 0.5,                           # Match when M <= alpha * W (paper delivery experiments)
        "k": 10,                                # k-NN window for batch greedy matching
    },

    # === TSGF RANDOMIZED SAMPLING BASELINE ===
    "tsgf": {
        "alpha": 0.4,                           # P(operator profit / proximity greedy)
        "beta": 0.3,                            # P(max-min worker fairness)
        "gamma": 0.3,                           # P(max-min task fairness); alpha+beta+gamma <= 1
        "k": 15,                                # k-NN window for spatial candidate search
        "seed": 42,                             # RNG seed for reproducible policy sampling
    },

    # === DISCRETE REVIEW LP BASELINE (Aveklouris et al.) ===
    "discrete_review_lp": {
        "review_period_seconds": 15.0,          # Review interval l (seconds); confirmed optimal by Gowalla sweep (JFI peaks at 15s > 5s/10s/30s)
    },

    # === ONRTA-OP BASELINE ===
    "onrta_op": {
        # Phase transition: expected_a=|R|, expected_b=sum(w.c); reset() fills from dataset
        "expected_a": None,
        "expected_b": None,
    },

    # === ONRTA-RT BASELINE ===
    "onrta_rt": {
        "seed": 42,                             # RNG seed for theta draw and candidate sampling
    },

    # === BIPARTITE RANKING (BRK) BASELINE ===
    "biranking": {
        "seed": 42,                             # RNG seed for permanent entity ranks
    },

    # === k-NEAREST LEAST-FIRST (k-NLF) ===
    # O(k) fairness signal: query k nearest workers, assign to the one with fewest
    # completed tasks (distance used only as tie-breaker).
    "knlf": {
        "k": 15,                                # Candidate pool size; sweep k ∈ {3,5,10,15} for ablation
    },

    # === k-NEAREST TEMPORAL FAIRNESS — ECONOMIC (k-NTF-EPH) ===
    # Addresses the "Billy vs John" flaw in k-NLF: instead of raw task counts,
    # sorts the k-nearest candidate pool by ascending Earnings Per Hour.
    # The worker who has earned the least relative to their time online wins.
    "kntf_eph": {
        "k": 15,
    },

    # === k-NEAREST TEMPORAL FAIRNESS — IDLE RATIO (k-NTF-IR) ===
    # Time-normalised idle time: sorts k-nearest candidates by descending
    # Idle Ratio (fraction of shift spent waiting). Corrects the EWMA
    # scalar-trap problem by applying the signal structurally, not as a weight.
    "kntf_ir": {
        "k": 15,
    },
}

# ============================================================================
# UTILITY FUNCTIONS FOR CONFIG ACCESS
# ============================================================================
from typing import Dict, Any, Optional

def get_simulation_config() -> Dict[str, Any]:
    """Get core simulation configuration."""
    return SIMULATION_CONFIG.copy()

def get_strategy_params(strategy_name: Optional[str] = None) -> Dict[str, Any]:
    """Get parameters for specific strategy or current default."""
    if strategy_name is None:
        strategy_name = SIMULATION_CONFIG["assignment_strategy"]
    return STRATEGY_PARAMS.get(strategy_name, {}).copy()

def get_data_sampling_config() -> Dict[str, Any]:
    """Get parameters for data loading and sampling."""
    return DATA_SAMPLING.copy()

def get_nyc_taxi_config() -> Dict[str, Any]:
    """Get parameters for the NYC TLC Yellow Taxi adapter."""
    return NYC_TAXI_CONFIG.copy()

def get_gowalla_config() -> Dict[str, Any]:
    """Get parameters for the Gowalla LBSN adapter."""
    return GOWALLA_CONFIG.copy()

def get_observation_static_scaling() -> Dict[str, Any]:
    """Full observation scaling dict for the RL env (refs, deltas, worker divisor)."""
    out = dict(OBSERVATION_STATIC_SCALING)
    out.setdefault(
        "worker_count_divisor",
        float(max(DATA_SAMPLING.get("target_workers", 10000), 1)),
    )
    return out

def get_platform_revenue_config() -> Dict[str, Any]:
    """Fare model for intrinsic task revenue: base_fare + per_km_rate * alpha."""
    return PLATFORM_REVENUE.copy()


def get_worker_acceptance_config() -> Dict[str, Any]:
    """Stochastic worker acceptance parameters (Basik et al.)."""
    return WORKER_ACCEPTANCE.copy()


def get_rl_reward_config() -> Dict[str, Any]:
    """RL reward coefficients and observation flags (Trial D / D1)."""
    return RL_REWARD.copy()

def create_composite_config(**overrides: Any) -> Dict[str, Any]:
    """
    Create complete simulation configuration with optional overrides.
    Used by DRL agents to inject dynamic weights into the environment.
    
    Raises:
        ValueError: If an override key does not exist in the default configurations.
    """
    config = get_simulation_config()
    strategy_name = overrides.get("assignment_strategy", config["assignment_strategy"])
    
    strategy_params = get_strategy_params(strategy_name)
    
    for key, value in overrides.items():
        if key in config:
            config[key] = value
        elif key in strategy_params:
            strategy_params[key] = value
        else:
            # FAIL LOUDLY: Prevent silent typos from ruining DRL training
            raise ValueError(
                f"Config Error: Unknown parameter '{key}'. "
                f"It does not exist in SIMULATION_CONFIG or STRATEGY_PARAMS['{strategy_name}']."
            )
            
    config["strategy_params"] = strategy_params
    return config