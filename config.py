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
    "dataset": "didi",                          # "didi" | "synthetic"
    "data_root_path": None,                     # Override data directory. Example: "data/didi/full_didi_gaia/496528674@qq.com_20161128". None = use default from DATA_SAMPLING
    "assignment_strategy": "composite",         # "greedy" | "composite" | "fatp_ann" | "ewma_only" | "random_assign" | "mmd_batch" | "cost_balancing" | "tsgf" | "discrete_review_lp" | "onrta_op"
}

# ============================================================================
# DATA SAMPLING PARAMETERS
# ============================================================================

DATA_SAMPLING = {
    "use_stratified_sampling": True,
    "target_tasks": 40000,
    "target_workers": 10000,
    "stratified_sampling_bins": 288,         
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

# Platform task revenue (Basık et al.): t_j.m = base_fare + per_km_rate × α
# α = pickup→dropoff distance (km). Independent of composite-strategy U = 1/(1+d_pick).
PLATFORM_REVENUE = {
    "base_fare": 2.00,
    "per_km_rate": 1.50,
}

# Worker stochastic acceptance (Basık et al.): P(accept) = exp(-d_pick) * c
# I implemented this to simulate worker task rejection however, this didn't work well but potentially is needed to improve the strength of the paper.
# The function is very severe, I found with the Didi dataset the task rejection rate was too high (95%), the paper which used this function used a smaller scale for their dataset.
WORKER_ACCEPTANCE = {
    "enabled": False,       # Off by default — RL training unchanged
    "c_willingness": 0.6,   # Basık willingness constant
    "seed": 42,               # Dedicated RNG seed for reproducible acceptance rolls
}

# ============================================================================
# STRATEGY-SPECIFIC PARAMETERS
# ============================================================================

STRATEGY_PARAMS = {
    # === COMPOSITE STRATEGY (DRL Target) ===
    "composite": {
        # Weights for scoring function: Score = (fairness_weight × F) + (starvation_weight × S) + (1.0 × U)
        "fairness_weight": 1.0,                 # Dynamic parameter controlled by DRL
        "starvation_weight": 0.2,               # Dynamic parameter controlled by DRL
        "utility_weight": 1.0,                  # HARDCODED: Anchors the DRL action space
        
        # EWMA fairness calculation
        "gamma": 0.1,                           # EWMA smoothing factor (0.1=responsive, 0.9=smooth), (0.1/0.15 was found to be optimal for a single days simulation)
        
        # Assignment mechanism
        "k": 15,                                # Number of nearest workers to consider
        "soft_threshold": 0.05,                  # Minimum score to assign immediately (0.0 = disabled)
        
        # Diagnostic Trackers
        "enable_diagnostics": False,            # Enable heavy evaluation metrics (IOR, Fairness Loss) - DISABLE FOR RL
        "enable_deferral_tracking": False,      # Track O(1) task deferral statistics for RQ3.3

        # Stochastic worker acceptance (Basık cascade dispatch)
        "worker_acceptance": dict(WORKER_ACCEPTANCE),
    },

    # === GREEDY BASELINE ===
    "greedy": {
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
        "mu": 0.5,                              # Decay factor for utility calculation 
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
        "review_period_seconds": 60.0,          # Review interval l (seconds); sweep for Pareto curves
    },

    # === ONRTA-OP BASELINE ===
    "onrta_op": {
        # Expected market size for phase transition; reset() defaults to len(tasks/workers)
        "expected_a": None,
        "expected_b": None,
    },
}

# ============================================================================
# UTILITY FUNCTIONS FOR CONFIG ACCESS
# ============================================================================

def get_simulation_config():
    """Get core simulation configuration."""
    return SIMULATION_CONFIG.copy()

def get_strategy_params(strategy_name=None):
    """Get parameters for specific strategy or current default."""
    if strategy_name is None:
        strategy_name = SIMULATION_CONFIG["assignment_strategy"]
    return STRATEGY_PARAMS.get(strategy_name, {}).copy()

def get_data_sampling_config():
    """Get parameters for data loading and sampling."""
    return DATA_SAMPLING.copy()

def create_composite_config(**overrides):
    """
    Create complete simulation configuration with optional overrides.
    Used by DRL agents to inject dynamic weights into the environment.
    """
    config = get_simulation_config()
    strategy_name = overrides.get("assignment_strategy", config["assignment_strategy"])
    
    strategy_params = get_strategy_params(strategy_name)
    
    for key, value in overrides.items():
        if key in config:
            config[key] = value
        elif key in strategy_params:
            strategy_params[key] = value
            
    config["strategy_params"] = strategy_params
    return config

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

def get_observation_static_scaling() -> Dict[str, Any]:
    """Full observation scaling dict for the RL env (refs, deltas, worker divisor)."""
    out = dict(OBSERVATION_STATIC_SCALING)
    out.setdefault(
        "worker_count_divisor",
        float(max(DATA_SAMPLING.get("target_workers", 10000), 1)),
    )
    return out

def get_platform_revenue_config() -> Dict[str, Any]:
    """Fare model for intrinsic task revenue: base_fare + per_km_rate × α."""
    return PLATFORM_REVENUE.copy()


def get_worker_acceptance_config() -> Dict[str, Any]:
    """Stochastic worker acceptance parameters (Basık et al.)."""
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
                f"❌ Config Error: Unknown parameter '{key}'. "
                f"It does not exist in SIMULATION_CONFIG or STRATEGY_PARAMS['{strategy_name}']."
            )
            
    config["strategy_params"] = strategy_params
    return config