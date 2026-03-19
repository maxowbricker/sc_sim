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
    "data_root_path": None,                     # Override data directory (None = auto)
    "assignment_strategy": "composite",         # "greedy" | "composite" | "fatp_ann" | "ewma_only" | "random_assign"
}

# ============================================================================
# DATA SAMPLING PARAMETERS
# ============================================================================

DATA_SAMPLING = {
    "use_stratified_sampling": False,
    "target_tasks": 5000,                   # Shrink from 200k to 5k
    "target_workers": 1250,                 # Keep a 1:4 ratio of workers to tasks
    "stratified_sampling_bins": 12,         
    "random_state": 42,                     
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
        "gamma": 0.1,                           # EWMA smoothing factor (0.1=responsive, 0.9=smooth)
        
        # Assignment mechanism
        "k": 15,                                # Number of nearest workers to consider
        "soft_threshold": 0.05,                  # Minimum score to assign immediately (0.0 = disabled)
        
        # Diagnostic Trackers
        "enable_diagnostics": False,            # Enable heavy evaluation metrics (IOR, Fairness Loss) - DISABLE FOR RL
        "enable_deferral_tracking": False,      # Track O(1) task deferral statistics for RQ3.3
    },
    
    # === BASELINE STRATEGIES ===
    "ewma_only": {
        "gamma": 0.3,                           # EWMA smoothing factor 
    },
    
    "random_assign": {
        "k": 15,                                # Number of nearest workers to consider for random selection
    },
    
    "fatp_ann": {
        "mu": 0.5,                              # Decay factor for utility calculation 
        "alpha_scale": 0.5,                     # Scaling factor for base utility (task distance)
        "use_k_nearest": False,                 # Use full worker scan (k-NN optimization disabled)
        "k": 15,                                # Number of nearest workers (only used if use_k_nearest=True)
    }
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