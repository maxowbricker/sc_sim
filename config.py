# ============================================================================
# SPATIAL CROWDSOURCING SIMULATOR - COMPREHENSIVE CONFIGURATION
# ============================================================================
"""
Central configuration file for all simulation components.
Eliminates parameter duplication across scripts and provides
organized configuration sections for different simulation aspects.
"""

# ============================================================================
# CORE SIMULATION SETTINGS
# ============================================================================

SIMULATION_CONFIG = {
    # Dataset and execution
    "dataset": "didi",                          # "didi" | "synthetic" | "checkins"
    "data_root_path": None,                     # Override data directory (None = auto)

    # Assignment strategy
    "assignment_strategy": "composite",         # "greedy" | "composite" | "fatp"
    
    # Execution behavior  
    "service_time_mode": "distance",            # How to calculate service times
    "teleport_on_complete": True,               # Worker teleports to dropoff instantly
}

# ============================================================================
# STRATEGY-SPECIFIC PARAMETERS
# ============================================================================

STRATEGY_PARAMS = {
    # === COMPOSITE STRATEGY (Your Main Contribution) ===
    "composite": {
        # Weights for scoring function: Score = fairness_weight×Fairness + starvation_weight×Starvation + utility_weight×Utility
        "fairness_weight": 1.0,                 # Fairness weight (formerly λ1)
        "starvation_weight": 1.0,               # Starvation weight (formerly λ2)
        "utility_weight": 0.5,                  # Utility weight (formerly λ3)
        
        # EWMA fairness calculation
        "gamma": 0.3,                           # EWMA smoothing factor (0.1=responsive, 0.9=smooth)
        
        # Assignment mechanism
        "k": 15,                                # Number of nearest workers to consider
        "soft_threshold": 0.5,                  # Minimum score to assign immediately
        
        # Fairness metric selection
        "fairness_metric": "ewma",              # "ewma" | "idle_time" | "task_count"
                                                # RESEARCH PROPOSAL: Use EWMA as per methodology
        
        # EXPERIMENT 008: Score Normalization and Threshold Ablation
        # These flags enable diagnostic experiments to understand the worker idle time paradox
        "normalize_scores": False,              # Enable min-max normalization of F, S, U components
                                                # Set to True to test if mis-scaled components cause the paradox
        "disable_soft_threshold": False,        # Bypass soft threshold check (always assign if worker exists)
                                                # Set to True to test if threshold delays cause the paradox
        "enable_diagnostics": False,            # Enable detailed diagnostic tracking (component dominance, deferrals)
                                                # WARNING: Has performance impact, only enable when needed
    },
    
    # === OTHER STRATEGIES ===
    "greedy": {
        # No additional parameters - uses pure distance
    },
    
    "fatp": {
        # Fair and efficient baseline parameters (if implemented)
    }
}

# ============================================================================
# EXPERIMENT CONFIGURATION PRESETS
# ============================================================================

EXPERIMENT_PRESETS = {
    # === PARAMETER SENSITIVITY ANALYSIS ===
    "parameter_ranges": {
        "lambda_sweep": {
            "fairness_weight": [0.5, 1.0, 1.5, 2.0],        # Fairness focus range
            "starvation_weight": [0.5, 1.0, 1.5, 2.0],      # Starvation focus range
            "utility_weight": [0.3, 0.5, 1.0, 1.5],         # Utility focus range
            "soft_threshold": [1.0],                         # Keep constant
            "gamma": [0.3],                                  # Keep constant
        },
        
        "threshold_sweep": {
            "fairness_weight": [1.0], "starvation_weight": [1.0], "utility_weight": [0.5],  # Keep weights constant
            "soft_threshold": [0.0, 0.5, 1.0, 1.5, 2.0],  # Realistic threshold range
            "gamma": [0.3],
        },
        
        "gamma_sweep": {
            "fairness_weight": [1.0], "starvation_weight": [1.0], "utility_weight": [0.5],  # Keep weights constant
            "soft_threshold": [1.0],                   # Keep threshold constant
            "gamma": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9], # EWMA responsiveness range
        },
        
        "focused_comparison": {
            "fairness_weight": [0.5, 1.0, 2.0],     # Low, medium, high fairness
            "starvation_weight": [1.0],              # Standard starvation
            "utility_weight": [0.5, 1.0],           # Medium vs high utility
            "soft_threshold": [0.5, 1.0, 1.5],      # Permissive, medium, strict
            "gamma": [0.3],                          # Standard EWMA
        }
    },
    
    # === QUICK EXPERIMENT CONFIGURATIONS ===
    "quick_test_configs": [
        # Baseline
        {"name": "Baseline (Current)", "params": {}},
        
        # Fairness-focused configurations
        {"name": "High Fairness Focus", "params": {"fairness_weight": 2.0, "starvation_weight": 1.0, "utility_weight": 0.3}},
        {"name": "Very High Fairness", "params": {"fairness_weight": 3.0, "starvation_weight": 1.0, "utility_weight": 0.2}},
        
        # Efficiency-focused configurations  
        {"name": "High Efficiency Focus", "params": {"fairness_weight": 0.3, "starvation_weight": 1.0, "utility_weight": 2.0}},
        {"name": "Very High Efficiency", "params": {"fairness_weight": 0.2, "starvation_weight": 0.5, "utility_weight": 3.0}},
        
        # Starvation-focused
        {"name": "High Starvation Prevention", "params": {"fairness_weight": 1.0, "starvation_weight": 3.0, "utility_weight": 0.5}},
        
        # Balanced approaches
        {"name": "Balanced Equal Weights", "params": {"fairness_weight": 1.0, "starvation_weight": 1.0, "utility_weight": 1.0}},
        {"name": "Balanced with Fairness Bias", "params": {"fairness_weight": 1.5, "starvation_weight": 1.0, "utility_weight": 0.7}},
        
        # Threshold experiments
        {"name": "Permissive Threshold", "params": {"soft_threshold": 0.5}},
        {"name": "Strict Threshold", "params": {"soft_threshold": 2.0}},
        
        # EWMA sensitivity
        {"name": "Responsive EWMA", "params": {"gamma": 0.1}},
        {"name": "Smooth EWMA", "params": {"gamma": 0.7}},
    ],
    
    # === BENCHMARK COMPARISON SETTINGS ===
    "benchmark": {
        "strategies": ["greedy", "composite"],   # Strategies to compare
        "num_runs": 1,                          # Runs per strategy for statistical significance
        "key_metrics": [                        # Primary evaluation metrics
            "completed_tasks", "final_jains_fairness_index", 
            "final_utility_difference_tasks", "final_fairness_loss", 
            "final_ewma_cv", "backlog_peak", "total_travel_km"
        ]
    }
}

# ============================================================================
# REINFORCEMENT LEARNING CONFIGURATION
# ============================================================================

RL_CONFIG = {
    # === PPO TRAINING PARAMETERS ===
    "ppo": {
        "learning_rate": 3e-4,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        
        # Network architecture
        "hidden_sizes": [64, 64],
        "activation": "tanh",
    },
    
    # === ENVIRONMENT SETTINGS ===
    "environment": {
        "episode_length": 50,                   # Number of decision points per episode
        "decision_interval": 300,               # Seconds between weight updates
        "reward_weights": {                     # Reward function: R = β₁×F + β₂×S + β₃×U
            "β1": 1.0,                          # Fairness reward weight
            "β2": 0.5,                          # Starvation penalty weight  
            "β3": 0.3,                          # Utility reward weight
        },
        "normalization_window": 100,            # Rolling window for metric normalization
    },
    
    # === TRAINING SETTINGS ===
    "training": {
        "num_episodes": 100,                    # Total training episodes
        "update_frequency": 10,                 # Update policy every N steps
        "save_frequency": 25,                   # Save model every N episodes
        "evaluation_frequency": 10,             # Evaluate performance every N episodes
        "early_stopping_patience": 20,         # Stop if no improvement for N episodes
    }
}

# ============================================================================
# FAIRNESS METRICS CONFIGURATION  
# ============================================================================

FAIRNESS_CONFIG = {
    # Available fairness metrics for comparison
    "available_metrics": ["ewma", "utility_difference", "jains", "fairness_loss"],
    
    # Metric-specific parameters
    "metric_params": {
        "ewma": {
            "gamma": 0.3,                       # Smoothing factor
            "signal_type": "idle_time",         # "idle_time" | "task_count" | "utility" | "recency"
        },
        "utility_difference": {
            "normalization": "max",             # How to normalize utility differences
        },
        "jains": {
            "window_size": 100,                 # Rolling window for fairness calculation
        },
        "fairness_loss": {
            "baseline": "equal_distribution",   # Fairness baseline for loss calculation
        }
    },
    
    # Fairness tracking settings
    "tracking": {
        "snapshot_frequency": 100,              # Record fairness every N events
        "enable_time_series": True,             # Collect fairness over time
        "enable_worker_analysis": True,         # Per-worker fairness breakdown
    }
}

# ============================================================================
# OUTPUT AND LOGGING CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    # Results directory structure
    "base_output_dir": "results",
    "subdirs": {
        "experiments": "experiments",
        "benchmarks": "benchmarks", 
        "rl_training": "rl_training",
        "fairness_analysis": "fairness_analysis",
        "plots": "plots"
    },
    
    # File formats and naming
    "formats": {
        "results": "json",                      # Experiment results format
        "plots": "png",                         # Plot output format
        "logs": "txt",                          # Log file format
    },
    
    # Logging levels
    "verbosity": {
        "simulation": "INFO",                   # Simulation event logging
        "strategy": "DEBUG",                    # Strategy decision logging
        "rl_training": "INFO",                  # RL training progress
        "fairness": "INFO",                     # Fairness calculation logging
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

def get_experiment_preset(preset_name):
    """Get predefined experiment configuration."""
    return EXPERIMENT_PRESETS.get(preset_name, {})

def get_rl_config():
    """Get reinforcement learning configuration."""
    return RL_CONFIG.copy()

def get_fairness_config():
    """Get fairness metrics configuration."""
    return FAIRNESS_CONFIG.copy()

def create_composite_config(**overrides):
    """
    Create complete simulation configuration with optional overrides.
    
    Usage:
        config = create_composite_config(
            assignment_strategy="composite",
            fairness_weight=2.0, starvation_weight=1.0, utility_weight=0.3,
            soft_threshold=1.5
        )
    """
    config = get_simulation_config()
    strategy_name = overrides.get("assignment_strategy", config["assignment_strategy"])
    
    # Get strategy-specific parameters
    strategy_params = get_strategy_params(strategy_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config:
            config[key] = value
        elif key in strategy_params:
            strategy_params[key] = value
    
    # Package final configuration
    config["strategy_params"] = strategy_params
    return config

# ============================================================================
# MIGRATION COMPLETE
# ============================================================================
# All scripts have been successfully migrated to the new configuration system!
# The old SIM_CONFIG has been removed and replaced with:
# - get_simulation_config() for basic settings
# - create_composite_config() for strategy-specific configurations
# - get_experiment_preset() for predefined experiment sets