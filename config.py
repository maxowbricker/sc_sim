SIM_CONFIG = {
    "dataset":   "didi",
    "time_step": "3s",

    # Assignment strategy
    "assignment_strategy": "composite",          # "greedy" | "composite" | "fatp" | …
    "strategy_params": {                      # anything the chosen strategy needs
        "λ1": 1.0,
        "λ2": 1.0,
        "λ3": 0.5,
        "gamma": 0.3,        # EWMA smoothing factor for fairness
        "k": 15,             # number of nearest workers to consider in phase-2
        "soft_threshold": 4.0, # minimum composite score to assign immediately
    },

    # Execution flags
    "service_time_mode": "distance",
    "teleport_on_complete": True,
}