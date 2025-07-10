SIM_CONFIG = {
    "dataset":   "didi",
    "time_step": "3s",

    # Assignment strategy
    "assignment_strategy": "greedy",          # "greedy" | "composite" | "fatp" | …
    "strategy_params": {                      # anything the chosen strategy needs
        "λ1": 1.0,
        "λ2": 1.0,
        "λ3": 0.5,
    },

    # Execution flags
    "service_time_mode": "distance",
    "teleport_on_complete": True,
}