"""
Strategy registry for assignment policies.

Importing this package auto-registers built-in strategies (greedy, composite, etc.).
External algorithms can register themselves using the @register decorator.
"""

from importlib import import_module
from typing import Callable, Dict

# Type-hinted registry dictionary
_STRATEGIES: Dict[str, Callable] = {}

def register(name: str) -> Callable:
    """Decorator to register an assignment function under a specific name."""
    def _decorator(fn: Callable) -> Callable:
        _STRATEGIES[name] = fn
        return fn
    return _decorator

def get_strategy(name: str) -> Callable:
    """Return the registered assignment callable for the given strategy name."""
    if not _STRATEGIES:  # Lazy-import bundled strategies on first call
        _auto_import_builtins()

    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown assignment strategy '{name}'. Available strategies: {list(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[name]

def _auto_import_builtins() -> None:
    """Dynamically import packaged strategy modules so they self-register."""
    builtin_strategies = (
        "greedy", 
        "composite", 
        "fatp", 
        "laf", 
        "ewma_only", 
        "random_assign", 
        "fatp_ann",
        "mmd_batch",
        "cb",
    )
    
    for mod in builtin_strategies:
        # We explicitly DO NOT use a try/except block here. 
        # If a strategy file is missing or contains a broken import, the simulation 
        # MUST fail loudly to prevent running experiments with corrupted codebases.
        import_module(f"{__name__}.{mod}")