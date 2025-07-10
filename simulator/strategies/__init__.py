"""Strategy registry for assignment policies.

Importing this package auto-registers built-in strategies (greedy, composite,…).
External algorithms can register themselves by calling

```python
from simulator.strategies import register

@register("my_algo")
def assign(...):
    ...
```
"""

from importlib import import_module

_STRATEGIES = {}


def register(name):
    """Decorator to register an assignment function under *name*."""

    def _decorator(fn):
        _STRATEGIES[name] = fn
        return fn

    return _decorator


def get_strategy(name):
    """Return the registered assignment callable for *name*."""

    if not _STRATEGIES:  # first call: lazy-import bundled strategies
        _auto_import_builtins()

    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown assignment strategy '{name}'. Available: {list(_STRATEGIES)}"
        )
    return _STRATEGIES[name]


def _auto_import_builtins():
    """Dynamically import packaged strategy modules so they self-register."""

    for mod in ("greedy", "composite", "fatp"):
        try:
            import_module(f"{__name__}.{mod}")
        except ModuleNotFoundError:
            # Optional module (fatp may not be implemented yet)
            pass
