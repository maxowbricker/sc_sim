"""
Stochastic worker acceptance behavior (Basık et al., CR-04).

P(accept) = exp(-d_pick) * c_willingness

Pure distance-based model; EWMA hunger twist can be added in acceptance_probability().
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

_rng = random.Random(42)


def seed_acceptance_rng(seed: int = 42) -> None:
    """Reset the dedicated acceptance RNG (call from EventSimulator.reset())."""
    _rng.seed(seed)


def acceptance_probability(d_pick_km: float, cfg: Optional[Dict[str, Any]] = None) -> float:
    """
    Basık acceptance probability from precomputed deadhead distance (km).

    Since a - b = -d_pick when revenue scales with trip distance, this simplifies to exp(-d_pick) * c.
    """
    if not cfg:
        return 1.0
    c = cfg.get("c_willingness", 0.6)
    return min(1.0, math.exp(-d_pick_km) * c)


def evaluate_worker_acceptance(d_pick_km: float, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    Roll acceptance dice for an offer at deadhead distance d_pick_km.

    Returns True immediately when acceptance is disabled (zero overhead path).
    """
    if not cfg or not cfg.get("enabled", False):
        return True
    prob = acceptance_probability(d_pick_km, cfg)
    return _rng.random() < prob
