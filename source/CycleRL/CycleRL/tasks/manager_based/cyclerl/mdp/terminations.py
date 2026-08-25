"""Terminations used by both stationary and dynamic balancing.

Lean / flywheel roll terms live in ``mdp.dynamic.terminations``.
Fallen (stationary) lives in ``mdp.stationary.terminations``.
"""

from .dynamic.terminations import bicycle_lean_termination, roll_exceeds_threshold  # noqa: F401
from .stationary.terminations import thubike_fallen_termination  # noqa: F401
