"""Observations used by both stationary and dynamic balancing.

Locomotion IMU / command terms live in ``mdp.dynamic.observations``.
Stationary IMU lives in ``mdp.stationary.observations``.
"""


def __getattr__(name: str):
    """Resolve locomotion observations from ``mdp.dynamic.observations`` (deploy yaml)."""
    from .dynamic import observations as _dynamic_obs

    try:
        return getattr(_dynamic_obs, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
