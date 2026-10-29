# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/paper_metrics.py (imports/paths rewritten for trackstand_rl locomotive).
"""CycleRL paper (2603.15013) Table II metric definitions, torch-free.

Shared by the Isaac ``eval_logger`` and the standalone MuJoCo paper suite
(``mujoco_sim2sim/scripts/locomotive/eval_paper_suite.py``) so both simulators score episodes
with the same code. Episode objects only need the attributes read below
(``complete``, ``bsr_balanced`` and the ladder level attributes).

BSR  — % of episodes maintaining |φ| < 0.5 rad.
BRT  — recovery time from disturbances exceeding 0.3 rad.
MBD  — longest continuous balancing period.
CAT  — maximum recoverable roll angle maintaining BSR > 95%.
STE  — RMS error between commanded and actual steering.
VTE  — mean absolute error in longitudinal velocity.
SRL  — time to reach within 10% of the target command value.
MNT  — highest sensor noise level maintaining BSR > 95%.
MSS  — minimum speed maintaining BSR > 95%.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

PAPER_BSR_ROLL_LIMIT_RAD = 0.5
PAPER_BRT_DISTURB_RAD = 0.3
PAPER_SRL_TOLERANCE = 0.10
PAPER_BSR_GATE = 0.95
# MSS walks down from the paper-suite cruise speed (eval_scenarios._PAPER_EVAL_VX).
PAPER_MSS_ANCHOR_MS = 1.4


def balance_success_rate(episodes: Sequence[Any]) -> float:
    """Paper BSR: fraction of complete episodes that never exceeded |φ| = 0.5 rad."""
    complete = [e for e in episodes if e.complete]
    if not complete:
        return 0.0
    return len([e for e in complete if e.bsr_balanced]) / len(complete)


def _bsr_upper_bound(episodes: Sequence[Any], z: float = 1.645) -> float:
    """One-sided 95% Wilson upper bound on BSR for a small rung."""
    complete = [e for e in episodes if e.complete]
    n = len(complete)
    if n == 0:
        return 0.0
    p = len([e for e in complete if e.bsr_balanced]) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return centre + half


def analyze_bsr_frontier(
    episodes: Sequence[Any],
    level_attr: str,
    *,
    direction: str = "max",
    anchor: float | None = None,
    bin_width: float | None = None,
    use_abs: bool = False,
    bsr_min: float = PAPER_BSR_GATE,
    min_samples: int = 5,
) -> float:
    """Contiguous BSR frontier along ``level_attr``.

    Walks the ladder from the easy end (``anchor``, or the lowest level for
    ``direction="max"`` / highest for ``"min"``) and returns the last rung before the
    first one that fails, so an isolated passing rung past a failure does not count.
    A rung fails when its BSR is significantly below ``bsr_min`` (Wilson upper bound
    under the gate): ladders put ~30 episodes per rung, where one crash already reads
    94%. Shared by CAT (initial roll), MNT (noise ρ) and MSS (commanded speed). Rungs
    with fewer than ``min_samples`` episodes are skipped. Returns NaN when the ladder
    was not swept or the first populated rung already fails.
    """
    bins: dict[float, list[Any]] = {}
    for ep in episodes:
        if not ep.complete:
            continue
        raw = getattr(ep, level_attr, float("nan"))
        if raw is None or not math.isfinite(float(raw)):
            continue
        level = abs(float(raw)) if use_abs else float(raw)
        key = round(level / bin_width) * bin_width if bin_width else level
        bins.setdefault(round(key, 6), []).append(ep)

    levels = sorted(bins, reverse=(direction != "max"))
    if anchor is not None:
        levels = [lv for lv in levels if (lv >= anchor if direction == "max" else lv <= anchor)]
    frontier = float("nan")
    for level in levels:
        eps = bins[level]
        if len(eps) < min_samples:
            continue
        if _bsr_upper_bound(eps) < bsr_min:
            break
        frontier = level
    return frontier


def analyze_critical_angle(episodes: Sequence[Any], **kwargs: Any) -> float:
    """CAT (deg): maximum recoverable spawn roll angle holding BSR > 95%.

    Paper Table IV footnote 2: reported as the minimum of both sides, so the left
    and right lean frontiers are found separately (the 0° rung counts for both).
    """
    kwargs.setdefault("bin_width", 2.5)
    sides = []
    for sign in (1.0, -1.0):
        side_eps = [
            e
            for e in episodes
            if math.isfinite(float(getattr(e, "spawn_roll_deg", float("nan"))))
            and sign * float(e.spawn_roll_deg) >= 0.0
        ]
        sides.append(
            analyze_bsr_frontier(side_eps, "spawn_roll_deg", direction="max", use_abs=True, **kwargs)
        )
    if any(math.isnan(s) for s in sides):
        return float("nan")
    return min(sides)


def analyze_noise_tolerance(episodes: Sequence[Any], **kwargs: Any) -> float:
    """MNT (fraction): highest observation-noise ρ holding BSR > 95%."""
    kwargs.setdefault("bin_width", 0.02)
    return analyze_bsr_frontier(episodes, "noise_rho", direction="max", **kwargs)


def analyze_min_sustaining_speed(episodes: Sequence[Any], **kwargs: Any) -> float:
    """MSS (m/s): lowest commanded speed holding BSR > 95%.

    Walks down from the paper-suite cruise speed: bikes also tip at high speed, so
    starting from the top of the ladder would stop on a high-speed failure.
    """
    kwargs.setdefault("bin_width", 0.1)
    kwargs.setdefault("anchor", PAPER_MSS_ANCHOR_MS)
    return analyze_bsr_frontier(episodes, "spawn_cmd_vx_ms", direction="min", **kwargs)
