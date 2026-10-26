"""V1 observation/action regression tests; no ROS, CAN, or motor actuation."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trackstand_deploy.contracts import (
    SteeringTargetLimiter, build_observation, physx_revolute_angle, scale_policy_action,
)
from trackstand_deploy.policy import OnnxPolicy


@pytest.mark.parametrize("rear", [0., 3.5, -3.5, 6.2, -6.2, 2 * math.pi + .01, -2 * math.pi - .01, 19.])
def test_rear_policy_coordinate_matches_observed_physx_branch(rear):
    obs = build_observation([0, 0, 0, 1], [0, 0, 0], .1, rear, 0, 0)
    expected = 2 * np.arctan2(np.sin(rear / 2), np.cos(rear / 2))
    assert obs[6] == pytest.approx(expected)
    if abs(rear) < 2 * math.pi:
        assert obs[6] == pytest.approx(rear)


def test_finite_observations_are_not_clipped_to_one_hundred():
    obs = build_observation([0, 0, 0, 1], [101, -102, 103], .1, 0, -104, 105)
    assert obs[[2, 3, 4, 7, 8]] == pytest.approx([101, -102, 103, -104, 105])


def test_nonfinite_repair_retains_finite_values():
    obs = build_observation([0, 0, 0, 1], [math.nan, math.inf, 101], math.inf, math.inf, -math.inf, 104)
    assert np.isfinite(obs).all()
    assert obs == pytest.approx([0, -1, 0, 0, 101, 0, 0, 0, 104, 0, 0])


def test_slew_steps_follow_policy_actions_and_reset_from_measured_angle():
    limiter = SteeringTargetLimiter()
    limiter.reset(.1)
    assert [limiter.update(1.57) for _ in range(4)] == pytest.approx([.578, 1.056, 1.534, 1.57])
    assert limiter.update(-1.57) == pytest.approx(1.092)
    limiter.reset(-.05)
    assert limiter.update(-1.57) == pytest.approx(-.528)
    limiter.reset(2)
    assert limiter.target_rad == 1.57


@pytest.mark.parametrize("raw", [[1.001, 0], [0, -1.001], [100, -100]])
def test_unbounded_actor_is_rejected_instead_of_clipped(raw):
    with pytest.raises(ValueError, match="within"):
        scale_policy_action(np.array(raw))


@pytest.mark.parametrize("output", [np.array([[1.001, 0]]), np.array([[0, math.nan]]), np.zeros((2, 2))])
def test_invalid_runtime_output_is_rejected(output):
    policy = object.__new__(OnnxPolicy)
    policy._session = SimpleNamespace(run=lambda *args: [output])
    with pytest.raises(RuntimeError, match="Beta"):
        policy.act(np.zeros(11, dtype=np.float32))


@pytest.mark.parametrize("model", ["2026-09-17_19-46-37_model_41050", "2026-09-22_19-45-06_model_91850"])
def test_bundled_v0_actors_are_incompatible(model):
    path = Path(__file__).parents[1] / "models" / ("trackstand_rl_rsl_" + model + ".onnx")
    with pytest.raises(ValueError, match="11"):
        OnnxPolicy(str(path))


def test_physx_branch_is_not_indefinitely_cumulative_or_single_turn():
    assert physx_revolute_angle(4) == pytest.approx(4)
    assert physx_revolute_angle(4 + 4 * math.pi) == pytest.approx(4)
    assert physx_revolute_angle(2 * math.pi + .01) == pytest.approx(-2 * math.pi + .01)
