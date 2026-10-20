# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Trackstand RL contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 5.0-compatible Beta policy with normalized actions in [-1, 1].

Uses the Softplus+1 parameterization from upstream RSL-RL's BetaDistribution:
https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/modules/distribution.py
The affine Jacobian is included in both log probability and entropy. No
third-party package is patched. The MLP predicts [..., 2, num_actions] shape
parameters, not motor targets; deterministic inference exports their mean.
"""

import math

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Beta, kl_divergence


class BetaMean(nn.Module):
    """Exportable mapping from unconstrained shape parameters to bounded actions."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = mlp_output.unbind(dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return 2.0 * (alpha / (alpha + beta)) - 1.0


class BetaDistribution(Distribution):
    """Independent Beta actions, rescaled from [0, 1] to [-1, 1].

    Alpha/beta are >= 1 (strictly > 1 in exact arithmetic), giving a unimodal
    family. The two motor actions require four internal MLP outputs. A persistent
    version marker makes standalone checkpoint export distinguish this head
    from a Gaussian without guessing from the output-layer size.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        self.register_buffer("_beta_policy_version", torch.tensor(1, dtype=torch.int64))
        self._distribution: Beta | None = None
        self._mean_output = BetaMean()

    def update(self, mlp_output: torch.Tensor) -> None:
        alpha_raw, beta_raw = mlp_output.unbind(dim=-2)
        self._distribution = Beta(
            torch.nn.functional.softplus(alpha_raw) + 1.0,
            torch.nn.functional.softplus(beta_raw) + 1.0,
            validate_args=False,
        )

    def _current(self) -> Beta:
        if self._distribution is None:
            raise RuntimeError("Call BetaDistribution.update() before requesting stochastic statistics.")
        return self._distribution

    def sample(self) -> torch.Tensor:
        return 2.0 * self._current().sample() - 1.0

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self._mean_output(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return BetaMean()

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        return 2.0 * self._current().mean - 1.0

    @property
    def std(self) -> torch.Tensor:
        return 2.0 * self._current().stddev

    @property
    def entropy(self) -> torch.Tensor:
        # H(2*x - 1) = H(x) + log(2), once per action dimension.
        return (self._current().entropy() + math.log(2.0)).sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        distribution = self._current()
        return distribution.concentration1, distribution.concentration0

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        # Rescaling can round a sample to +/-1. Use an interior value only for
        # the density evaluation, not to clip the action delivered to the task.
        eps = torch.finfo(outputs.dtype).eps
        unit_actions = ((outputs + 1.0) * 0.5).clamp(eps, 1.0 - eps)
        log_prob = (self._current().log_prob(unit_actions) - math.log(2.0)).sum(dim=-1)
        inside = ((outputs >= -1.0) & (outputs <= 1.0)).all(dim=-1)
        return torch.where(inside, log_prob, -torch.inf)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # The common invertible affine rescaling preserves KL divergence.
        old = Beta(*old_params, validate_args=False)
        new = Beta(*new_params, validate_args=False)
        return kl_divergence(old, new).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        # Both shape heads start identically: zero deterministic action and a
        # broad symmetric Beta(softplus(0)+1, softplus(0)+1), std ~= 0.4775.
        output_layer = next(layer for layer in reversed(list(mlp.modules())) if isinstance(layer, nn.Linear))
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
