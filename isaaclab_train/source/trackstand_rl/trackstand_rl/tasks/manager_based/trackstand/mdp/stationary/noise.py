"""Per-channel gaussian observation noise for the balance task.

The stock ``GaussianNoiseCfg`` applies ONE scalar std to the whole term output,
and a ``torch.Tensor`` std serializes into an unreadable binary blob in the
run's env.yaml dump (``class_to_dict`` passes tensors through; pyyaml falls
back to the pickle-style reduce protocol). A plain tuple std dumps cleanly
(``!!python/tuple``, as the parsers already handle) and keeps the cfg
declarative — this cfg subclass + func provide the same per-channel behavior
without that cost.
"""

import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise.noise_cfg import NoiseCfg


@configclass
class PerChannelGaussianNoiseCfg(NoiseCfg):
    """Additive gaussian noise with a per-channel std given as a plain tuple.

    ``std`` must have one entry per observation channel; verified at first call
    with a clear error. The observation manager applies this noise after the
    term func returns, with fresh i.i.d. draws every step (term #8 white noise).
    """

    func: str = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.mdp.stationary.noise:per_channel_gaussian_noise"
    std: tuple = (0.0,)


def per_channel_gaussian_noise(data: torch.Tensor, cfg: PerChannelGaussianNoiseCfg) -> torch.Tensor:
    """Apply ``cfg.std`` as per-channel gaussian noise (broadcast over the batch).

    Accepts tuple (cfg default), tensor, or scalar std — the dr_ramp curriculum
    writes back element-wise-lerped values in whichever type the cfg uses.
    """
    std = cfg.std
    if isinstance(std, tuple):
        if len(std) != data.shape[-1]:
            raise ValueError(
                f"PerChannelGaussianNoiseCfg.std has {len(std)} entries but the observation "
                f"has {data.shape[-1]} channels — update the cfg tuple or the obs layout."
            )
        std = torch.tensor(std, dtype=data.dtype, device=data.device)
    elif isinstance(std, torch.Tensor):
        std = std.to(dtype=data.dtype, device=data.device)

    if cfg.operation == "add":
        return data + std * torch.randn_like(data)
    elif cfg.operation == "scale":
        # parity with the stock gaussian_noise (mean=0): data * (std * randn)
        return data * (std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return std * torch.randn_like(data)
    raise ValueError(f"Unknown operation in noise: {cfg.operation}")
