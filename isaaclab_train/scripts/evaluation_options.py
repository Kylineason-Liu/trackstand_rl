"""Evaluation-only options; importing this module does not launch Isaac Sim."""

import argparse


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pure_survival",
        action="store_true",
        help="Keep only time_out and fallen terminations; disable yaw, drift, wheel-net and divergence checks.",
    )


def configure_pure_survival(env_cfg, enabled: bool) -> None:
    """Override only termination configuration, before constructing the environment."""
    if not enabled:
        return
    term_cfg = getattr(env_cfg, "terminations", None)
    if term_cfg is None:
        raise ValueError("--pure_survival requires configured time_out and fallen termination terms")
    terms = term_cfg if isinstance(term_cfg, dict) else vars(term_cfg)
    keep = ("time_out", "fallen")
    if any(terms.get(name) is None for name in keep):
        raise ValueError("--pure_survival requires configured time_out and fallen termination terms")
    disabled = [name for name, value in terms.items() if name not in keep and value is not None]
    for name in disabled:
        if isinstance(term_cfg, dict):
            term_cfg[name] = None
        else:
            setattr(term_cfg, name, None)
    print("[INFO] pure_survival: only time_out and fallen terminations; disabled: " + ", ".join(disabled))
    if "diverged" in disabled:
        print("[WARN] pure_survival disables the diverged containment termination.")
