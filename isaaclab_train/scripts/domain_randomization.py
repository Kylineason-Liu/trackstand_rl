"""Apply the task's explicit, pre-construction domain-randomization switch."""


def configure_domain_randomization(env_cfg, enabled: bool) -> None:
    """Configure supported project tasks without altering third-party task defaults.

    Call before gym.make(): changing configuration after managers initialize
    cannot reliably add/remove reset events or restore nominal physics.
    """
    configure = getattr(env_cfg, "set_domain_randomization", None)
    if configure is None:
        if enabled:
            raise ValueError("--dr requires a task configuration implementing set_domain_randomization().")
        return
    configure(enabled)
    state = "ON (full per-episode randomization)" if enabled else "OFF (nominal physics and fixed initial state)"
    print(f"[INFO] domain randomization: {state}; observation white noise is unchanged.")
