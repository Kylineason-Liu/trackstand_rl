"""Port ablation: which plant/physics settings explain CycleRL -> Locomotive-Cylinder-Aligned-v0.

``Locomotive-Cylinder-Aligned-v0`` (this repo) transfers to MuJoCo far better than CycleRL's
original ``1900_drive03`` (flat STE 2.4 vs 12.3 deg, gravel BSR 93 vs 0 %). Both runs share the
USD (byte-identical), skrl agent YAML, 7,500 PPO updates, rewards, commands, DR and observations;
their saved env.yaml files differ only in the seven factors below. Each ``ablation_*`` field
selects one factor's setting; ``ABLATIONS`` registers the variants as
``Locomotive-Abl-<name>-v0``.

| factor            | port (this repo)                         | CycleRL original                  |
|-------------------|------------------------------------------|-----------------------------------|
| physics_hz        | 400 (dt 0.0025, decimation 8)            | 100 (dt 0.01, decimation 2)       |
| actuator          | IdealPD (explicit, *_sim clips)          | Implicit (PhysX joint drive)      |
| drive_kd          | 0.2 N m s/rad                            | 5.0                               |
| passive_friction  | steer 0.046 + viscous, rear 0.106 N m    | none                              |
| default_friction  | 0.65 / min (robot shapes: the tires)     | 1.0 (PhysX default combine)       |
| solver            | velocity iters 2, ext. forces every iter | velocity iters 0, PhysxCfg()      |
| spawn_z           | -0.016 m (on the ground)                 | +0.02 m                           |

The USD authors no tire material, so ``default_friction`` is the tire-ground friction (ground
plane 1.0). All variants keep the 50 Hz policy rate (step_dt 0.02 s).
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import PhysxRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from .locomotive_env_cfg import _DRIVE_ACTUATOR, _STEERING_ACTUATOR, LocomotiveEnvCfg

_DECIMATION_BY_HZ = {400: 8, 100: 2}  # step_dt stays 0.02 s


@configclass
class LocomotiveAblationEnvCfg(LocomotiveEnvCfg):
    """Port settings by default; each field reverts one factor to CycleRL's original."""

    ablation_physics_hz: int = 400
    ablation_actuator: str = "ideal_pd"  # "ideal_pd" | "implicit"
    ablation_drive_kd: float = 0.2
    ablation_passive_friction: bool = True
    ablation_default_friction: float = 0.65  # 0.65 -> min combine; 1.0 -> PhysX default combine
    ablation_solver: str = "trackstand"  # "trackstand" | "cyclerl"
    ablation_spawn_z: float = -0.016

    def __post_init__(self):
        super().__post_init__()
        self.decimation = _DECIMATION_BY_HZ[self.ablation_physics_hz]
        self.sim.dt = 0.02 / self.decimation
        self.sim.render_interval = self.decimation

        steering, drive = _STEERING_ACTUATOR, _DRIVE_ACTUATOR.replace(damping=self.ablation_drive_kd)
        if self.ablation_actuator == "implicit":
            # CycleRL original: PhysX drive gains, effort/velocity limits only (no *_sim clips).
            steering, drive = (
                ImplicitActuatorCfg(
                    joint_names_expr=a.joint_names_expr, stiffness=a.stiffness, damping=a.damping,
                    effort_limit=a.effort_limit, velocity_limit=a.velocity_limit,
                    friction=a.friction, dynamic_friction=a.dynamic_friction, viscous_friction=a.viscous_friction,
                )
                for a in (steering, drive)
            )
        elif self.ablation_actuator != "ideal_pd":
            raise ValueError(f"unknown ablation_actuator {self.ablation_actuator!r}")
        if not self.ablation_passive_friction:
            steering, drive = (a.replace(friction=None, dynamic_friction=None, viscous_friction=None)
                               for a in (steering, drive))
        self.scene.robot.actuators = {"steering": steering, "drive": drive}

        if self.ablation_default_friction == 1.0:
            self.sim.physics_material = PhysxRigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        elif self.ablation_default_friction != 0.65:
            raise ValueError("ablation_default_friction must be 0.65 (port) or 1.0 (CycleRL)")

        if self.ablation_solver == "cyclerl":
            self.sim.physics = None  # SimulationContext -> PhysxCfg(): no external forces every iteration
            self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 0
        elif self.ablation_solver != "trackstand":
            raise ValueError(f"unknown ablation_solver {self.ablation_solver!r}")

        self.scene.robot.init_state.pos = (0.0, 0.0, self.ablation_spawn_z)


_CYCLERL = dict(ablation_physics_hz=100, ablation_actuator="implicit", ablation_drive_kd=5.0,
                ablation_passive_friction=False, ablation_default_friction=1.0, ablation_solver="cyclerl",
                ablation_spawn_z=0.02)

# name -> factor overrides (unlisted factors keep the port's setting)
ABLATIONS = {
    # Full CycleRL plant in this repo: does the in-repo replica reproduce CycleRL's numbers?
    "CycleRL": _CYCLERL,
    # 2x2x2 factorial over the interacting plant factors (minor factors at port settings);
    # 400Hz-IdealPD-Kd0.2 is the port itself (Locomotive-Cylinder-Aligned-v0).
    "400Hz-IdealPD-Kd5": dict(ablation_drive_kd=5.0),
    "400Hz-Implicit-Kd0.2": dict(ablation_actuator="implicit"),
    "400Hz-Implicit-Kd5": dict(ablation_actuator="implicit", ablation_drive_kd=5.0),
    "100Hz-IdealPD-Kd0.2": dict(ablation_physics_hz=100),
    "100Hz-IdealPD-Kd5": dict(ablation_physics_hz=100, ablation_drive_kd=5.0),
    "100Hz-Implicit-Kd0.2": dict(ablation_physics_hz=100, ablation_actuator="implicit"),
    "100Hz-Implicit-Kd5": dict(ablation_physics_hz=100, ablation_actuator="implicit", ablation_drive_kd=5.0),
    # Port with one minor factor reverted to CycleRL's.
    "Port-NoPassiveFriction": dict(ablation_passive_friction=False),
    "Port-Friction1.0": dict(ablation_default_friction=1.0),
    "Port-CycleRLSolver": dict(ablation_solver="cyclerl"),
    "Port-SpawnHigh": dict(ablation_spawn_z=0.02),
}


def _variant_cfg(name: str, overrides: dict):
    cls = type(f"LocomotiveAbl{name.replace('-', '').replace('.', 'p')}EnvCfg", (LocomotiveAblationEnvCfg,),
               dict(overrides))
    return configclass(cls)


ABLATION_CFGS = {name: _variant_cfg(name, overrides) for name, overrides in ABLATIONS.items()}
