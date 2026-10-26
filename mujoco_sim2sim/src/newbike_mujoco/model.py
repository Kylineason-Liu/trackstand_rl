"""In-memory control/contact configuration; verified model files stay read-only."""

from itertools import combinations
from pathlib import Path

import mujoco

from . import contract as c

JOINT_RENAMES = {"": "base_free", "steering_link": "steering_joint",
                 "rear_wheel_link": "rearwheel_joint", "front_wheel_link": "frontwheel_joint"}


def augment_trackstand_model(spec, model_dir, physics_hz=None):
    """Attach v1 torque-only actuators, without changing geometry/inertials/axes.

    PD is evaluated explicitly by BikeBalanceEnv once per physics step.
    Contact solref/solimp retain the previously verified MuJoCo settings; their
    numeric values are not claimed to reproduce PhysX TGS/SDF internals.
    """
    for joint in spec.joints:
        joint.name = JOINT_RENAMES.get(joint.name, joint.name)
    spec.compiler.meshdir = str(model_dir)
    spec.option.timestep = c.physics_timing(physics_hz)[0]
    spec.stat.center, spec.stat.extent = [0., 0., .4], 1.2
    if not any(geom.name == "floor" for geom in spec.geoms):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[10., 10., .05], contype=1, conaffinity=1)
    for geom in spec.geoms:
        floor = geom.name == "floor"
        geom.friction = [c.GROUND_FRICTION if floor else c.NOMINAL_FRICTION, 0., 0.]
        # PhysX min(ground1,tire.5-.8)=tire. MuJoCo normally uses MAX,
        # so let the tire's higher priority choose the pair coefficient.
        geom.priority = 0 if floor else 1
        geom.solref = [.005, 1.5]
        geom.solimp = [.9, .95, .001, .5, 2.]

    # PhysX articulation self-collision is disabled. Keep ground collisions.
    bodies = [body.name for body in spec.bodies if body.name and body.name != "world"]
    existing = {(exclude.bodyname1, exclude.bodyname2) for exclude in spec.excludes}
    for one, two in combinations(bodies, 2):
        if (one, two) not in existing and (two, one) not in existing:
            spec.add_exclude(bodyname1=one, bodyname2=two)
    for name, target, limit in (("steering_act", "steering_joint", c.STEER_EFFORT_LIMIT),
                                 ("drive_act", "rearwheel_joint", c.DRIVE_EFFORT_LIMIT)):
        if not any(act.name == name for act in spec.actuators):
            spec.add_actuator(name=name, target=target, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                              gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                              gainprm=[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                              forcelimited=True, forcerange=[-limit, limit])


def load_trackstand_model(path, physics_hz=None):
    path = Path(path).resolve()
    spec = mujoco.MjSpec.from_file(str(path))
    augment_trackstand_model(spec, path.parent, physics_hz=physics_hz)
    return spec.compile()


def build_env_from_mjcf(mjcf_path, seed, enable_dr=False, **kwargs):
    """Shared viewer/evaluator loader; never writes hidden XML into models/."""
    from .env import BikeBalanceEnv

    model = load_trackstand_model(mjcf_path, physics_hz=kwargs.get("physics_hz"))
    return BikeBalanceEnv(model=model, seed=seed, enable_dr=enable_dr, **kwargs), True
