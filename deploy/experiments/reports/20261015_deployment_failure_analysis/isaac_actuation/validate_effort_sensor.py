"""Validate the installed PhysX projected-effort sensor on a free, unit-inertia rotor.

This creates a separate synthetic stage; no bicycle asset or configuration is edited.
Run with Isaac Sim's Python, --headless --device cuda:0 --output <fresh-directory>.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import numpy as np
import torch
import warp as wp
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab_physx.physics import PhysxCfg


def tensor(value):
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value) if isinstance(value, wp.array) else value


try:
    dt = 0.0025
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, gravity=(0.0, 0.0, 0.0), device=args.device, physics=PhysxCfg())
    )
    stage = sim.stage
    UsdGeom.Xform.Define(stage, "/World/Rotor")
    for name in ("base", "wheel"):
        prim = UsdGeom.Xform.Define(stage, f"/World/Rotor/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateAngularDampingAttr(0.0)
        body.CreateLinearDampingAttr(0.0)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(1.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Rotor/fixed")
    fixed.CreateBody1Rel().SetTargets(["/World/Rotor/base"])
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Rotor/hinge")
    joint.CreateBody0Rel().SetTargets(["/World/Rotor/base"])
    joint.CreateBody1Rel().SetTargets(["/World/Rotor/wheel"])
    joint.CreateAxisAttr("Z")
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(10.0)
    drive.CreateDampingAttr(0.2)
    drive.CreateMaxForceAttr(5.6)
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Rotor",
            spawn=None,
            actuators={
                "rotor": ImplicitActuatorCfg(
                    joint_names_expr=["hinge"], stiffness=10.0, damping=0.2,
                    effort_limit_sim=5.6, velocity_limit_sim=10000.0,
                )
            },
        )
    )
    sim.reset()
    result = []
    # Joint-axis effort includes the implicit drive AND external feed-forward.
    # A unit-inertia, contact/gravity/friction-free joint independently checks it
    # against J * (omega_after - omega_before) / dt.
    for label, kp, kd, target, ff in (
        ("positive_feedforward_only", 0.0, 0.0, 0.0, 2.0),
        ("negative_feedforward_only", 0.0, 0.0, 0.0, -2.0),
        ("implicit_drive_only", 10.0, 0.2, 0.1, 0.0),
        ("drive_and_feedforward", 10.0, 0.2, 0.1, 2.0),
        ("oversized_feedforward", 0.0, 0.0, 0.0, 20.0),
        ("saturated_implicit_drive", 10.0, 0.2, 2.0, 0.0),
    ):
        zero = torch.zeros((1, 1), device=robot.device)
        robot.write_joint_state_to_sim(zero, zero)
        robot.reset()
        robot.write_joint_stiffness_to_sim(torch.full_like(zero, kp))
        robot.write_joint_damping_to_sim(torch.full_like(zero, kd))
        robot.set_joint_position_target(torch.full_like(zero, target))
        robot.set_joint_velocity_target(zero)
        robot.set_joint_effort_target(torch.full_like(zero, ff))
        for index in range(4):
            velocity_before = tensor(robot.root_view.get_dof_velocities()).clone()
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            velocity_after = tensor(robot.root_view.get_dof_velocities()).clone()
            effort = tensor(robot.root_view.get_dof_projected_joint_forces()).clone()
            acceleration_effort = (velocity_after - velocity_before) / dt
            row = {
                "case": label, "step": index, "kp": kp, "kd": kd, "target_rad": target,
                "feedforward_nm": ff, "measured_joint_effort_nm": effort.item(),
                "unit_inertia_acceleration_effort_nm": acceleration_effort.item(),
                "absolute_error_nm": (effort - acceleration_effort).abs().item(),
            }
            result.append(row)
            print("[SENSOR VALIDATION]", json.dumps(row), flush=True)
    (args.output / "sensor_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    errors = [row["absolute_error_nm"] for row in result]
    if not np.isfinite(errors).all() or max(errors) > 0.005:
        raise RuntimeError(f"Projected joint effort does not match unit-inertia acceleration: max error {max(errors)}")
    for label in ("implicit_drive_only", "saturated_implicit_drive"):
        if max(abs(row["measured_joint_effort_nm"]) for row in result if row["case"] == label) < 0.5:
            raise RuntimeError(f"Implicit drive did not actuate in validation case {label}")
    print("[SENSOR VALIDATION] PASS: signs, feed-forward, implicit drive, and saturation cases", flush=True)
finally:
    app.close()
