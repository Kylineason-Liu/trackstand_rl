"""Isolate rear implicit-controller damping on parallel unit-inertia rotors.

No bicycle geometry/configuration is modified. These are controlled actuator
experiments, not bicycle balance trials or identified real-motor parameters.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists() or args.steps < 1:
    raise ValueError("Use a fresh output directory and a positive step count")
args.output.mkdir(parents=True)
invocation = sys.argv.copy()
app = AppLauncher(args).app

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


SCENARIOS = (
    ("coast_10", 10.0, 0.0),
    ("coast_61_6", 61.6, 0.0),
    ("drive_from_rest", 0.0, 6.28),
    ("reverse_from_rest", 0.0, -6.28),
    ("hold_10", 10.0, 0.5),
    ("brake_10", 10.0, -6.28),
    ("brake_61_6", 61.6, -6.28),
)

try:
    torch.manual_seed(args.seed)
    dt = 0.0025
    cases = [
        {"scenario": name, "damping": damping, "initial_velocity_rad_s": speed, "feedforward_nm": torque}
        for name, speed, torque in SCENARIOS for damping in (0.0, 0.05)
    ]
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=dt, gravity=(0.0, 0.0, 0.0), device=args.device,
            physics=PhysxCfg(enable_external_forces_every_iteration=True),
        )
    )
    stage = sim.stage
    for index in range(len(cases)):
        root = f"/World/Rotors/rotor_{index:02d}"
        UsdGeom.Xform.Define(stage, root)
        for name in ("base", "wheel"):
            prim = UsdGeom.Xform.Define(stage, f"{root}/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateAngularDampingAttr(0.0)
            body.CreateLinearDampingAttr(0.0)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(1.0)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/fixed")
        fixed.CreateBody1Rel().SetTargets([f"{root}/base"])
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/hinge")
        joint.CreateBody0Rel().SetTargets([f"{root}/base"])
        joint.CreateBody1Rel().SetTargets([f"{root}/wheel"])
        joint.CreateAxisAttr("Z")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(6.28)
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Rotors/rotor_.*", spawn=None,
            actuators={"drive": ImplicitActuatorCfg(
                joint_names_expr=["hinge"], stiffness=0.0, damping=0.05,
                effort_limit_sim=6.28, velocity_limit_sim=10000.0,
                friction=0.0, dynamic_friction=0.0, viscous_friction=0.0, armature=0.0,
            )},
        )
    )
    sim.reset()
    order = [int(re.search(r"/rotor_(\d+)/", paths[0]).group(1)) for paths in robot.root_view.dof_paths]
    if sorted(order) != list(range(len(cases))):
        raise RuntimeError(f"Unexpected rotor view ordering: {order}")
    cases = [cases[index] for index in order]
    zero = torch.zeros((len(cases), 1), device=robot.device)
    speed = torch.tensor([[case["initial_velocity_rad_s"]] for case in cases], device=robot.device)
    damping = torch.tensor([[case["damping"]] for case in cases], device=robot.device)
    feedforward = torch.tensor([[case["feedforward_nm"]] for case in cases], device=robot.device)
    robot.write_joint_state_to_sim(zero, speed)
    robot.reset()
    robot.actuators["drive"].damping[:] = damping
    robot.write_joint_damping_to_sim(damping)
    robot.set_joint_position_target(zero)
    robot.set_joint_velocity_target(zero)
    robot.set_joint_effort_target(feedforward)
    velocities = np.empty((args.steps + 1, len(cases)), dtype=np.float64)
    efforts = np.empty((args.steps, len(cases)), dtype=np.float64)
    velocities[0] = tensor(robot.root_view.get_dof_velocities())[:, 0].cpu().numpy()
    for step in range(args.steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        velocities[step + 1] = tensor(robot.root_view.get_dof_velocities())[:, 0].cpu().numpy()
        efforts[step] = tensor(robot.root_view.get_dof_projected_joint_forces())[:, 0].cpu().numpy()
    times = np.arange(args.steps + 1) * dt
    summary = []
    for index, case in enumerate(cases):
        kd, initial, torque = case["damping"], case["initial_velocity_rad_s"], case["feedforward_nm"]
        expected = initial + torque * times if kd == 0 else torque / kd + (initial - torque / kd) * np.exp(-kd * times)
        summary.append({
            **case,
            "final_velocity_rad_s": float(velocities[-1, index]),
            "analytic_final_velocity_rad_s": float(expected[-1]),
            "max_velocity_error_rad_s": float(np.max(np.abs(velocities[:, index] - expected))),
            "initial_measured_effort_nm": float(efforts[0, index]),
            "max_absolute_effort_nm": float(np.max(np.abs(efforts[:, index]))),
            "max_acceleration_effort_error_nm": float(
                np.max(np.abs(np.diff(velocities[:, index]) / dt - efforts[:, index]))
            ),
        })
    np.savez_compressed(args.output / "rotor_samples.npz", time_s=times, velocity_rad_s=velocities, effort_nm=efforts)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Keep the numerical tolerance explicit: this is a single-precision PhysX
    # integration test, not an exact symbolic solver or motor identification.
    validation = {
        "max_velocity_error_rad_s": max(row["max_velocity_error_rad_s"] for row in summary),
        "velocity_error_tolerance_rad_s": 0.1,
        "max_acceleration_effort_error_nm": max(row["max_acceleration_effort_error_nm"] for row in summary),
        "acceleration_effort_error_tolerance_nm": 0.05,
    }
    validation["passed"] = bool(
        np.isfinite(velocities).all()
        and np.isfinite(efforts).all()
        and validation["max_velocity_error_rad_s"] < validation["velocity_error_tolerance_rad_s"]
        and validation["max_acceleration_effort_error_nm"] < validation["acceleration_effort_error_tolerance_nm"]
    )
    (args.output / "metadata.json").write_text(json.dumps({
        "command": invocation, "seed": args.seed, "dt": dt, "steps": args.steps,
        "duration_s": args.steps * dt, "inertia_kg_m2": 1.0,
        "gravity_contacts_passive_friction": "absent", "effort_limit_sim": 6.28,
        "velocity_limit_sim": 10000.0, "scope": "synthetic free-rotor actuator test, not bicycle balance",
        "validation": validation,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    panels = (("coast_10", False), ("drive_from_rest", False), ("brake_61_6", False), ("brake_61_6", True))
    for ax, (name, use_effort) in zip(axes.flat, panels, strict=True):
        for index, case in enumerate(cases):
            if case["scenario"] == name:
                values = efforts[:, index] if use_effort else velocities[:, index]
                ax.plot(times[1:] if use_effort else times, values, label=f"Kd={case['damping']:g}")
        if use_effort:
            ax.axhline(-6.28, color="gray", linestyle="--", label="-6.28 N m reference")
        ax.set(
            title=name.replace("_", " "), xlabel="Time (s)",
            ylabel="Joint effort (N m)" if use_effort else "Velocity (rad/s)",
        )
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Synthetic unit-inertia rotor: controller damping 0 vs 0.05")
    fig.savefig(args.output / "drive_damping_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["passed"]:
        raise RuntimeError("Rotor dynamics / measured-effort validation failed; see metadata.json")
    print(json.dumps(summary, indent=2), flush=True)
    print("PASS: rotor dynamics and measured-effort cross-check", flush=True)
finally:
    app.close()
