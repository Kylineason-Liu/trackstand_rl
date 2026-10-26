"""Matched motor-input diagnostic using the actual v1 IdealPD actuators.

Bypass policy/action preprocessing: targets in inputs.npz are ALREADY scaled,
clipped and slew-limited. Do not reset, interpolate or generate fresh targets.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "isaaclab_train"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics_hz", type=int, choices=(400, 800), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
__import__("source.trackstand_rl.trackstand_rl.tasks")


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch if hasattr(value, "torch") else wp.to_torch(value)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    output = HERE / f"isaac{args.physics_hz}"
    output.mkdir(exist_ok=False)
    bank = dict(np.load(HERE / "inputs.npz"))
    task = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()
    task.sim.dt, task.sim.device = 1 / args.physics_hz, args.device
    task.sim.render_interval = args.physics_hz // 50
    sim = sim_utils.SimulationContext(task.sim)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.,
                                             friction_combine_mode="min", restitution=0.)
    floor = sim_utils.GroundPlaneCfg(physics_material=material)
    floor.func("/World/Ground", floor)
    robots = {}
    for name, fixed, n in (("airborne", True, 10), ("ground", False, 2)):
        cfg = task.scene.robot.copy()
        cfg.prim_path = f"/World/{name}_.*/Robot"
        cfg.spawn.articulation_props.fix_root_link = fixed
        if fixed:
            # fix_root_link materializes the world joint and promotes the
            # articulation root to Geometry. Resolve that root explicitly.
            cfg.articulation_root_prim_path = "/Geometry"
        cfg.init_state.pos = (0., 0., 2. if fixed else -.016)
        for index in range(n):
            sim_utils.create_prim(f"/World/{name}_{index}", "Xform",
                                  translation=(2. * index, 0. if fixed else 4., 0.))
        robots[name] = Articulation(cfg)
    sim.reset()
    metadata, samples = {}, {}
    nsteps, substeps = 400, args.physics_hz // 50
    fields = ("position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
              "computed_nm", "submitted_nm", "solver_effort_nm")
    for name, robot in robots.items():
        n = 10 if name == "airborne" else 2
        print("FIXTURE", name, "instances", robot.num_instances, "fixed", robot.is_fixed_base, flush=True)
        assert robot.num_instances == n and robot.is_fixed_base == (name == "airborne")
        # sim.reset() may perform a warm-up step. Restore floating poses and
        # velocities explicitly, including their isolated scene-grid offsets.
        if name == "ground":
            pose = tensor(robot.data.default_root_pose).clone()
            pose[:, 0] += 2. * torch.arange(n, device=sim.device)
            pose[:, 1] += 4.
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=sim.device))
        joints = [robot.find_joints(key)[0][0] for key in ("steering_joint", "rear_wheel_joint")]
        robot._replay_joints = joints
        q = torch.zeros_like(tensor(robot.data.joint_pos))
        robot.write_joint_position_to_sim_index(position=q)
        robot.write_joint_velocity_to_sim_index(velocity=q)
        robot.reset()
        sim.forward()
        if name == "airborne":
            friction = torch.tensor([[.046036056104 if i % 2 == 0 else 0.] for i in range(n)],
                                    device=sim.device, dtype=torch.float32)
            viscous = friction * (.000910713492 / .046036056104)
            robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=friction, joint_dynamic_friction_coeff=friction,
                joint_viscous_friction_coeff=viscous, joint_ids=[joints[0]])
        metadata[name] = dict(
            fixed_base=robot.is_fixed_base, body_names=robot.body_names, joint_names=robot.joint_names,
            initial_joint_position=tensor(robot.data.joint_pos).cpu().tolist(),
            initial_joint_velocity=tensor(robot.data.joint_vel).cpu().tolist(),
            initial_root_pose=tensor(robot.data.root_pose_w).cpu().tolist(),
            initial_root_velocity=tensor(robot.data.root_vel_w).cpu().tolist(),
            masses=tensor(robot.root_view.get_masses()).cpu().tolist(),
            initial_com_world_m=tensor(robot.data.body_com_pos_w).cpu().tolist(),
            joint_friction_properties=tensor(robot.root_view.get_dof_friction_properties()).cpu().tolist(),
            gains={key: {field: tensor(getattr(act, field)).cpu().tolist() for field in ("stiffness", "damping")}
                   for key, act in robot.actuators.items()})
        samples[name] = {key: torch.empty((nsteps, substeps, n, 2), device=sim.device) for key in fields}
        samples[name]["roll_rad"] = torch.empty((nsteps, substeps, n), device=sim.device)
        robot._replay_target = torch.as_tensor(bank[f"{name}_target_rad"], device=sim.device)
        robot._replay_rear = torch.as_tensor(bank[f"{name}_rear_nm"], device=sim.device)
    import isaaclab.utils.math as math_utils
    for step in range(nsteps):
        for name, robot in robots.items():
            steer, rear = robot._replay_joints
            robot.set_joint_position_target_index(target=robot._replay_target[step, :, None], joint_ids=[steer])
            robot.set_joint_effort_target_index(target=robot._replay_rear[step, :, None], joint_ids=[rear])
        for sub in range(substeps):
            for name, robot in robots.items():
                robot.write_data_to_sim()
                joints, store = robot._replay_joints, samples[name]
                for key, value in (("position_before_rad", robot.data.joint_pos),
                                   ("velocity_before_rad_s", robot.data.joint_vel),
                                   ("computed_nm", robot.data.computed_torque),
                                   ("submitted_nm", robot._joint_effort_target_sim)):
                    store[key][step, sub] = tensor(value)[:, joints]
            sim.step(render=False)
            for name, robot in robots.items():
                robot.update(1 / args.physics_hz)
                joints, store = robot._replay_joints, samples[name]
                for key, value in (("position_after_rad", robot.data.joint_pos),
                                   ("velocity_after_rad_s", robot.data.joint_vel),
                                   ("solver_effort_nm", robot.root_view.get_dof_projected_joint_forces())):
                    store[key][step, sub] = tensor(value)[:, joints]
                store["roll_rad"][step, sub] = math_utils.euler_xyz_from_quat(tensor(robot.data.root_quat_w))[0]
        if step % 100 == 0:
            print(f"MATCHED REPLAY {step}/{nsteps}", flush=True)
    for name, store in samples.items():
        arrays = {key: value.cpu().numpy() for key, value in store.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        np.savez_compressed(output / f"{name}.npz", **arrays,
                            target_rad=bank[f"{name}_target_rad"], rear_nm=bank[f"{name}_rear_nm"],
                            names=bank[f"{name}_names"], physics_dt_s=1 / args.physics_hz)
    metadata.update(completed=True, policy_hz=50, physics_hz=args.physics_hz, seed=42,
                    dr_resampling=False, policy_inference=False, automatic_resets=False,
                    parameters="nominal v1; only fixture and paired steering friction ablation differ")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("MATCHED REPLAY COMPLETE", output, flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
