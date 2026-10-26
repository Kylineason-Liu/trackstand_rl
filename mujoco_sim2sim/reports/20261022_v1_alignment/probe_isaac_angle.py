"""Isolate the installed v1 backend's rear-angle convention; no task source edits.

Run with Isaac Sim Python, --headless --device cuda:0. Gravity, motor gains and
joint losses are disabled ONLY in this synthetic airborne probe configuration.
This is not a policy performance evaluation.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "isaaclab_train"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401

env = None
try:
    phases = np.array([0, 3.5, -3.5, 2*np.pi-.01, 2*np.pi+.01, -2*np.pi-.01,
                       3*np.pi, -3*np.pi, 4*np.pi+.01, -4*np.pi-.01, 6*np.pi+.01, 100.0])
    cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs, cfg.sim.device = len(phases), args.device
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 2.0)
    for actuator in cfg.scene.robot.actuators.values():
        actuator.stiffness = actuator.damping = 0.0
        actuator.friction = actuator.dynamic_friction = actuator.viscous_friction = 0.0
    env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot = env.scene["robot"]
    zero_body_quaternions = robot.data.body_link_quat_w.torch[0].cpu().numpy().copy()
    import omni.usd
    from pxr import PhysxSchema
    stage = omni.usd.get_context().get_stage()
    link_defaults = []
    for path in robot.root_view.link_paths[0]:
        prim = stage.GetPrimAtPath(path)
        schema = PhysxSchema.PhysxRigidBodyAPI(prim)
        link_defaults.append(dict(path=path, has_api=prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI),
                                  angular_damping=schema.GetAngularDampingAttr().Get(),
                                  linear_damping=schema.GetLinearDampingAttr().Get(),
                                  max_depenetration_velocity=schema.GetMaxDepenetrationVelocityAttr().Get()))
    joint = robot.find_joints("rear_wheel_joint")[0][0]
    pos = robot.data.joint_pos.torch.clone()
    vel = torch.zeros_like(pos)
    pos[:, joint] = torch.tensor(phases, device=env.device, dtype=pos.dtype)
    robot.write_joint_state_to_sim(pos, vel)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    after = robot.data.joint_pos.torch[:, joint].cpu().numpy().copy()

    # Continuous positive and negative rotation, with no resets or state writes.
    pos.zero_()
    vel.zero_()
    vel[:, joint] = 4.0
    vel[1::2, joint] = -4.0
    robot.write_joint_state_to_sim(pos, vel)
    traces = []
    for _ in range(2400):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        traces.append(robot.data.joint_pos.torch[:2, joint].cpu().numpy().copy())
    trace = np.asarray(traces)
    np.savez_compressed(Path(__file__).with_name("isaac_angle_probe.npz"),
                        requested=phases, after_step=after, continuous_position_rad=trace,
                        physics_dt_s=env.physics_dt)
    result = {"completed": True, "task": "Trackstand-Balance-v1",
              "body_names": robot.body_names,
              "resolved_usd_link_defaults": link_defaults,
              "zero_joint_body_quaternions_xyzw": zero_body_quaternions.tolist(),
              "synthetic_airborne_zero_gravity_zero_loss_probe": True,
              "requested_position_rad": phases.tolist(), "after_step_position_rad": after.tolist(),
              "continuous_min_max_rad": [trace.min(0).tolist(), trace.max(0).tolist()],
              "jumps_larger_than_pi": [np.flatnonzero(np.abs(np.diff(trace[:, i])) > np.pi).tolist()
                                       for i in range(2)]}
    Path(__file__).with_name("isaac_angle_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
