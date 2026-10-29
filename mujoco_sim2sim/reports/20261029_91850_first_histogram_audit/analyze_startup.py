"""Capture first 100 ms at every physics step using the historical loader."""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("historical_runner_for_startup", HERE / "evaluate_historical.py")
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
np, mujoco, ort = h.np, h.mujoco, h.ort

options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(h.ONNX), options, providers=["CPUExecutionProvider"])
result = []
for dr in [False, True]:
    for case in ["baseline", "physics_800", "implicitfast_400"]:
        model, mod, _ = h.model_and_env(case)
        envs = [mod.BikeBalanceEnv(model=model, seed=s, enable_dr=dr) for s in range(42, 106)]
        obs = np.stack([e.reset() for e in envs])
        env = envs[0]
        traces, actions = [], []
        native_step = mujoco.mj_step

        def step(m, d):
            if d is env.data:
                before_q = d.qpos[env._steer_qadr]
                before_v = d.qvel[env._steer_dof]
                native_step(m, d)
                traces.append(dict(time_s=float(d.time), position_before_rad=float(before_q),
                                   velocity_before_rad_s=float(before_v), position_after_rad=float(d.qpos[env._steer_qadr]),
                                   velocity_before_projection_rad_s=float(d.qvel[env._steer_dof]),
                                   actuator_force_nm=float(d.actuator_force[0]), target_rad=float(d.ctrl[0]),
                                   rear_actuator_force_nm=float(d.actuator_force[1])))
            else:
                native_step(m, d)

        mujoco.mj_step = step
        try:
            for policy_step in range(5):
                raw = session.run(None, {"observation": obs})[0]
                actions.append(raw[0].copy())
                for i, e in enumerate(envs):
                    obs[i], done = e.step(raw[i])
                    assert not done, (case, dr, policy_step, i)
        finally:
            mujoco.mj_step = native_step
        force = np.array([r["actuator_force_nm"] for r in traces])
        reversals = int(np.count_nonzero(np.sign(force[1:]) * np.sign(force[:-1]) < 0))
        result.append(dict(case=case, dr=dr, seed=42, first_100ms_steering_torque_sign_reversals=reversals,
                           steering_speed_projection_count=sum(abs(r["velocity_before_projection_rad_s"]) > 23.9 for r in traces),
                           max_preprojection_steering_speed_rad_s=max(abs(r["velocity_before_projection_rad_s"]) for r in traces),
                           raw_actions=actions, traces=traces))
h.save(HERE / "startup_traces.json", result)
for r in result:
    print(json.dumps({k:v for k,v in r.items() if k not in ["raw_actions", "traces"]}))
