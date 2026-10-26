"""Replay the exact shared 50-Hz targets through the production MuJoCo PD law."""

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco import contract as c
from newbike_mujoco.env import BikeBalanceEnv, _roll_from_quat
from newbike_mujoco.model import augment_trackstand_model


def base_model(fixed, rate):
    spec = mujoco.MjSpec.from_file(str(c.DEFAULT_MJCF))
    augment_trackstand_model(spec, c.DEFAULT_MJCF.parent, physics_hz=rate)
    if fixed:
        spec.delete(spec.joint("base_free"))
        spec.body("model_root").pos = [0., 0., 2.]
    return spec.compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics_hz", type=int, choices=(400, 800), required=True)
    args = parser.parse_args()
    output = HERE / f"mujoco{args.physics_hz}"
    output.mkdir(exist_ok=False)
    bank = dict(np.load(HERE / "inputs.npz"))
    substeps = args.physics_hz // 50
    metadata = {}
    for name in ("airborne", "ground"):
        fixed = name == "airborne"
        targets, rear = bank[f"{name}_target_rad"], bank[f"{name}_rear_nm"]
        n = targets.shape[1]
        template = base_model(fixed, args.physics_hz)
        fields = ("position_before_rad", "velocity_before_rad_s", "position_after_rad", "velocity_after_rad_s",
                  "computed_nm", "submitted_nm", "actuator_nm", "passive_nm", "dry_friction_nm")
        samples = {key: np.empty((400, substeps, n, 2)) for key in fields}
        samples["roll_rad"] = np.empty((400, substeps, n))
        samples["contacts"] = np.empty((400, substeps, n), dtype=int)
        configs = []
        for index in range(n):
            model = copy.copy(template)
            data = mujoco.MjData(model)
            joints = [model.joint(key).id for key in ("steering_joint", "rearwheel_joint")]
            dofs = model.jnt_dofadr[joints]
            qids = model.jnt_qposadr[joints]
            acts = [model.actuator(key).id for key in ("steering_act", "drive_act")]
            dummy = SimpleNamespace(model=model, _acts=acts, _steer_jid=joints[0], _rear_jid=joints[1],
                                    _limits=np.array([5.6, 6.28]))
            BikeBalanceEnv._configure_motors(dummy)
            model.dof_damping[dofs] = [c.STEER_VISCOUS_FRICTION, 0.]
            model.dof_frictionloss[dofs] = [c.STEER_FRICTION, .106]
            model.jnt_stiffness[joints] = 0.
            if fixed and index % 2:
                model.dof_damping[dofs[0]] = model.dof_frictionloss[dofs[0]] = 0.
            if not fixed:
                adr = int(model.joint("base_free").qposadr[0])
                data.qpos[adr:adr + 3] = [0., 0., -.016]
                data.qpos[adr + 3:adr + 7] = [1., 0., 0., 0.]
            data.qpos[qids], data.qvel[:] = 0., 0.
            mujoco.mj_forward(model, data)
            body_ids = np.flatnonzero(model.body_mass > 0)
            configs.append(dict(
                case=str(bank[f"{name}_names"][index]),
                body_names=[model.body(int(i)).name for i in body_ids],
                masses=model.body_mass[body_ids].tolist(), initial_com_world_m=data.xipos[body_ids].tolist(),
                initial_joint_position=data.qpos[qids].tolist(), damping=model.dof_damping[dofs].tolist(),
                frictionloss=model.dof_frictionloss[dofs].tolist(), fixed_base=fixed))
            for step in range(400):
                for sub in range(substeps):
                    slot = step, sub, index
                    samples["position_before_rad"][slot] = data.qpos[qids]
                    samples["velocity_before_rad_s"][slot] = data.qvel[dofs]
                    computed = np.array([10. * (float(targets[step, index]) - data.qpos[qids[0]])
                                         - .2 * data.qvel[dofs[0]], float(rear[step, index])])
                    data.ctrl[acts] = np.clip(computed, [-5.6, -6.28], [5.6, 6.28])
                    samples["computed_nm"][slot], samples["submitted_nm"][slot] = computed, data.ctrl[acts]
                    mujoco.mj_step(model, data)
                    samples["position_after_rad"][slot] = data.qpos[qids]
                    samples["velocity_after_rad_s"][slot] = data.qvel[dofs]
                    samples["actuator_nm"][slot] = data.qfrc_actuator[dofs]
                    samples["passive_nm"][slot] = data.qfrc_passive[dofs]
                    friction = np.zeros(model.nv)
                    force = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                                     data.efc_force, 0.)
                    mujoco.mj_mulJacTVec(model, data, friction, force)
                    samples["dry_friction_nm"][slot] = friction[dofs]
                    samples["roll_rad"][slot] = 0. if fixed else _roll_from_quat(data.qpos[adr + 3:adr + 7])
                    samples["contacts"][slot] = data.ncon
                    data.qvel[dofs] = np.clip(data.qvel[dofs], [-23.9, -61.6], [23.9, 61.6])
                # Match the production policy-boundary kinematic refresh.
                mujoco.mj_forward(model, data)
            print(name, index, "complete", flush=True)
        assert all(np.isfinite(value).all() for value in samples.values())
        if fixed:
            assert not samples["contacts"].any(), "Airborne motor bench must have zero contacts"
        np.savez_compressed(output / f"{name}.npz", **samples, target_rad=targets, rear_nm=rear,
                            names=bank[f"{name}_names"], physics_dt_s=1 / args.physics_hz)
        metadata[name] = configs
    metadata.update(completed=True, policy_hz=50, physics_hz=args.physics_hz, seed=42,
                    dr_resampling=False, policy_inference=False, automatic_resets=False,
                    version=mujoco.__version__, velocity_limit_mode="production hard projection")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
