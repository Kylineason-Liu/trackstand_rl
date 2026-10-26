"""Read-only force decomposition around the unchanged production evaluator.

Repeat the original 64-environment DR run, then require all 13 production
archive arrays to be byte-identical before accepting the extra force data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = REPORT.parents[2]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco.viewer_envs import ViewerEnvGroup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics_hz", type=int, choices=(400, 800), required=True)
    args = parser.parse_args()
    tag = f"mujoco{args.physics_hz}"
    output = HERE / tag
    output.mkdir(exist_ok=False)
    shape = (3000, args.physics_hz // 50, 64, 2)
    fields = ("actuator_nm", "passive_nm", "dry_friction_nm", "all_constraint_nm")
    arrays = {key: np.empty(shape) for key in fields}
    slots = {"step": 0, "active": False, "sub": np.zeros(64, dtype=int)}
    by_data, configurations = {}, []
    initialize, group_step, physics_step = ViewerEnvGroup.__init__, ViewerEnvGroup.step, mujoco.mj_step

    def init(group, *positional, **keywords):
        initialize(group, *positional, **keywords)
        for index, env in enumerate(group.envs):
            by_data[id(env.data)] = index, env
            model = env.model
            configurations.append(dict(
                env=index, damping=model.dof_damping[env._dofs].tolist(),
                frictionloss=model.dof_frictionloss[env._dofs].tolist(),
                disableflags=int(model.opt.disableflags),
                solref_friction=model.dof_solref[env._dofs].tolist(),
                solimp_friction=model.dof_solimp[env._dofs].tolist(),
            ))

    def advance(model, data):
        if not slots["active"]:
            return physics_step(model, data)
        index, env = by_data[id(data)]
        slot = slots["step"], slots["sub"][index], index
        velocity = data.qvel[env._dofs].copy()
        physics_step(model, data)
        # Frictionloss is a CONSTRAINT, not part of qfrc_passive. Isolate its
        # generalized force by multiplying only friction rows by Jacobian^T.
        force = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                         data.efc_force, 0.)
        friction = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, data, friction, force)
        arrays["actuator_nm"][slot] = data.qfrc_actuator[env._dofs]
        arrays["passive_nm"][slot] = data.qfrc_passive[env._dofs]
        arrays["dry_friction_nm"][slot] = friction[env._dofs]
        arrays["all_constraint_nm"][slot] = data.qfrc_constraint[env._dofs]
        np.testing.assert_allclose(data.qfrc_passive[env._dofs],
                                   -model.dof_damping[env._dofs] * velocity, rtol=0, atol=1e-12)
        assert np.all(np.abs(friction[env._dofs]) <= model.dof_frictionloss[env._dofs] + 1e-10)
        slots["sub"][index] += 1

    def step(group, actions):
        slots["sub"][:] = 0
        slots["active"] = True
        try:
            result = group_step(group, actions)
        finally:
            slots["active"] = False
        assert (slots["sub"] == shape[1]).all()
        slots["step"] += 1
        return result

    script = ROOT / "mujoco_sim2sim/scripts/run_deploy_eval.py"
    checkpoint = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
    protected = [script, checkpoint, ROOT / "progress_record.md", ROOT / "progress_record.txt"]
    protected += list((ROOT / "mujoco_sim2sim/src/newbike_mujoco").glob("*.py"))
    protected += [ROOT / "mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    argv = [str(script), "--pt", str(checkpoint), "--dr", "--num_envs", "64", "--num_steps", "3000",
            "--seed", "42", "--physics_hz", str(args.physics_hz), "--output_dir", str(output)]
    with patch.object(ViewerEnvGroup, "__init__", init), patch.object(ViewerEnvGroup, "step", step), \
            patch.object(mujoco, "mj_step", advance), patch.object(sys, "argv", argv):
        runpy.run_path(str(script), run_name="capture_force_decomposition")["main"]()
    assert slots["step"] == shape[0]
    with np.load(next(output.glob("*_evaluation.npz"))) as new, \
            np.load(next((REPORT / tag).glob("*_evaluation.npz"))) as old:
        assert set(new.files) == set(old.files)
        for key in old.files:
            a, b = old[key], new[key]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(output / "forces.npz", **arrays)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    metadata = dict(completed=True, original_13_arrays_byte_identical=True, production_sources_unchanged=True,
                    motor_order=["steering", "rear"], configurations=configurations, source_hashes=hashes,
                    archive_fields=fields, invocation=argv,
                    meaning="Actuator + passive damping + isolated joint dry friction excludes contact/inertia/bias")
    (output / "verification.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Verified read-only friction decomposition:", output, flush=True)


if __name__ == "__main__":
    main()
