"""Small causal diagnostics of the historical 91850 MuJoCo controller.

All sources load from Git without checkout. Only this output folder is written.
These are first-episode diagnostics, not a reproduction of the 6057-episode run.
"""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path("/home/lmbike/trackstand_rl")
OUT = Path(__file__).resolve().parent
SNAPSHOT = "c4b2df7d63bf978ff784f909f872c7545e10ea58"
XML = ROOT / "mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
ONNX = ROOT / "mujoco_sim2sim/policies/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
REFERENCE = ROOT / "deploy/experiments/reports/20261015_deployment_failure_analysis/question_3_observations/mujoco_observations.npz"
SOURCES = {
    "env": "mujoco_sim2sim/src/newbike_mujoco/env.py",
    "policy": "mujoco_sim2sim/src/newbike_mujoco/policy.py",
    "viewer": "mujoco_sim2sim/scripts/run_deploy_viewer.py",
}


def git_source(path):
    return subprocess.check_output(["git", "show", f"{SNAPSHOT}:{path}"], cwd=ROOT, text=True)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")


def load_module(name, source):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(source, f"<{SNAPSHOT}:{name}>", "exec"), mod.__dict__)
    return mod


def model_and_env(case):
    env_source = git_source(SOURCES["env"])
    # Exact replacement points; baseline source remains byte-for-byte unchanged.
    if case in {"no_steer_target_bound", "no_steer_bound_or_slew", "all_action_relaxations"}:
        old = "desired = float(np.clip(action[0] * self.STEER_SCALE, -1.5708, 1.5708))"
        assert env_source.count(old) == 1
        env_source = env_source.replace(old, "desired = float(action[0] * self.STEER_SCALE)")
    if case in {"no_steer_slew", "no_steer_bound_or_slew", "all_action_relaxations"}:
        old = "self._steer_cmd += float(np.clip(desired - self._steer_cmd, -23.9 * 0.02, 23.9 * 0.02))"
        assert env_source.count(old) == 1
        env_source = env_source.replace(old, "self._steer_cmd = desired")
    if case in {"no_speed_projection", "all_action_relaxations"}:
        for dof, speed in [("_steer_dof", "23.9"), ("_rear_dof", "61.6")]:
            old = f"if abs(d.qvel[self.{dof}]) > {speed}:"
            assert env_source.count(old) == 1
            env_source = env_source.replace(old, "if False:")
    if case == "rear_startup_uncapped":
        old = "d.ctrl[1] = float(action[1]) * self.DRIVE_SCALE"
        assert env_source.count(old) == 1
        env_source = env_source.replace(old, old + "\n        self.model.actuator_ctrllimited[1] = self._step != 0\n        self.model.actuator_forcelimited[1] = self._step != 0")
    if case == "refresh_dr_constants":
        old = "self._apply_dr_episode()\n        d = self.data"
        assert env_source.count(old) == 1
        env_source = env_source.replace(old, "self._apply_dr_episode()\n            mujoco.mj_setConst(self.model, self.data)\n        d = self.data")
    env_module = load_module("historical_env", env_source)
    source = git_source(SOURCES["viewer"])
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "augment_trackstand_model"
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_TRACKSTAND_JOINT_RENAMES" for t in n.targets)]
    scope = {"mujoco": mujoco}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<historical augmentation>", "exec"), scope)
    spec = mujoco.MjSpec.from_file(str(XML))
    scope["augment_trackstand_model"](spec, str(XML.parent))
    if case == "physics_800":
        spec.option.timestep = 1 / 800
    if case == "implicitfast_400":
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The original viewer wrote spec.to_xml() and reloaded that XML. Preserve
    # the serialization step in memory: direct spec.compile() is NOT identical.
    model = mujoco.MjModel.from_xml_string(spec.to_xml())
    if case in {"no_steer_target_bound", "no_steer_bound_or_slew", "all_action_relaxations"}:
        model.actuator_ctrllimited[0] = False
    if case in {"rear_no_ctrl_clip", "rear_no_saturation", "all_action_relaxations"}:
        model.actuator_ctrllimited[1] = False
    if case in {"rear_no_saturation", "all_action_relaxations"}:
        model.actuator_forcelimited[1] = False
    return model, env_module, env_source


def run_case(case, dr):
    started = time.monotonic()
    label = case + ("_dr" if dr else "_no_dr")
    target = OUT / "serialized_runs" / (label + ".json")
    if target.exists():
        raise FileExistsError(target)
    model, mod, source = model_and_env(case)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX), sess_options=options, providers=["CPUExecutionProvider"])
    seeds = list(range(42, 106))
    envs = [mod.BikeBalanceEnv(model=model, seed=seed, enable_dr=dr) for seed in seeds]
    obs = np.stack([env.reset() for env in envs])
    initial_observations = obs.copy()
    initial_states = [dict(seed=s, qpos=e.data.qpos.copy(), qvel=e.data.qvel.copy(), dr=e.dr) for s, e in zip(seeds, envs)]
    active = np.ones(len(envs), dtype=bool)
    results = [None] * len(envs)
    peaks = np.zeros((len(envs), 2))
    traces = []
    for step in range(1500):
        actions = sess.run(None, {"observation": obs})[0]
        if not np.isfinite(actions[active]).all():
            raise RuntimeError(f"Nonfinite actor output in {label}")
        for i in np.flatnonzero(active):
            env = envs[i]
            peaks[i] = np.maximum(peaks[i], np.abs(actions[i] * [1.57, 6.28]))
            obs[i], done = env.step(actions[i])
            if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                raise RuntimeError(f"Nonfinite plant state in {label}, seed {seeds[i]}")
            warnings = [int(x) for x in env.data.warning.number]
            numerical_failure = any(warnings[int(getattr(mujoco.mjtWarning, name))]
                                    for name in ["mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL"])
            time_error = abs(env.data.time - (step + 1) * env.CTRL_DT)
            numerical_failure = numerical_failure or time_error > 1e-8
            if i < 8 and step < 10:
                traces.append(dict(seed=seeds[i], step=step, raw_action=actions[i].copy(),
                                   qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                                   actuator_force=env.data.actuator_force.copy()))
            if done or numerical_failure:
                results[i] = dict(seed=seeds[i], steps=step + 1, seconds=(step + 1) * .02,
                                  timeout=bool(env.stats.timeout) and not numerical_failure,
                                  death="numerical_failure" if numerical_failure else env.stats.death,
                                  numerical_failure=bool(numerical_failure), native_warning_counts=warnings,
                                  native_sim_time_s=env.data.time, time_error_s=time_error,
                                  peak_scaled_request=peaks[i], max_roll_deg=np.degrees(env.stats.roll_max),
                                  max_drive_speed=env.stats.drive_speed_max, drift_max=env.stats.drift_max)
                active[i] = False
                obs[i] = 0
        if not active.any():
            break
    assert all(r is not None for r in results)
    assert abs(envs[0].CTRL_DT - round(envs[0].CTRL_DT / model.opt.timestep) * model.opt.timestep) < 1e-12
    counts, _ = np.histogram([r["seconds"] for r in results], bins=np.arange(0, 31, 3))
    assert counts.sum() == len(seeds)
    result = dict(case=case, dr=dr, seeds=seeds, episodes=len(results), first_episode_only=True,
                  policy_hz=50, physics_hz=1 / model.opt.timestep, integrator=int(model.opt.integrator),
                  mean_seconds=float(np.mean([r["seconds"] for r in results])),
                  timeouts=sum(r["timeout"] for r in results), bin_counts=counts,
                  numerical_failures=sum(r["numerical_failure"] for r in results),
                  source_after_experiment_patch_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  initial_observations=initial_observations, initial_states=initial_states,
                  episodes_data=results, first_200ms_first_8_seeds=traces,
                  elapsed_seconds=time.monotonic() - started)
    if case == "baseline":
        with np.load(REFERENCE, allow_pickle=False) as ref:
            key = "dr" if dr else "no_dr"
            assert np.array_equal(initial_observations, ref[key + "_observation"][0]), "Initial inputs differ from historical capture"
        if dr:
            original = json.loads((OUT / "original_histogram_reconstruction.json").read_text())
            expected = np.array(original["first_episode_steps"])
            actual = np.array([r["steps"] for r in results])
            result["original_histogram_first_episode_matches"] = int((actual == expected).sum())
            assert np.array_equal(actual, expected), (label, actual.tolist(), expected.tolist())
            with np.load(ROOT / original["npz"]) as history:
                errors = [max(abs(t["actuator_force"][0] - history["steer"][t["step"], t["seed"] - 42]),
                              abs(t["actuator_force"][1] - history["drive"][t["step"], t["seed"] - 42]))
                          for t in traces]
            result["original_startup_force_max_error_nm"] = max(errors)
            assert max(errors) < 1e-10, max(errors)
    save(target, result)
    return {k: result[k] for k in ["case", "dr", "episodes", "mean_seconds", "timeouts", "bin_counts", "numerical_failures", "elapsed_seconds"]}


def main():
    (OUT / "serialized_runs").mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [XML, ONNX, REFERENCE]}
    source_hashes = {k: hashlib.sha256(git_source(v).encode()).hexdigest() for k, v in SOURCES.items()}
    save(OUT / "serialized_experiment.json", dict(snapshot=SNAPSHOT, sources=SOURCES, source_sha256=source_hashes,
         input_sha256=hashes, mujoco=mujoco.__version__, onnxruntime=ort.__version__,
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         note="Historical XML serialization/reload, 64 first episodes per case, seeds 42-105, no autoresets; not the original fixed-budget histogram population"))
    cases = [("baseline", dr) for dr in [False, True]]
    cases += [(name, True) for name in ["no_steer_slew", "no_steer_target_bound", "no_steer_bound_or_slew",
              "rear_no_ctrl_clip", "rear_no_saturation", "rear_startup_uncapped", "no_speed_projection",
              "physics_800", "implicitfast_400", "all_action_relaxations", "refresh_dr_constants"]]
    cases += [(name, False) for name in ["no_steer_bound_or_slew", "rear_startup_uncapped", "physics_800"]]
    summaries = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_case, *case) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result)
            print(json.dumps(result, default=serial), flush=True)
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    save(OUT / "serialized_summary.json", sorted(summaries, key=lambda r: (r["dr"], r["case"])))


if __name__ == "__main__":
    main()
