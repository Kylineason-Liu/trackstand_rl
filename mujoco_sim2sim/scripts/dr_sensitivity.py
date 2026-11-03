#!/usr/bin/env python3
"""MuJoCo DR sensitivity: how far must each DR term shrink before DR-on matches DR-off?

Each of the six v1 MuJoCo DR terms gets a scale s (1 = current contract range, 0 = term off):

    mass      body mass x U(1 - 0.2 s, 1 + 0.2 s)          (contract DR_MASS 0.8..1.2)
    com       COM offset U(+-0.005 s) m                     (DR_COM 5 mm)
    friction  tire friction U(0.65 - 0.15 s, 0.65 + 0.15 s) (DR_FRICTION 0.5..0.8, nominal 0.65)
    gains     actuator Kp/Kd x U(1 - 0.1 s, 1 + 0.1 s)      (DR_GAINS 0.9..1.1)
    init      initial roll U(+-0.10 s) rad, roll rate U(+-0.25 s) rad/s
    imu       IMU turn-on bias sigma x s (gyro 0.0002 rad/s, accel 0.002)

DR stays enabled for every scaled config (s = 0 draws degenerate ranges), so all scaled configs consume
the same random-number stream; "off" is the real enable_dr=False reference. The scales are applied by
patching newbike_mujoco.contract in each worker process before the envs are built (env.py reads them at
runtime); no source file changes. Same env / ViewerEnvGroup / DeployedPolicy / terminations as
run_deploy_eval.py, without the trace archive.

    python3 scripts/dr_sensitivity.py --policy 5L-Norm=/abs/policy.onnx --policy Pi=/abs/pi.onnx \\
        --plan sweep --out reports/<date>_dr_sensitivity --workers 22
    python3 scripts/dr_sensitivity.py ... --plan custom --config "rec:mass=0.5,com=1,init=0.25"

Run with Isaac python (~/IsaacLab/_isaac_sim/python.sh): it has mujoco + onnxruntime.
Writes <out>/results.jsonl (one row per policy x config; reruns skip rows already present).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS = ("mass", "com", "friction", "gains", "init", "imu")
SCALES = (0.25, 0.5, 0.75, 1.0)


def scaled_contract(c, s: dict) -> None:
    """Patch contract DR ranges in place (this process only)."""
    c.DR_MASS = (1 - 0.2 * s["mass"], 1 + 0.2 * s["mass"])
    c.DR_COM = 0.005 * s["com"]
    c.DR_FRICTION = (0.65 - 0.15 * s["friction"], 0.65 + 0.15 * s["friction"])
    c.DR_GAINS = (1 - 0.1 * s["gains"], 1 + 0.1 * s["gains"])
    c.DR_INIT_ROLL = (-0.10 * s["init"], 0.10 * s["init"])
    c.DR_INIT_ROLL_RATE = (-0.25 * s["init"], 0.25 * s["init"])
    c.DR_GYRO_RES = 0.0002 * s["imu"]
    c.DR_ACCEL_RES = 0.002 * s["imu"]


def plan_sweep() -> list[dict]:
    ones, zeros = dict.fromkeys(TERMS, 1.0), dict.fromkeys(TERMS, 0.0)
    cfgs = [dict(name="off", dr=False, scales=zeros, seed=42), dict(name="off_seed43", dr=False, scales=zeros, seed=43),
            dict(name="all", dr=True, scales=ones), dict(name="zero", dr=True, scales=zeros)]
    for t in TERMS:
        for s in SCALES:
            cfgs.append(dict(name=f"only_{t}@{s:g}", dr=True, scales={**zeros, t: s}))
        cfgs.append(dict(name=f"all_except_{t}", dr=True, scales={**ones, t: 0.0}))
    for s in (0.25, 0.5, 0.75):
        cfgs.append(dict(name=f"all@{s:g}", dr=True, scales=dict.fromkeys(TERMS, s)))
    return cfgs


def parse_custom(spec: str) -> dict:
    """'name:mass=0.5,com=1' (unlisted terms = 0)."""
    name, _, body = spec.partition(":")
    scales = dict.fromkeys(TERMS, 0.0)
    for kv in filter(None, body.split(",")):
        k, v = kv.split("=")
        if k not in TERMS:
            raise SystemExit(f"unknown term {k}; use {TERMS}")
        scales[k] = float(v)
    return dict(name=name, dr=True, scales=scales)


def run_one(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import onnxruntime as ort

    _orig = ort.InferenceSession

    def _single_thread(path, sess_options=None, **kw):  # 20+ workers: keep each session on one core
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        return _orig(path, sess_options=so, **kw)

    ort.InferenceSession = _single_thread
    from newbike_mujoco import DeployedPolicy, build_env_from_mjcf
    from newbike_mujoco import contract as c
    from newbike_mujoco.viewer_envs import ViewerEnvGroup

    scaled_contract(c, job["scales"])
    seed = job.get("seed", job["seed_default"])
    policy = DeployedPolicy(onnx_path=job["onnx"])
    first, _ = build_env_from_mjcf(str(c.DEFAULT_MJCF), seed, job["dr"], velocity_limit_mode="project")
    group = ViewerEnvGroup(first, job["num_envs"], seed)
    t0, obs, recs = time.monotonic(), group.reset(), []
    for _ in range(job["num_steps"]):
        obs, done, _ = group.step(policy.act_batch(obs))
        for i in np.flatnonzero(done):
            st = group.envs[i].stats
            recs.append((st.length, bool(st.timeout), {k: bool(v) for k, v in st.terminations.items()},
                         float(np.sqrt(st.roll_sq / max(st.length, 1)))))
            obs[i] = group.envs[i].reset()
    n = max(len(recs), 1)
    lens = np.array([r[0] for r in recs]) * c.CONTROL_DT
    return dict(policy=job["policy"], config=job["name"], dr=job["dr"], scales=job["scales"], seed=seed,
                episodes=len(recs), mean_len_s=float(lens.mean()) if recs else float("nan"),
                median_len_s=float(np.median(lens)) if recs else float("nan"),
                timeout_rate=sum(r[1] for r in recs) / n,
                terms={k: sum(r[2].get(k, False) for r in recs) / n for k in c.TERMINATIONS},
                roll_rms_deg=float(np.rad2deg(np.sqrt(np.mean([r[3] ** 2 for r in recs])))) if recs else float("nan"),
                wall_s=time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, help="name=/abs/path/policy.onnx (repeatable)")
    ap.add_argument("--plan", choices=("sweep", "custom"), default="sweep")
    ap.add_argument("--config", action="append", default=[], help="custom plan: 'name:term=s,...' (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--num_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = out / "results.jsonl"
    done = set()
    if res.exists():
        done = {(r["policy"], r["config"]) for r in map(json.loads, res.read_text().splitlines()) if r}
    cfgs = plan_sweep() if a.plan == "sweep" else [parse_custom(s) for s in a.config]
    jobs = []
    for spec in a.policy:
        name, _, path = spec.partition("=")
        for cfg in cfgs:
            if (name, cfg["name"]) not in done:
                jobs.append(dict(cfg, policy=name, onnx=path, num_envs=a.num_envs, num_steps=a.num_steps,
                                 seed_default=a.seed))
    print(f"[INFO] {len(jobs)} jobs ({len(done)} already in {res}), {a.workers} workers", flush=True)
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, res.open("a") as f:
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k}/{len(jobs)} {time.monotonic() - t0:6.0f}s] {r['policy']:<18} {r['config']:<22} "
                  f"len {r['mean_len_s']:5.1f} s  TO {100 * r['timeout_rate']:3.0f}%  n={r['episodes']}", flush=True)
    print("[INFO] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
