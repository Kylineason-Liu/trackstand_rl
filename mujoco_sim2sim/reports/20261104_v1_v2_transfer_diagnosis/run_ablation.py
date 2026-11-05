"""One-process diagnostic overrides around the unchanged native v2 evaluator."""

import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
MUJOCO = ROOT / "mujoco_sim2sim"
sys.path.insert(0, str(MUJOCO / "src"))
from newbike_mujoco_v2 import contract as c

CASES = {
    "nominal": {},
    "zero_initial": {"DR_INIT_ROLL": (0., 0.), "DR_INIT_ROLL_RATE": (0., 0.)},
    "zero_com": {"DR_COM": 0.},
    "zero_lateral_com": {"DR_COM": [.005, 0., .005]},
    "only_lateral_com": {"DR_COM": [0., .005, 0.]},
    "nominal_mass": {"DR_MASS": (1., 1.)},
    "nominal_contact": {"DR_FRICTION": (.65, .65)},
    "zero_imu_residual": {"DR_GYRO_RES": 0., "DR_ACCEL_RES": 0.},
    "no_speed_projection": {},
    "zero_integral": {},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    output = REPORT / args.case
    output.mkdir(exist_ok=False)
    from newbike_mujoco_v2.env import BikeBalanceEnv
    original_init = BikeBalanceEnv.__init__

    def constructed(self, *pos, **kwargs):
        original_init(self, *pos, **kwargs)
        if args.case == "zero_integral":
            self.integral.gain_nm_per_rad = 0.

    # Avoid many unused ORT worker threads when independent CPU cases coexist.
    import onnxruntime as ort
    original_session = ort.InferenceSession

    def session(*pos, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        kwargs["sess_options"] = options
        return original_session(*pos, **kwargs)

    argv = [str(MUJOCO / "scripts/run_deploy_eval_v2.py"), "--onnx",
            str(MUJOCO / "policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx"),
            "--num_envs", "64", "--episodes", "64", "--max_policy_steps", "1500",
            "--seed", "42", "--physics_hz", "1200", "--pure_survival", "--output_dir", str(output)]
    if args.case != "nominal":
        argv.append("--dr")
    if args.case == "no_speed_projection":
        argv += ["--velocity_limit_mode", "none"]
    (output / "case.json").write_text(json.dumps(dict(
        case=args.case, constants_overridden=CASES[args.case], native_argv=argv,
        runtime_integral_gain_override=0. if args.case == "zero_integral" else None,
        thread_limits="ONNX Runtime intra/inter-op=1; numerical controller unchanged",
        production_source_modified=False, multiagent_used=False), indent=2) + "\n")
    # Patch only this process. The standard loader still validates the original
    # trained action contract; case.json separately declares each counterfactual.
    overrides = {key: np.asarray(value) if key == "DR_COM" and isinstance(value, list) else value
                 for key, value in CASES[args.case].items()}
    with patch.multiple(c, **overrides) if overrides else patch.object(c, "TASK", c.TASK):
        with patch.object(BikeBalanceEnv, "__init__", constructed), \
                patch.object(ort, "InferenceSession", session), patch.object(sys, "argv", argv):
            namespace = runpy.run_path(argv[0], run_name="diagnostic_module")
            result = namespace["main"]()
    if result != 0:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
