"""Exercise v2 CLI budgets, relative paths and viewer logic without a GUI/GPU."""

from contextlib import nullcontext, redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco

OUT = Path(__file__).resolve().parent
COMPONENT = OUT.parents[2]
sys.path.insert(0, str(COMPONENT / "src"))
from newbike_mujoco_v2 import contract as c


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, COMPONENT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_script("run_deploy_eval_v2")
viewer = load_script("run_deploy_viewer_v2")
results = {}

for module in (evaluation, viewer):
    with (OUT / (module.__name__ + "_help.txt")).open("w") as stream, redirect_stdout(stream), patch.object(
            sys, "argv", [module.__name__, "--help"]):
        try:
            module.main()
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("--help did not exit")


def evaluate(label, options, expected_status):
    original = Path.cwd()
    relative_model = os.path.relpath(c.DEFAULT_MJCF, original)
    directory = OUT / label
    argv = [evaluation.__name__, "--mjcf", relative_model, "--output_dir", os.path.relpath(directory, original)]
    with (OUT / (label + ".log")).open("w") as stream, redirect_stdout(stream), patch.object(
            sys, "argv", argv + options):
        status = evaluation.main()
    assert status == expected_status and Path.cwd() == original
    metadata_path = next(directory.glob("*_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    assert Path(metadata["runtime_dir"]).parent == directory
    return metadata, metadata_path


quota, path = evaluate("quota_smoke", ["--num_envs", "2", "--episodes", "5", "--steps", "3"], 0)
assert quota["episode_quotas"] == quota["completed_per_env"] == [3, 2]
assert quota["environment_transitions"] == 15 and len(quota["episodes"]) == 5
assert quota["completed_policy_steps"] == 9 and quota["in_progress_lengths_steps"] == [0, 0]
results["quota"] = {"metadata": str(path), "episodes": 5, "quota": [3, 2], "transitions": 15}

partial, path = evaluate("partial_smoke", ["--num_envs", "2", "--episodes", "4", "--steps", "3",
                                           "--max_policy_steps", "4"], 2)
assert partial["completed_per_env"] == [1, 1] and partial["in_progress_lengths_steps"] == [1, 1]
assert not partial["completed"] and len(partial["episodes"]) == 2
results["partial_budget"] = {"metadata": str(path), "status": 2, "unfinished_lengths": [1, 1]}

fixed, path = evaluate("fixed_smoke", ["--num_envs", "1", "--num_steps", "7"], 0)
assert fixed["environment_transitions"] == 7 and not fixed["episodes"]
assert fixed["in_progress_lengths_steps"] == [7]
results["fixed_budget"] = {"metadata": str(path), "transitions": 7, "unfinished_length": 7}


class FakeViewer:
    def __init__(self, model, _data):
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.user_scn = mujoco.MjvScene(model, maxgeom=10000)
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.steps < 3

    def sync(self):
        self.steps += 1


original = Path.cwd()
with (OUT / "viewer_smoke.log").open("w") as stream, redirect_stdout(stream), patch.object(
        sys, "argv", [viewer.__name__, "--num_envs", "2", "--speed", "100000",
                      "--output_dir", str(OUT / "viewer_smoke")]), patch.object(
        viewer.mujoco.viewer, "launch_passive", FakeViewer):
    viewer.main()
assert Path.cwd() == original
assert list((OUT / "viewer_smoke").glob("*_torque.png"))
results["viewer"] = {"environments": 2, "policy_steps": 3, "real_GUI_opened": False,
                     "physics_and_scene_assembly_executed": True}
results.update(passed=True, original_working_directory_restored=True,
               help_checks=2, gpu=False, hardware=False)
(OUT / "cli_validation.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
