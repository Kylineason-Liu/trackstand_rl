"""Independently verify plotted-data exports, extrema, pages and preserved inputs."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE.parent / "20261103_v2_parallel_backend/validation/no_dr/202611032149_26600_no_dr_evaluation.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(data, kind, row, env):
    tp, th = np.arange(1500) / 50, np.arange(36000) / 1200
    def held(t, y, dt):
        return np.r_[t, t[-1] + dt], np.r_[y, y[-1]]
    if kind == "demand":
        if row == 0:
            return [held(tp, data["raw_action"][:, env, 0] * np.float32(1.57), .02),
                    held(tp, data["held_steering_target_rad"][:, env], .02),
                    (np.r_[0., tp + .02], np.r_[data["observation"][0, env, 5], data["position_rad"][:, env, 0]])]
        return [held(tp, data["raw_action"][:, env, 1] * np.float32(61.6), .02),
                held(tp, data["held_drive_target_rad_s"][:, env], .02),
                (np.r_[th, 30.], np.r_[data["physics_drive_velocity_rad_s"][:, env].ravel(),
                                      data["velocity_rad_s"][-1, env, 1]])]
    if kind == "observation":
        return [(tp, data["observation"][:, env, row])]
    if kind == "raw":
        return [held(tp, data["raw_action"][:, env, row], .02)]
    if kind == "speed":
        return [(tp + .02, data["velocity_rad_s"][:, env, 0])] if row == 0 else [
            (th, data["physics_drive_velocity_rad_s"][:, env].ravel())]
    if kind == "torque":
        values = []
        if row == 1:
            values += [held(th, .3 * data["physics_drive_error_rad_s"][:, env].ravel(), 1 / 1200),
                       held(th, data["physics_drive_integral_nm"][:, env].ravel(), 1 / 1200)]
        return values + [held(th, data["physics_computed_torque_nm"][:, env, :, row].ravel(), 1 / 1200),
                         (th + 1 / 1200, data["physics_motor_torque_nm"][:, env, :, row].ravel())]
    raise AssertionError(kind)


def panel_matches(panel, traces):
    assert panel["legend_location"] == "lower right" and panel["samples_outside_y_view"] == 0
    low, high = panel["window_s"]
    assert panel["spawn_times_s"] == ([0.0] if low == 0 else [])
    assert len(panel["extrema"]) == len(traces)
    for metadata, (time, value) in zip(panel["extrema"].values(), traces):
        selection = (time >= low - 1e-10) & (time <= high + 1e-10)
        t, v = time[selection], value[selection]
        assert len(v) == metadata["plotted_samples"]
        assert float(v.min()) == metadata["minimum"]
        assert float(v.max()) == metadata["maximum"]
        assert abs(float(t[v.argmin()]) - metadata["minimum_time_s"]) < 1e-12
        assert abs(float(t[v.argmax()]) - metadata["maximum_time_s"]) < 1e-12
        assert panel["ylim"][0] <= v.min() and panel["ylim"][1] >= v.max()


def main():
    with np.load(RAW, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    summary = json.loads((HERE / "summary.json").read_text())
    settings = json.loads((HERE / "plot_settings.json").read_text())
    provenance = json.loads((HERE / "provenance.json").read_text())
    for name, expected in provenance["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert summary["capture_checks"]["physics_samples"] == 2304000
    assert data["observation"].shape == (1500, 64, 11)
    assert np.all(data["active_mask"])
    panels = 0
    for name, figure in settings.items():
        if figure["kind"] == "envelope":
            values = [data["position_rad"][..., 0], data["velocity_rad_s"][..., 1],
                      data["physics_motor_torque_nm"][..., 0], data["physics_motor_torque_nm"][..., 1]]
            for panel, value in zip(figure["panels"], values):
                assert panel["all64_minimum"] == float(value.min())
                assert panel["all64_maximum"] == float(value.max())
                assert panel["ylim"][0] < value.min() and panel["ylim"][1] > value.max()
                assert panel["legend_location"] == "lower right"
                panels += 1
        elif "pages" in figure:
            assert [page["env_id"] for page in figure["pages"]] == list(range(64))
            for page in figure["pages"]:
                for row, panel in enumerate(page["panels"]):
                    panel_matches(panel, records(data, figure["kind"], row, page["env_id"]))
                    panels += 1
        else:
            for row, panel in enumerate(figure["panels"]):
                panel_matches(panel, records(data, figure["kind"], row, 0))
                panels += 1
    assert panels == 885

    policy = np.genfromtxt(HERE / "env0_policy_traces.csv", delimiter=",", names=True)
    physics = np.genfromtxt(HERE / "env0_physics_traces.csv", delimiter=",", names=True)
    assert policy.shape == (1500,) and physics.shape == (36000,)
    policy_expected = {
        "policy_time_s": np.arange(1500) * .02,
        "response_time_s": np.arange(1500) * .02 + .02,
        "raw_steering": data["raw_action"][:, 0, 0], "raw_drive_velocity": data["raw_action"][:, 0, 1],
        "scaled_raw_steering_rad": data["raw_action"][:, 0, 0] * np.float32(1.57),
        "scaled_raw_drive_velocity_rad_s": data["raw_action"][:, 0, 1] * np.float32(61.6),
        "held_steering_target_rad": data["held_steering_target_rad"][:, 0],
        "held_drive_target_rad_s": data["held_drive_target_rad_s"][:, 0],
        "actual_steering_position_rad": data["position_rad"][:, 0, 0],
        "actual_rear_position_rad": data["position_rad"][:, 0, 1],
        "actual_steering_velocity_rad_s": data["velocity_rad_s"][:, 0, 0],
        "actual_rear_velocity_rad_s": data["velocity_rad_s"][:, 0, 1],
    }
    for name, value in policy_expected.items():
        np.testing.assert_array_equal(policy[name], value)
    observation_fields = list(policy.dtype.names)[12:]
    assert len(observation_fields) == 11
    np.testing.assert_array_equal(np.column_stack([policy[key] for key in observation_fields]), data["observation"][:, 0])
    th = (np.arange(1500)[:, None] * .02 + np.arange(24)[None, :] / 1200).ravel()
    physics_expected = {
        "controller_time_s": th, "physics_end_time_s": th + 1 / 1200,
        "rear_velocity_feedback_rad_s": data["physics_drive_velocity_rad_s"][:, 0].ravel(),
        "rear_velocity_error_rad_s": data["physics_drive_error_rad_s"][:, 0].ravel(),
        "rear_P_torque_nm": .3 * data["physics_drive_error_rad_s"][:, 0].ravel(),
        "rear_I_torque_nm": data["physics_drive_integral_nm"][:, 0].ravel(),
        "steering_preclip_torque_nm": data["physics_computed_torque_nm"][:, 0, :, 0].ravel(),
        "rear_preclip_torque_nm": data["physics_computed_torque_nm"][:, 0, :, 1].ravel(),
        "steering_applied_torque_nm": data["physics_motor_torque_nm"][:, 0, :, 0].ravel(),
        "rear_applied_torque_nm": data["physics_motor_torque_nm"][:, 0, :, 1].ravel(),
    }
    for name, value in physics_expected.items():
        np.testing.assert_array_equal(physics[name], value)
    with (HERE / "observation_extrema_all64.csv").open() as stream:
        extrema = list(csv.DictReader(stream))
    assert len(extrema) == 64 * 11
    for item in extrema:
        values = data["observation"][:, int(item["env_id"]), int(item["observation_index"]) - 1]
        assert float(item["minimum"]) == float(values.min()) and float(item["maximum"]) == float(values.max())

    pdf_pages = {}
    for name in summary["figures"]:
        for suffix in ("png", "svg", "pdf"):
            path = HERE / f"{name}.{suffix}"
            assert path.stat().st_size > 1000
        with Image.open(HERE / f"{name}.png") as image:
            assert image.width >= 1800 and image.height >= 1300
        tree = ET.parse(HERE / f"{name}.svg")
        text = " ".join(tree.getroot().itertext())
        assert "no DR" in text and "26600" in text
        assert "min " in text and "max " in text
    for stem in ("all64_demand_vs_actuation", "all64_observation_traces"):
        path = HERE / f"{stem}.pdf"
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        count = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        assert count == 64
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        assert [int(item) for item in re.findall(r"env\s+(\d+)/63", text)] == list(range(64))
        pdf_pages[stem] = count
    links = re.findall(r'(?:href|src)="([^"]+)"', (HERE / "index.html").read_text())
    assert all((HERE / name).exists() for name in links)
    result = dict(passed=True, independent_panel_count=panels,
                  csv_policy_rows=1500, csv_physics_rows=36000,
                  csv_observation_extrema_rows=len(extrema), csv_columns_checked=33,
                  actual_input_policy_observation_samples=96000,
                  actual_physics_samples=2304000, figure_sets_checked=len(summary["figures"]),
                  large_pdf_pages=pdf_pages, local_html_links_checked=len(links),
                  protected_input_files_hash_identical=len(provenance["input_sha256"]),
                  no_simulation_or_hardware=True,
                  verified_source="Original NPZ and separate metadata/CSV/PDF/SVG checks, without importing analyze.py",
                  manually_reviewed_pngs=["mujoco_demand_vs_actuation.png", "mujoco_motor_torque_flow_startup.png",
                                         "observation_traces.png"])
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
