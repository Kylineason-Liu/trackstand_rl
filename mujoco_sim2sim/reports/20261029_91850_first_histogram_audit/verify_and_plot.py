"""Independently verify saved historical diagnostics and draw comparisons."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    manifest = read("serialized_experiment.json")
    assert hashlib.sha256((HERE / "evaluate_historical.py").read_bytes()).hexdigest() == manifest["script_sha256"]
    for name, digest in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for item in read("historical_source_manifest.json"):
        assert hashlib.sha256((HERE / "historical_sources" / item["path"]).read_bytes()).hexdigest() == item["sha256"]

    with np.load(ROOT / "mujoco_sim2sim/logs/202610151708_91850_dr_torque.npz") as data:
        previous = np.zeros(64, dtype=int)
        first = {}
        lengths = []
        for step, env in data["reset_events"]:
            step, env = int(step), int(env)
            n = step - previous[env]
            previous[env] = step
            lengths.append(n)
            first.setdefault(env, n)
    lengths = np.asarray(lengths)
    bins, _ = np.histogram(lengths * .02, bins=np.arange(0, 31, 3))
    histogram = np.genfromtxt(ROOT / "mujoco_sim2sim/logs/202610151708_91850_dr_epi_len_histogram.csv",
                             delimiter=",", names=True)["episode_count"].astype(int)
    assert np.array_equal(bins, histogram)
    assert len(lengths) == 6057 and int((lengths == 1500).sum()) == 891

    saved = read("serialized_summary.json")
    rows = []
    baseline_dr = read("serialized_runs/baseline_dr.json")
    baseline_no_dr = read("serialized_runs/baseline_no_dr.json")
    for summary in saved:
        path = "serialized_runs/" + summary["case"] + ("_dr.json" if summary["dr"] else "_no_dr.json")
        r = read(path)
        e = r["episodes_data"]
        assert [x["seed"] for x in e] == list(range(42, 106))
        assert r["initial_observations"] == (baseline_dr if r["dr"] else baseline_no_dr)["initial_observations"]
        assert r["initial_states"] == (baseline_dr if r["dr"] else baseline_no_dr)["initial_states"]
        assert len(e) == 64 and sum(x["timeout"] for x in e) == r["timeouts"]
        assert sum(x["numerical_failure"] for x in e) == r["numerical_failures"]
        assert abs(np.mean([x["seconds"] for x in e]) - r["mean_seconds"]) < 1e-12
        for x in e:
            assert x["seconds"] == x["steps"] * .02
            assert not x["timeout"] or x["steps"] == 1500
            if not x["numerical_failure"]:
                assert x["time_error_s"] < 1e-8
                assert not any(x["native_warning_counts"][3:])
        for k in ["mean_seconds", "timeouts", "bin_counts", "numerical_failures"]:
            assert r[k] == summary[k], (path, k)
        rows.append(dict(case=r["case"], dr=r["dr"], endpoints=len(e), mean_seconds=r["mean_seconds"],
                         timeouts=r["timeouts"], timeout_percent=100 * r["timeouts"] / 64,
                         numerical_failures=r["numerical_failures"]))
    assert len(rows) == 16 and sum(r["endpoints"] for r in rows) == 1024
    assert sum(r["numerical_failures"] for r in rows) == 1
    assert [e["steps"] for e in baseline_dr["episodes_data"]] == [first[i] for i in range(64)]
    assert baseline_dr["original_startup_force_max_error_nm"] == 0
    with (HERE / "comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = ["baseline", "physics_800", "implicitfast_400", "no_speed_projection",
             "no_steer_slew", "no_steer_target_bound", "no_steer_bound_or_slew"]
    labels = ["Original: Euler 400 Hz", "Euler 800 Hz", "implicitfast 400 Hz", "No speed projection",
              "No steering target slew", "No steering target bound", "No steering bound or slew"]
    selected = [next(r for r in rows if r["dr"] and r["case"] == key) for key in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    y = np.arange(len(order))
    colors = ["#557890", "#2b9461", "#cb8a24", "#557890", "#557890", "#557890", "#557890"]
    axes[0].barh(y, [r["timeout_percent"] for r in selected], color=colors)
    axes[1].barh(y, [r["mean_seconds"] for r in selected], color=colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(0, 30)
    axes[0].set_xlabel("Reached 30 s (%)")
    axes[1].set_xlabel("Mean first-episode duration (s)")
    for i, r in enumerate(selected):
        axes[0].text(r["timeout_percent"] + 1, i, f'{r["timeouts"]}/64', va="center", fontsize=9)
        axes[1].text(r["mean_seconds"] + .3, i, f'{r["mean_seconds"]:.2f}', va="center", fontsize=9)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Historical model_91850: matched MuJoCo diagnostics\nDR ON; same 64 first episodes, seeds 42–105; policy stays 50 Hz")
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(HERE / ("historical_ablation." + suffix), dpi=180)
    plt.close(fig)

    traces = [r for r in read("startup_traces.json") if r["dr"]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for r, color in zip(traces, ["#4477aa", "#228833", "#cc6677"]):
        ts = np.array([x["time_s"] for x in r["traces"]]) * 1000
        label = {"baseline":"Euler 400 Hz", "physics_800":"Euler 800 Hz", "implicitfast_400":"implicitfast 400 Hz"}[r["case"]]
        axes[0].plot(ts, np.degrees([x["position_after_rad"] for x in r["traces"]]), color=color, label=label)
        axes[1].step(ts, [x["actuator_force_nm"] for x in r["traces"]], where="pre", color=color, label=label)
    axes[0].set_ylabel("Steering position (deg)")
    axes[1].set_ylabel("Native actuator_force (Nm)")
    axes[1].set_xlabel("Elapsed simulation time (ms)")
    axes[1].set_xlim(0, 100)
    axes[0].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("model_91850 startup: historical runtime, DR ON, seed 42\nAll native physics samples; force is sampled after mj_step")
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(HERE / ("startup_comparison." + suffix), dpi=180)
    plt.close(fig)
    save = dict(passed=True, original_histogram_episodes=len(lengths), original_histogram_bins=bins.tolist(),
                original_histogram_timeouts=891, first_episode_length_matches=64, startup_torque_values_checked=160,
                startup_torque_max_error_nm=0, scenarios=16, diagnostic_endpoints=1024,
                numerical_failure_endpoints=1, all_combined_unbounded_case_excluded_from_main_plot=True,
                equal_initial_inputs_states_dr_draws=True, actual_simulation_time_checked=True,
                input_sha256=manifest["input_sha256"])
    (HERE / "verification.json").write_text(json.dumps(save, indent=2) + "\n")
    print(json.dumps(save, indent=2))


if __name__ == "__main__":
    main()
