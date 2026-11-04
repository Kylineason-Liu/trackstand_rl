"""Charts for summaries/2026-11-04_*_trackstand_v3_heading_penalty.md.
Run with Isaac python (system matplotlib fails under numpy 2): ~/IsaacLab/_isaac_sim/python.sh plot.py"""
import csv, glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

Q = os.path.expanduser("~/bike_test/trackstand_rl/isaaclab_train/logs/v3_queue")
TB = {"2L-Base (v1)": os.path.expanduser("~/bike_test/trackstand_rl_test_layer/isaaclab_train/logs/rsl_rl/"
                                         "trackstand_rl_abl_2l_base/2026-11-02_20-57-25_dr"),
      "v3": os.path.expanduser("~/bike_test/trackstand_rl/isaaclab_train/logs/rsl_rl/trackstand_rl_v3_rsl/2026-11-04_00-03-22_dr")}
# 2L-Base with the v1 yaw termination: 2026-11-03 DR ablation (summaries/figures/2026-11-04_5L_no_gain/figures_data.csv).
ABL = {"isaac_nodr": 30.0, "isaac_dr": 25.35, "mujoco_nodr": 13.12, "mujoco_dr": 5.39}
B, B2, O, O2, INK, MUTED = "#2a78d6", "#8fb8ea", "#eb6834", "#f4b392", "#0b0b0b", "#52514e"
TERMS = ["fallen", "drifted", "yaw_deviation", "wheel_net", "time_out"]
TCOL = {"fallen": "#eb6834", "drifted": "#2a78d6", "yaw_deviation": "#e3b021", "wheel_net": "#7a4fc2", "time_out": "#c9c7c1"}
out = [["figure", "panel", "run", "metric", "value"]]


def isaac(path):
    t = open(path).read()
    g = lambda pat: [float(x) for x in re.search(pat, t).groups()]
    return {"len_s": g(r"mean_episode_length\s*:\s*[\d.]+ steps\s*\(\s*([\d.]+) s")[0],
            "timeout": g(r"timeout_rate \(survived\):\s*([\d.]+)")[0],
            "roll_rms": g(r"roll_RMS\s*:\s*([\d.]+)")[0],
            "drift_mean": g(r"drift mean/max\s*:\s*([\d.]+)")[0],
            "yaw_mean": g(r"yaw mean/max\s*:\s*([\d.]+)")[0],
            "turn": g(r"turn/episode mean\s*:\s*([\d.]+)")[0]}


def mujoco(d):
    s = json.load(open(glob.glob(f"{Q}/{d}/*_summary.json")[0]))
    n = max(s["completed_episodes"], 1)
    z = np.load(glob.glob(f"{Q}/{d}/*_evaluation.npz")[0])
    a, o = z["raw_action"], z["observation"]
    return {"len_s": s["mean_episode_length_s"], "timeout": s["timeout_rate"],
            "terms": {k: s["termination_counts"].get(k, 0) / n for k in TERMS},
            "steer_deg": float(np.rad2deg(np.abs(o[..., 5])).mean()), "yaw_rate": float(np.abs(o[..., 4]).mean()),
            "jitter": float(np.abs(np.diff(a[..., 0], axis=0)).mean()), "drift_mean": s["mean_drift_m"]}


R = {"isaac": {("v3", "nodr"): isaac(f"{Q}/v3/isaac_nodr/eval.txt"), ("v3", "dr"): isaac(f"{Q}/v3/isaac_dr/eval.txt"),
               ("2L", "nodr"): isaac(f"{Q}/ref_2L-Base/isaac_nodr_noyaw/eval.txt"),
               ("2L", "dr"): isaac(f"{Q}/ref_2L-Base/isaac_dr_noyaw/eval.txt")},
     "mujoco": {(r, m, t): mujoco(f"{'v3' if r == 'v3' else 'ref_2L-Base'}/mujoco_{m}_{t}")
                for r, m, t in [("v3", "nodr", "noyaw"), ("v3", "dr", "noyaw"), ("v3", "nodr", "v1terms"),
                                ("v3", "dr", "v1terms"), ("2L", "nodr", "noyaw"), ("2L", "dr", "noyaw")]}}

# 1) episode length
PANELS = [("isaac_nodr", "Isaac, DR off"), ("isaac_dr", "Isaac, DR on"), ("mujoco_nodr", "MuJoCo, DR off"), ("mujoco_dr", "MuJoCo, DR on")]
def L(series, panel):
    sim, m = panel.split("_")
    if series == "2L-yaw":
        return ABL[panel]
    if series == "2L-noyaw":
        return R[sim][("2L", m)]["len_s"] if sim == "isaac" else R[sim][("2L", m, "noyaw")]["len_s"]
    if series == "v3-noyaw":
        return R[sim][("v3", m)]["len_s"] if sim == "isaac" else R[sim][("v3", m, "noyaw")]["len_s"]
    return None if sim == "isaac" else R[sim][("v3", m, "v1terms")]["len_s"]
SER = [("2L-yaw", "2L-Base (v1), yaw termination ON", B), ("2L-noyaw", "2L-Base (v1), yaw termination OFF", B2),
       ("v3-noyaw", "v3, yaw termination OFF (as trained)", O), ("v3-yaw", "v3, yaw termination ON (v1 MuJoCo default)", O2)]
fig, ax = plt.subplots(figsize=(10, 5.6))
y = np.arange(len(PANELS)); h = 0.2
for k, (key, lab, c) in enumerate(SER):
    for i, (p, _) in enumerate(PANELS):
        v = L(key, p)
        if v is None:
            if k == 3: ax.text(0.3, y[i] + (k - 1.5) * h, "n/a (Isaac v3 has no yaw termination)", va="center", fontsize=8, color=MUTED)
            continue
        ax.barh(y[i] + (k - 1.5) * h, v, h * 0.9, color=c, label=lab if i == 2 else None)
        ax.text(v + 0.3, y[i] + (k - 1.5) * h, f"{v:.2f} s", va="center", fontsize=8.5, color=INK, bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
        out.append(["episode_length", p, key, "len_s", round(v, 3)])
ax.set_yticks(y); ax.set_yticklabels([t for _, t in PANELS]); ax.invert_yaxis()
ax.axvline(30, ls="--", color=MUTED, lw=1); ax.set_xlim(0, 34); ax.set_xlabel("mean episode length (s), 64 envs x 3000 steps, seed 42")
ax.set_title("v3 matches 2L-Base in Isaac but lasts 3.3 s vs 13.1 s in MuJoCo", fontsize=11)
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("1_episode_length.png", dpi=130); plt.close(fig)

# 2) what v3 learned: spin in place
MET = [("Isaac DR on: heading turned per episode (deg)", lambda r: R["isaac"][(r, "dr")]["turn"], "{:.0f}°"),
       ("Isaac DR on: mean drift from spawn (m)", lambda r: R["isaac"][(r, "dr")]["drift_mean"], "{:.3f} m"),
       ("MuJoCo DR off: mean |steering angle| (deg)", lambda r: R["mujoco"][(r, "nodr", "noyaw")]["steer_deg"], "{:.1f}°"),
       ("MuJoCo DR off: mean |yaw rate| (rad/s)", lambda r: R["mujoco"][(r, "nodr", "noyaw")]["yaw_rate"], "{:.2f}"),
       ("MuJoCo DR off: steering jitter |Δa| per step", lambda r: R["mujoco"][(r, "nodr", "noyaw")]["jitter"], "{:.3f}")]
fig, axes = plt.subplots(len(MET), 1, figsize=(10, 6.2))
for ax, (lab, f, fmt) in zip(axes, MET):
    vals = [f("2L"), f("v3")]
    ax.barh([0, 1], vals, 0.7, color=[B, O])
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, fmt.format(v), va="center", fontsize=9, color=INK)
        out.append(["behaviour", lab, ["2L-Base", "v3"][i], "value", round(v, 4)])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["2L-Base", "v3"]); ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title(lab, fontsize=9.5, loc="left"); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=8)
fig.suptitle("v3 pivots in place: bar far over, heading keeps turning, drift stays small (both evals without yaw termination)", fontsize=10.5)
fig.tight_layout(); fig.savefig("2_behaviour.png", dpi=130); plt.close(fig)

# 3) training reward per alive second
tr = {}
for name, run in TB.items():
    e = EventAccumulator(run, size_guidance={"scalars": 0}); e.Reload()
    fin = lambda t: (lambda v: float(v[-max(1, int(len(v) * 0.013)):].mean()))(np.array([x.value for x in e.Scalars(t)]))
    ep = fin("Train/mean_episode_length")
    scale = 1500.0 / ep  # Episode_Reward = episode sum / 30 s; rescale to per second alive
    tr[name] = {"ep_len": ep, **{k: fin(f"Episode_Reward/{k}") * scale for k in ("survival", "drift_gauss", "rate_penalty")},
                "yaw_penalty": fin("Episode_Reward/yaw_penalty") * scale if name == "v3" else 0.0}
    tr[name]["heading+drift"] = tr[name]["drift_gauss"] + tr[name]["yaw_penalty"]
fig, ax = plt.subplots(figsize=(10, 3.9))
cats = ["drift_gauss", "yaw_penalty", "heading+drift"]; x = np.arange(len(cats))
for k, (name, c) in enumerate((("2L-Base (v1)", B), ("v3", O))):
    vals = [tr[name][m] for m in cats]
    ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=c, label=f"{name} (final ep length {tr[name]['ep_len']:.0f} steps)")
    for xi, v in zip(x + (k - 0.5) * 0.36, vals):
        ax.text(xi, v + (0.04 if v >= 0 else -0.04), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    out += [["training_reward_per_alive_s", m, name, "value", round(tr[name][m], 4)] for m in cats + ["survival", "rate_penalty", "ep_len"]]
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["drift_gauss (max +2.0)", "yaw_penalty (v3 only)", "drift + yaw"])
ax.set_ylim(-1.15, 2.3); ax.set_ylabel("reward per second alive\n(end of training)")
ax.set_title("Pivoting earns +1.10/s more drift reward and costs only 0.81/s in yaw penalty (net +0.29/s)", fontsize=10.5)
ax.legend(loc="upper right", frameon=False, fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("3_training_reward.png", dpi=130); plt.close(fig)

# 4) MuJoCo terminations (no yaw termination)
ROWS = [("2L-Base, DR off", ("2L", "nodr", "noyaw")), ("v3, DR off", ("v3", "nodr", "noyaw")),
        ("2L-Base, DR on", ("2L", "dr", "noyaw")), ("v3, DR on", ("v3", "dr", "noyaw"))]
fig, ax = plt.subplots(figsize=(10, 3.4))
for i, (lab, key) in enumerate(ROWS):
    left = 0
    for t in TERMS:
        v = 100 * R["mujoco"][key]["terms"][t]
        if v <= 0: continue
        ax.barh(i, v, 0.7, left=left, color=TCOL[t], edgecolor="white", linewidth=1.5, label=t if i == 0 or (t not in ("drifted", "wheel_net", "time_out", "fallen")) else None)
        if v >= 6: ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color=INK)
        out.append(["mujoco_terminations_noyaw", lab, t, "share_pct", round(v, 2)]); left += v
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r for r, _ in ROWS]); ax.invert_yaxis()
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
ax.legend(u.values(), u.keys(), ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8.5)
ax.set_xlim(0, 100); ax.set_xlabel("share of completed episodes (%; terms may overlap)")
ax.set_title("In MuJoCo, v3 falls (97% / 87%); 2L-Base mostly drifts or spins up the wheel", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("4_mujoco_terminations.png", dpi=130); plt.close(fig)

csv.writer(open("figures_data.csv", "w", newline="")).writerows(out)
json.dump({"isaac": {f"{a}_{b}": v for (a, b), v in R["isaac"].items()},
           "mujoco": {"_".join(k): v for k, v in R["mujoco"].items()}, "training": tr}, open("report_data.json", "w"), indent=1)
print(json.dumps(tr, indent=1))
