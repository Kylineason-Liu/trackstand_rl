"""Charts for summaries/2026-11-04_*_trackstand_5L_no_gain_report.md (data: 2026-11-03_dr_ablation/figures_data.csv)."""
import csv, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
rows = list(csv.DictReader(open("../2026-11-03_dr_ablation/figures_data.csv")))
V = {(r["figure"], r["panel"], r["run"], r["metric"]): float(r["value"]) for r in rows}
B, O, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FIVE = ["5L-Base", "5L-Norm", "5L-Cos", "5L-NormCos", "5L-NormCosEnt", "5L-Base-DriftQuad", "5L-Base-WheelBand"]
PANELS = [("isaac_nodr", "Isaac, DR off"), ("isaac_dr", "Isaac, DR on"), ("mujoco_nodr", "MuJoCo, DR off"), ("mujoco_dr", "MuJoCo, DR on")]
L = lambda p, r: V[("episode_length", p, r, "len_s")]
out = [["figure", "panel", "run", "metric", "value"]]
data = {"panels": PANELS, "five": FIVE, "len": {}, "train": {}, "diag": {}}
for p, _ in PANELS:
    for r in ["2L-Base", "2L-NormCosEnt"] + FIVE:
        data["len"].setdefault(p, {})[r] = L(p, r)
for r in ["2L-Base", "2L-NormCosEnt"] + FIVE:
    data["train"][r] = V[("training", "train", r, "final_ep_len_steps")]
    data["diag"][r] = {k: V[("diagnostics", "mujoco_nodr", r, m)] for k, m in
                       (("jitter", "steering jitter mean |Δa| per step"), ("steer_deg", "mean |steering angle| (deg)"))}
json.dump(data, open("report_data.json", "w"), indent=1)

# 1) matched pair
fig, ax = plt.subplots(figsize=(10, 4.2))
y = np.arange(len(PANELS))
for off, run, c in ((-0.2, "2L-Base", B), (0.2, "5L-Base", O)):
    vals = [L(p, run) for p, _ in PANELS]
    ax.barh(y + off, vals, 0.38, color=c, label=run + (" [32,32]" if run.startswith("2L") else " [512,256,128,64]"))
    for yi, v in zip(y + off, vals):
        ax.text(v + 0.3, yi, f"{v:.1f} s", va="center", fontsize=9, color=INK)
    out += [["matched_pair", p, run, "len_s", L(p, run)] for p, _ in PANELS]
ax.set_yticks(y); ax.set_yticklabels([t for _, t in PANELS]); ax.invert_yaxis()
ax.axvline(30, ls="--", color=MUTED, lw=1); ax.text(30, -0.62, "30 s cap", ha="center", fontsize=8, color=MUTED)
ax.set_xlim(0, 33); ax.set_xlabel("mean episode length (s)")
ax.set_title("Only the network differs: [32,32] lasts longer in every interface", fontsize=11)
ax.legend(loc="lower right", frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("1_matched_pair.png", dpi=130); plt.close(fig)

# 2) every 5-layer run vs 2L-Base
fig, axes = plt.subplots(len(PANELS), 1, figsize=(10, 6.4), sharex=True)
for ax, (p, title) in zip(axes, PANELS):
    ref = L(p, "2L-Base"); vals = [L(p, r) for r in FIVE]
    ax.axvline(ref, color=B, lw=2); ax.text(ref - 0.3, 0.55, f"2L-Base {ref:.2f} s", color=B, fontsize=8.5, ha="right", va="bottom")
    ax.scatter(vals, np.zeros(len(vals)), s=70, color=O, edgecolor="white", linewidth=1.5, zorder=3)
    best = int(np.argmax(vals)); near = abs(vals[best] - ref) < 2.5
    ax.text(vals[best] + ((0.3 if vals[best] > ref else -0.3) if near else 0), -0.55, f"best 5L: {FIVE[best]} {vals[best]:.2f} s", fontsize=8,
            ha=("left" if vals[best] > ref else "right") if near else "center", va="top", color=INK)
    beats = sum(v > ref + 0.05 for v in vals); ties = sum(abs(v - ref) <= 0.05 for v in vals)
    tie = f" ({ties} {"tie" if ties == 1 else "ties"} within 0.05 s)" if ties else ""
    ax.set_title(f"{title}: {beats} of 7 five-layer runs beat 2L-Base{tie}", fontsize=9.5, loc="left")
    ax.set_ylim(-1.1, 1.1); ax.set_yticks([]); ax.set_xlim(-0.5, 32); ax.spines[["top", "right", "left"]].set_visible(False)
    out += [["all_5L", p, r, "len_s", v] for r, v in zip(FIVE, vals)] + [["all_5L", p, "2L-Base", "len_s", ref]]
axes[-1].set_xlabel("mean episode length (s); orange = each 5-layer run, blue line = 2L-Base")
fig.tight_layout(); fig.savefig("2_all_5L_vs_2L.png", dpi=130); plt.close(fig)

# 3) training score vs MuJoCo transfer
fig, ax = plt.subplots(figsize=(9, 5))
for r in ["2L-Base", "2L-NormCosEnt"] + FIVE:
    x, yv = data["train"][r], L("mujoco_nodr", r); c = B if r.startswith("2L") else O
    ax.scatter(x, yv, s=80, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(f"{r} ({x:.0f}, {yv:.1f} s)", (x, yv), xytext={"5L-NormCosEnt": (6, -13)}.get(r, (6, 5)), textcoords="offset points", fontsize=8, color=INK)
    out.append(["train_vs_mujoco", "mujoco_nodr", r, "train_final_steps|mujoco_len_s", f"{x}|{yv}"])
ax.set_xlabel("training: final mean episode length (steps of 1,500)"); ax.set_ylabel("MuJoCo DR off: mean episode length (s)")
ax.set_xlim(0, 1700); ax.set_ylim(0, 16.5); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Higher training scores (5-layer, orange) do not carry over to MuJoCo", fontsize=11)
fig.tight_layout(); fig.savefig("3_training_vs_mujoco.png", dpi=130); plt.close(fig)

# 4) steering behaviour in MuJoCo
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
runs = ["2L-Base"] + FIVE
for ax, key, lab in ((axes[0], "steer_deg", "mean |steering angle| (deg)"), (axes[1], "jitter", "steering jitter, mean |Δa| per step")):
    vals = [data["diag"][r][key] for r in runs]
    ax.barh(range(len(runs)), vals, color=[B] + [O] * len(FIVE), height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.0f}°" if key == "steer_deg" else f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        out.append(["steering", "mujoco_nodr", runs[i], key, v])
    ax.set_yticks(range(len(runs))); ax.set_yticklabels(runs, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel(lab); ax.set_xlim(0, max(vals) * 1.18); ax.spines[["top", "right"]].set_visible(False)
axes[1].set_yticklabels([])
fig.suptitle("MuJoCo DR off: 2L-Base steers least and most smoothly", fontsize=11)
fig.tight_layout(); fig.savefig("4_steering.png", dpi=130); plt.close(fig)
csv.writer(open("figures_data.csv", "w", newline="")).writerows(out)
print("ok")
