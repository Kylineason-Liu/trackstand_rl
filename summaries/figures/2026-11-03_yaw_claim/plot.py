import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("../2026-11-03_dr_ablation/figures_data.csv")))
runs = ["2L-Base", "5L-Base", "5L-Norm", "5L-Base-DriftQuad"]
terms = [("time_out", "#9aa5b1"), ("yaw_deviation", "#e6b422"), ("fallen", "#e0803a"),
         ("drifted", "#3a7bd5"), ("wheel_net", "#8e5bc4")]
panels = [("isaac_dr", "Isaac, DR on"), ("mujoco_nodr", "MuJoCo, DR off"), ("mujoco_dr", "MuJoCo, DR on")]
val = {(r["panel"], r["run"], r["metric"]): float(r["value"]) for r in rows if r["figure"] == "terminations"}
out = [["panel", "run", "term", "percent"]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, (p, title) in zip(axes, panels):
    for i, run in enumerate(runs):
        left = 0.0
        for t, c in terms:
            v = val.get((p, run, t), 0.0); out.append([p, run, t, v])
            ax.barh(i, v, left=left, color=c, label=t if (i == 0 and p == "isaac_dr") else None)
            if v >= 9: ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
            left += v
    ax.set_title(title); ax.set_xlim(0, 100); ax.set_xlabel("% of episodes ended by")
axes[0].set_yticks(range(len(runs))); axes[0].set_yticklabels(runs); axes[0].invert_yaxis()
fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("How episodes end (trackstand v1, trained with --dr, seed 42)", fontsize=12)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("terminations_4runs.png", dpi=130)
csv.writer(open("figures_data.csv", "w", newline="")).writerows(out)
