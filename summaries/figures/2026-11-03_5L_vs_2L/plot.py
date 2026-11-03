import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
src = "../2026-11-03_dr_ablation/figures_data.csv"
rows = list(csv.DictReader(open(src)))
get = lambda fig, panel, run, metric: float(next(r["value"] for r in rows if (r["figure"], r["panel"], r["run"], r["metric"]) == (fig, panel, run, metric)))
panels = [("isaac_dr", "Isaac, DR on"), ("mujoco_nodr", "MuJoCo, DR off"), ("mujoco_dr", "MuJoCo, DR on")]
groups = [("Base recipe\n(only network differs)", "2L-Base", "5L-Base"),
          ("Norm+Cos+Ent recipe\n(only network differs)", "2L-NormCosEnt", "5L-NormCosEnt"),
          ("Best of each\n(5L-Norm adds obs norm)", "2L-Base", "5L-Norm")]
out = [["panel", "group", "run", "len_s"]]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
for ax, (p, title) in zip(axes, panels):
    x = np.arange(len(groups)); w = 0.36
    for k, (label, two, five) in enumerate(groups):
        for off, run, color in ((-w/2, two, "#3a7bd5"), (w/2, five, "#e0803a")):
            v = get("episode_length", p, run, "len_s"); out.append([p, label.split("\n")[0], run, v])
            ax.bar(k + off, v, w, color=color, label=("[32,32]" if off < 0 else "5-layer") if k == 0 else None)
            ax.text(k + off, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{g[0]}\n{g[1]} vs {g[2]}" for g in groups], fontsize=8)
    ax.axhline(30, ls="--", color="gray", lw=1); ax.set_title(title); ax.set_ylim(0, 34)
axes[0].set_ylabel("mean episode length (s), cap 30 s"); axes[0].legend(loc="upper right")
fig.suptitle("Trackstand v1, all trained with --dr, seed 42, 7,500 updates: [32,32] vs 5-layer", fontsize=12)
fig.tight_layout(); fig.savefig("episode_length_pairs.png", dpi=130)
csv.writer(open("figures_data.csv", "w", newline="")).writerows(out)
