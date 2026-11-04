#!/usr/bin/env python3
"""Charts + tables for the MuJoCo DR sensitivity sweep (scripts/dr_sensitivity.py results.jsonl).

    ~/IsaacLab/_isaac_sim/python.sh scripts/plot_dr_sensitivity.py \\
        --results reports/2026-11-03_dr_sensitivity/results.jsonl \\
        --out ../summaries/figures/2026-11-04_mujoco_dr_sensitivity

Metric: mean completed-episode length as % of the same policy's DR-off run ("off", seed 42).
A scale s "passes" for a policy when that one term at s (others 0) keeps >= PASS of DR-off; a term's
safe scale is the largest grid s such that it and every smaller grid s pass (noise-robust).
Run with Isaac python (system matplotlib fails under numpy 2).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TERMS = ("mass", "com", "friction", "gains", "init", "imu")
TERM_LABEL = {"mass": "mass ×U(1±0.2s)", "com": "COM ±5s mm", "friction": "tire friction 0.65±0.15s",
              "gains": "actuator gains ×U(1±0.1s)", "init": "init roll ±0.10s rad / rate ±0.25s",
              "imu": "IMU bias σ ×s"}
PASS = 0.90
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TERM_ORDER = [("time_out", "timeout (success)"), ("fallen", "fallen"), ("drifted", "drifted > 0.5 m"),
              ("yaw_deviation", "yaw > 30°"), ("wheel_net", "wheel_net"), ("diverged", "diverged")]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
ROWS = []  # (figure, policy, config, metric, value)


def load(path):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    pols = list(dict.fromkeys(r["policy"] for r in rows))
    return pols, {(r["policy"], r["config"]): r for r in rows}


def pct(d, p, cfg):
    ref = "off_seed43" if cfg.endswith("_s43") else "off"  # seed-43 configs vs the seed-43 DR-off run
    return 100 * d[(p, cfg)]["mean_len_s"] / d[(p, ref)]["mean_len_s"]


def curve(d, p, t):
    """[(s, %off)] for one term, s = 0 from the 'zero' config (DR on, every range 0)."""
    pts = [(0.0, pct(d, p, "zero"))]
    for (pp, cfg), r in d.items():
        if pp == p and cfg.startswith(f"only_{t}@"):
            pts.append((float(cfg.split("@")[1]), pct(d, p, cfg)))
    return sorted(pts)


def safe_scale(pts):
    s_ok = 0.0
    for s, v in pts:
        if s == 0:
            continue
        if v < 100 * PASS:
            break
        s_ok = s
    return s_ok


def fig_dose(pols, d, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharey=True)
    safe = {}
    for ax, t in zip(axes.flat, TERMS):
        ends = []
        for j, p in enumerate(pols):
            pts = curve(d, p, t)
            x, y = zip(*pts)
            ax.plot(x, y, color=C[j], lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1.2,
                    label=f"{p} (off {d[(p, 'off')]['mean_len_s']:.1f} s)")
            safe[(p, t)] = safe_scale(pts)
            ends.append([y[-1], j])
            for s, v in pts:
                ROWS.append(("dose_response", p, f"only_{t}@{s:g}" if s else "zero", "pct_of_off", round(v, 1)))
        ends.sort()
        for k in range(1, len(ends)):  # de-overlap endpoint labels
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 6.5)
        for yl, j in ends:
            ax.text(1.03, yl, f"{curve(d, pols[j], t)[-1][1]:.0f}%", va="center", fontsize=8.2, color=INK)
        ax.axhline(100 * PASS, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.axhline(100, color=GRID, lw=1.2)
        s_min = min(safe[(p, t)] for p in pols)
        ax.set_title(f"{TERM_LABEL[t]}\nsafe s (all 4 ≥ 90%): {s_min:g}", fontsize=10)
        ax.set_xlim(-0.03, 1.14)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim(0, 135)
        ax.set_xlabel("scale s of this term alone (1 = current contract range, others 0)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean episode length, % of DR-off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=4, frameon=False, bbox_to_anchor=(0.995, 0.995), fontsize=8.8)
    fig.suptitle("One DR term at a time (dashed = 90% of DR-off; labels = value at s = 1)", x=0.01,
                 ha="left", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "dose_response.png"), dpi=150)
    plt.close(fig)
    return safe


def fig_safe(pols, safe, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(TERMS))
    h = 0.19
    for j, p in enumerate(pols):
        v = [safe[(p, t)] for t in TERMS]
        ax.barh(y + (j - 1.5) * h, v, height=h, color=C[j], edgecolor=SURFACE, linewidth=1.2, label=p)
        for i, vi in enumerate(v):
            ax.text(vi + 0.012, y[i] + (j - 1.5) * h, f"{vi:g}" if vi else "< grid", va="center", fontsize=7.4,
                    color=INK)
            ROWS.append(("safe_scale", p, TERMS[i], "max_s_ge_90pct", vi))
    ax.set_yticks(y, [TERM_LABEL[t] for t in TERMS])
    ax.set_ylim(len(TERMS) - 0.5, -0.6)
    ax.set_xlim(0, 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("largest scale s that keeps ≥ 90% of DR-off, term alone (0 = fails at smallest grid s)")
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=8.2, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Safe scale per DR term and policy", pad=22)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "safe_scale.png"), dpi=150)
    plt.close(fig)


def fig_grid(pols, d, cfgs, out, name, title):
    cfgs = [c for c in cfgs if all((p, c) in d for p in pols)]
    m = np.array([[pct(d, p, c) for p in pols] for c in cfgs])
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(cfgs) + 1.6))
    ax.imshow(np.clip(m, 0, 110), cmap="Blues", vmin=0, vmax=140, aspect="auto")
    for i, c in enumerate(cfgs):
        for j, p in enumerate(pols):
            r = d[(p, c)]
            ok = m[i, j] >= 100 * PASS
            ax.text(j, i, f"{r['mean_len_s']:.1f} s · {m[i, j]:.0f}%{'' if ok else '  ✗'}  TO {100 * r['timeout_rate']:.0f}%",
                    ha="center", va="center", fontsize=8.2, color="white" if m[i, j] >= 93 else INK,
                    fontweight="bold" if ok else "normal")
            ROWS.append((name, p, c, "mean_len_s", round(r["mean_len_s"], 2)))
            ROWS.append((name, p, c, "pct_of_off", round(m[i, j], 1)))
            ROWS.append((name, p, c, "timeout_pct", round(100 * r["timeout_rate"], 1)))
    ax.set_xticks(range(len(pols)), pols)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(cfgs)), cfgs)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title + "   (mean length · % of DR-off · timeout share; ✗ = below 90%)", pad=24, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
    plt.close(fig)


def fig_terms(pols, d, cfgs, out):
    cfgs = [c for c in cfgs if all((p, c) in d for p in pols)]
    fig, axes = plt.subplots(1, len(pols), figsize=(16, 0.5 * len(cfgs) + 1.8), sharey=True)
    for ax, p in zip(axes, pols):
        y = np.arange(len(cfgs))
        left = np.zeros(len(cfgs))
        for ci, (term, _) in enumerate(TERM_ORDER):
            v = np.array([100 * d[(p, c)]["terms"].get(term, 0.0) for c in cfgs])
            ax.barh(y, v, left=left, height=0.62, color=C[ci], edgecolor=SURFACE, linewidth=2)
            for i, (vi, li) in enumerate(zip(v, left)):
                if vi >= 9:
                    ax.text(li + vi / 2, i, f"{vi:.0f}", ha="center", va="center", fontsize=7.8,
                            color="white" if ci in (0, 5, 6) else INK)
                ROWS.append(("terminations", p, cfgs[i], term, round(vi, 1)))
            left += v
        ax.set_yticks(y, cfgs)
        ax.set_ylim(len(cfgs) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(p)
        ax.set_xlabel("share of completed episodes (%)")
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[i]) for i in range(len(TERM_ORDER))]
    fig.legend(handles, [t[1] for t in TERM_ORDER], loc="upper right", ncol=6, frameon=False,
               bbox_to_anchor=(0.995, 0.99), fontsize=8.5)
    fig.suptitle("How episodes end (labels ≥ 9%)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "terminations.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pols, d = load(a.results)
    safe = fig_dose(pols, d, a.out)
    fig_safe(pols, safe, a.out)
    rec = [c for c in ("fric_imu", "recC", "recA", "recSafe", "recSafe_s43", "recB", "recB_s43") if (pols[0], c) in d]
    fig_grid(pols, d, ["off", "off_seed43", "zero", "all", "all@0.75", "all@0.5", "all@0.25"]
             + [f"all_except_{t}" for t in TERMS], a.out, "combined", "All terms together / leave one out")
    if rec:
        fig_grid(pols, d, ["off", "off_seed43", "all", "all@0.25"] + rec, a.out, "candidates",
                 "Combined reduced-DR candidates")
    fig_terms(pols, d, ["off", "zero", "all", "all@0.25"] + rec, a.out)
    with open(os.path.join(a.out, "figures_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "policy", "config", "metric", "value"])
        w.writerows(ROWS)
    print("safe scale (min over policies):", {t: min(safe[(p, t)] for p in pols) for t in TERMS})
    for p in pols:
        print(p, {t: safe[(p, t)] for t in TERMS})


if __name__ == "__main__":
    main()
