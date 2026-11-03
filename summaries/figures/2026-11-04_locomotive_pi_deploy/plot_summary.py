#!/usr/bin/env python3
"""Figures for summaries/2026-11-04_*_locomotive_pi_deploy.md (exact value printed on every mark).

Run with Isaac python (system matplotlib fails under numpy 2):

    ~/IsaacLab/_isaac_sim/python.sh summaries/figures/2026-11-04_locomotive_pi_deploy/plot_summary.py

Reads transfer_report.json of the deployed rank-0 run (Isaac + MuJoCo paper suites, and the CycleRL
original 1900_drive03 rerun for reference). Writes PNGs + figures_data.csv beside this script.
"""

import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(REPO, "isaaclab_train/logs/skrl/locomotive/Locomotive-Cylinder-Aligned-v0/"
                   "2026-10-29_17-06-47_ppo_torch_locomotive_ppo")
REPORT = json.load(open(os.path.join(RUN, "transfer_report.json")))
PORT_KEY, CYCLERL_KEY = list(REPORT)

C = ["#2a78d6", "#eb6834"]  # reference palette slots 1-2: Isaac, MuJoCo
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})
ROWS = []  # (figure, panel, item, series, metric, value)

MODES = [("paper_flat", "flat"), ("paper_no_noise", "no noise"), ("paper_dropout", "dropout"),
         ("paper_normal_speed", "normal speed"), ("paper_low_speed", "low speed"),
         ("paper_gravel", "gravel"), ("paper_max_noise", "max noise"),
         ("paper_rough", "rough"), ("paper_slope", "slope"), ("paper_steps", "steps")]
SIMS = [("isaac", "Isaac Lab"), ("mujoco", "MuJoCo")]


def metric(key, sim, mode, name):
    value = REPORT[key][sim]["modes"].get(mode, {}).get(name)
    return math.nan if value is None else float(value)


def grouped(ax, labels, series, fmt, xmax, figure, panel, metric_name):
    y = np.arange(len(labels))
    h = 0.38
    for i, (name, values) in enumerate(series):
        pos = y - h / 2 + i * h
        for p, v, lab in zip(pos, values, labels):
            ROWS.append((figure, panel, lab, name, metric_name, "" if math.isnan(v) else f"{v:.6g}"))
            if math.isnan(v):
                ax.text(0.5, p, "not modelled", va="center", ha="left", color=MUTED, fontsize=8)
                continue
            ax.barh(p, v, height=h - 0.04, color=C[i], edgecolor=SURFACE, linewidth=2, label=None)
            ax.text(v + xmax * 0.012, p, fmt(v), va="center", ha="left", color=INK2, fontsize=8)
        ax.barh(np.nan, 0, color=C[i], label=name)
    ax.set_yticks(y, labels)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)


def fig_modes():
    labels = [lab for _, lab in MODES]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    bsr = [(name, [100 * metric(PORT_KEY, sim, m, "bsr") for m, _ in MODES]) for sim, name in SIMS]
    ste = [(name, [metric(PORT_KEY, sim, m, "ste_deg") for m, _ in MODES]) for sim, name in SIMS]
    grouped(axes[0], labels, bsr, lambda v: f"{v:.1f}%", 118, "fig1_modes", "BSR", "bsr_pct")
    axes[0].set_title("Balance success rate (BSR, %) — higher is better")
    grouped(axes[1], labels, ste, lambda v: f"{v:.2f}°", 82, "fig1_modes", "STE", "ste_deg")
    axes[1].set_title("Heading RMSE (STE, °) — lower is better")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2, frameon=False)
    fig.suptitle("Deployed rank-0 locomotive policy (Locomotive-Cylinder-Aligned-v0, 2026-10-29_17-06-47): "
                 "paper suite, 1024 episodes per mode", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(HERE, "fig1_paper_modes.png"), dpi=150)
    plt.close(fig)


def fig_transfer():
    runs = [(PORT_KEY, "Port (deployed, rear Kd 0.2)"), (CYCLERL_KEY, "CycleRL original (rear Kd 5)")]
    labels = [lab for _, lab in runs]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    ste = [(name, [metric(k, sim, "paper_flat", "ste_deg") for k, _ in runs]) for sim, name in SIMS]
    gravel = [(name, [100 * metric(k, sim, "paper_gravel", "bsr") for k, _ in runs]) for sim, name in SIMS]
    grouped(axes[0], labels, ste, lambda v: f"{v:.2f}°", 15, "fig2_transfer", "flat STE", "ste_deg")
    axes[0].set_title("Flat heading RMSE (STE, °) — lower is better")
    grouped(axes[1], labels, gravel, lambda v: f"{v:.1f}%", 118, "fig2_transfer", "gravel BSR", "bsr_pct")
    axes[1].set_title("Gravel balance success (BSR, %)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Transfer vs CycleRL original (same 1900_drive03 recipe)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(HERE, "fig2_transfer_vs_cyclerl.png"), dpi=150)
    plt.close(fig)


fig_modes()
fig_transfer()
with open(os.path.join(HERE, "figures_data.csv"), "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["figure", "panel", "item", "series", "metric", "value"])
    writer.writerows(ROWS)
print(f"wrote {len(ROWS)} values to {HERE}")
