"""Shared viewer/headless evaluation reports; actuator effort is not net joint wrench."""

import os
import json
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .env import BikeBalanceEnv
from .contract import TERMINATIONS


def summarize(ep_records, num_envs=1):
    """Completed-episode statistics; overlapping term activations stay separate."""
    stats = [record[2] for record in ep_records]
    lengths = np.asarray([st.length for st in stats], dtype=float)
    n, steps = len(stats), sum(st.length for st in stats)
    return dict(
        environments=num_envs, completed_episodes=n,
        mean_episode_length_steps=float(lengths.mean()) if n else None,
        mean_episode_length_s=float(lengths.mean()*BikeBalanceEnv.CTRL_DT) if n else None,
        min_episode_length_s=float(lengths.min()*BikeBalanceEnv.CTRL_DT) if n else None,
        max_episode_length_s=float(lengths.max()*BikeBalanceEnv.CTRL_DT) if n else None,
        std_episode_length_steps=float(lengths.std()) if n else None,
        mean_reward=float(np.mean([st.reward for st in stats])) if n else None,
        timeout_rate=sum(st.timeout for st in stats)/n if n else None,
        termination_counts={name: sum(bool(st.terminations.get(name, False)) for st in stats) for name in TERMINATIONS},
        roll_rms_deg=float(np.degrees(np.sqrt(sum(st.roll_sq for st in stats)/steps))) if steps else None,
        max_roll_deg=float(np.degrees(max(st.roll_max for st in stats))) if n else None,
        mean_worst_lean_deg=float(np.degrees(np.mean([st.roll_max for st in stats]))) if n else None,
        mean_drift_m=sum(st.drift_sum for st in stats)/steps if steps else None,
        max_drift_m=max((st.drift_max for st in stats), default=None),
        mean_yaw_deg=sum(st.yaw_sum for st in stats)/steps if steps else None,
        max_yaw_deg=max((st.yaw_max for st in stats), default=None),
        mean_turn_deg=float(np.mean([st.turn for st in stats])) if n else None,
        mean_drive_speed_rad_s=sum(st.drive_speed_sum for st in stats)/steps if steps else None,
        max_drive_speed_rad_s=max((st.drive_speed_max for st in stats), default=None),
        mean_steer_speed_rad_s=sum(st.steer_speed_sum for st in stats)/steps if steps else None,
        max_steer_speed_rad_s=max((st.steer_speed_max for st in stats), default=None),
    )

def write_metrics_log(ep_records, log_path, num_envs=1, timing=None):
    """Write accurate step-weighted completed-episode metrics, human and JSON."""
    summary = summarize(ep_records, num_envs)
    summary.update(timing or {})
    lines = ["===== TRACKSTAND-BALANCE-V1 METRICS (MuJoCo) ====="]
    for key, value in summary.items():
        if key == "termination_counts":
            lines.append("  termination activations (may overlap):")
            lines.extend(f"    {name:22s}: {count}" for name, count in value.items())
        else:
            lines.append(f"  {key:30s}: {value}")
    lines.append("  Metrics include terminal state BEFORE caller reset; unfinished episodes excluded.")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w") as stream:
        stream.write("\n".join(lines)+"\n")
    with open(log_path.removesuffix(".log")+".json", "w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("\n"+"\n".join(lines))
    print(f"[INFO] metrics written: {log_path}")


def _evaluation_output_base(output_dir, policy_path, enable_dr):
    """Choose one timestamp/model/DR prefix for all reports, preserving earlier runs."""
    policy_name = os.path.basename(policy_path)
    model = re.search(r"model_(\d+)", policy_name)
    model_id = model.group(1) if model else os.path.splitext(policy_name)[0]
    stem = f"{time.strftime('%Y%m%d%H%M')}_{model_id}_{'dr' if enable_dr else 'no_dr'}"
    out_base = os.path.join(output_dir, stem)
    endings = ("_epi_len_histogram.png", "_epi_len_histogram.csv", "_torque.png", "_torque.npz",
               "_summary.log", "_summary.json", "_evaluation.npz", "_metadata.json")
    suffix = 2
    while any(os.path.exists(out_base + ending) for ending in endings):
        out_base = os.path.join(output_dir, f"{stem}_{suffix}")
        suffix += 1
    return out_base


def _plot_episode_length_distribution(ep_records, out_base, policy_path, enable_dr):
    """Count completed episodes in [0, 3), ..., [27, 30] s; save PNG and counts."""
    lengths_s = np.asarray([record[0] for record in ep_records]) * BikeBalanceEnv.CTRL_DT
    edges_s = np.arange(0.0, 31.0, 3.0)
    counts, _ = np.histogram(lengths_s, bins=edges_s)
    policy_name = os.path.basename(policy_path)
    model = re.search(r"model_\d+", policy_name)
    model_label = model.group() if model else policy_name

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(edges_s[:-1], counts, width=3.0, align="edge", color="#1f77b4", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_xticks(edges_s)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(1, int(counts.max()) * 1.15))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Episode length (s)")
    ax.set_ylabel("Number of completed episodes")
    ax.set_title(f"Episode length distribution — {len(ep_records)} completed episodes\n"
                 f"{model_label} | Domain randomization: {'ON' if enable_dr else 'OFF'}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.02, "Bins: [0, 3), ..., [27, 30] s. Timeouts included; unfinished episodes excluded.",
             ha="center", fontsize=9)
    fig.savefig(out_base + "_epi_len_histogram.png", dpi=150)
    plt.close(fig)
    np.savetxt(
        out_base + "_epi_len_histogram.csv",
        np.column_stack((edges_s[:-1], edges_s[1:], counts)), delimiter=",",
        header="start_s,end_s,episode_count", comments="", fmt=("%.1f", "%.1f", "%d"),
    )
    print(f"[INFO] episode length distribution written: {out_base}_epi_len_histogram.png / .csv")


def _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base):
    """play_eval-style torque profile: applied drive/steer torque vs step with
    +-forcerange dashed lines. The single-env MuJoCo analog of play_eval's
    6-panel plot (MuJoCo's ``actuator_force`` IS the post-forcerange-clip
    applied torque, matching the sim's ``applied_torque``). Episode boundaries
    are dotted vertical lines. Saved as <out_base>_torque.png (+ raw .npz).
    """
    boundaries = np.cumsum([len(e) for e in tau_eps])[:-1]
    steps = np.arange(sum(len(e) for e in tau_eps), dtype=float)
    drive = np.concatenate([e[:, 0] for e in tau_eps])
    steer = np.concatenate([e[:, 1] for e in tau_eps])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, tau, lim, name in ((axes[0], drive, drive_lim, "drive"), (axes[1], steer, steer_lim, "steer")):
        ax.plot(steps, tau, color="#1f77b4", lw=0.7)
        ax.axhline(lim, color="k", ls="--", lw=0.8)
        ax.axhline(-lim, color="k", ls="--", lw=0.8)
        for b in boundaries:
            ax.axvline(b, color="gray", ls=":", lw=0.6)
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.set_ylabel(f"{name} τ (N·m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(f"applied motor torque vs step — MuJoCo deploy_sim ({len(tau_eps)} episode(s))")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_base + "_torque.png", dpi=150)
    plt.close(fig)
    np.savez(out_base + "_torque.npz", steps=steps, boundaries=boundaries, drive=drive, steer=steer)
    print(f"[INFO] torque profile written to: {out_base}_torque.png / .npz")


def _plot_multi_torque_traces(traces, resets, drive_lim, steer_lim, out_base, seed):
    """All environments on a shared control-step axis; preserve IDs in the NPZ."""
    torque = np.asarray(traces)
    steps = np.arange(len(torque))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, values, limit, name in (
        (axes[0], torque[:, :, 0], drive_lim, "drive"),
        (axes[1], torque[:, :, 1], steer_lim, "steer"),
    ):
        ax.plot(steps, values, color="#1f77b4", alpha=0.15, lw=0.5)
        ax.plot(steps, values.mean(axis=1), color="black", lw=0.9, label="mean across environments")
        ax.axhline(limit, color="gray", ls="--", lw=0.8)
        ax.axhline(-limit, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-1.15 * limit, 1.15 * limit)
        ax.set_ylabel(f"{name} torque (N m)")
        ax.legend(fontsize=8)
    axes[0].set_title(f"Applied motor torque — {torque.shape[1]} independent environments")
    axes[-1].set_xlabel("global control step (0.02 s each)")
    fig.tight_layout()
    fig.savefig(out_base + "_torque.png", dpi=150)
    plt.close(fig)
    np.savez(
        out_base + "_torque.npz", steps=steps,
        drive=torque[:, :, 0], steer=torque[:, :, 1],
        env_ids=np.arange(torque.shape[1]), seeds=seed + np.arange(torque.shape[1]),
        reset_events=np.asarray(resets, dtype=int).reshape(-1, 2),
        control_dt=BikeBalanceEnv.CTRL_DT,
    )
    print(f"[INFO] all-environment torque profile written: {out_base}_torque.png / .npz")



