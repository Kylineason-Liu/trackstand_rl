#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/eval_command_sweeps.py (imports/paths rewritten for trackstand_rl locomotive).
"""Headless MuJoCo command + domain-randomization eval for Paper policies.

Command suites
--------------
speed    — straight heading, sweep vx*
heading  — fixed vx*, sweep constant yaw-rate command (rad/s); like Isaac, the
           observed command is clipped to the training range ±0.5 rad/s

For the CycleRL paper metrics (BSR / STE / VTE / CAT / MNT / MSS) on the same
conditions as the Isaac suite, use ``eval_paper_suite.py`` instead.
combo    — grid of (vx*, yaw-rate)

Domain suites (Paper Table I axes; default vx*=2.0, yaw*=0 unless noted)
-----------------------------------------------------------------------
friction — slide μ ∈ {0.5, 0.7, 1.0, 1.2} on floor + tires
mass     — body mass/inertia scale ∈ {0.6, 1.0, 1.4, 1.8}
com      — base COM lateral/vertical offsets
gains    — PD gain scale ∈ {0.9, 1.0, 1.1}
imu      — Gaussian obs noise on 6-D IMU block
roll     — initial roll (deg)
push     — lateral velocity impulse after settle
dr       — all domain suites above

Usage::

    python3 mujoco_sim2sim/scripts/locomotive/eval_command_sweeps.py --suite=dr
    python3 mujoco_sim2sim/scripts/locomotive/eval_command_sweeps.py --suite=all --compare \\
        mujoco_sim2sim/logs/locomotive/19-00-58_best_agent.npz \\
        mujoco_sim2sim/logs/locomotive/act_refine_lo_agent_100000.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parents[2]  # trackstand_rl root
sys.path.insert(0, str(_SCRIPT_DIR))

from play_act_refine_lo import (  # noqa: E402
    _DEFAULT_NPZ,
    _DEFAULT_XML,
    _HEADING_STIFFNESS,
    _POLICY_DT,
    PaperBikeSim,
)
from policy_numpy import PaperPolicy  # noqa: E402

_LEAN_LIMIT_DEG = 45.0
_COMMAND_SUITES = ("speed", "heading", "combo")
_DOMAIN_SUITES = ("friction", "mass", "com", "gains", "imu", "roll", "push")
_ALL_SUITES = _COMMAND_SUITES + _DOMAIN_SUITES


@dataclass
class DomainCfg:
    friction: float | None = None
    mass_scale: float = 1.0
    com_y: float = 0.0
    com_z: float = 0.0
    gain_scale: float = 1.0
    imu_noise_std: float = 0.0
    roll_init_deg: float = 0.0
    push_vy: float = 0.0
    label: str = "nominal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    suite: str
    vx_cmd: float
    yaw_rate_cmd: float
    duration_s: float
    settle_s: float
    survived: bool
    t_crash_s: float | None
    max_abs_roll_deg: float
    mean_abs_roll_deg: float
    mean_vx: float
    mean_vx_err: float
    net_heading_deg: float
    path_length_m: float
    n_steps: int
    domain: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepReport:
    created: str
    npz: str
    xml: str
    suites: list[str]
    episodes: list[EpisodeResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _wrap_pi(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def run_episode(
    sim: PaperBikeSim,
    *,
    suite: str,
    vx_cmd: float,
    yaw_rate_cmd: float,
    duration_s: float,
    settle_s: float,
    domain: DomainCfg | None = None,
    lean_limit_deg: float = _LEAN_LIMIT_DEG,
) -> EpisodeResult:
    """Settle straight at ``vx_cmd``, then apply constant yaw-rate command."""
    domain = domain or DomainCfg()
    sim.apply_domain(
        friction=domain.friction,
        mass_scale=domain.mass_scale,
        com_y=domain.com_y,
        com_z=domain.com_z,
        gain_scale=domain.gain_scale,
        imu_noise_std=domain.imu_noise_std,
    )
    sim.vx_cmd = float(vx_cmd)
    sim.reset(roll_rad=math.radians(domain.roll_init_deg))
    heading0 = sim._heading()
    sim.heading_target = heading0

    n_settle = max(0, int(round(settle_s / _POLICY_DT)))
    n_cmd = max(1, int(round(duration_s / _POLICY_DT)))

    rolls: list[float] = []
    vxs: list[float] = []
    path = 0.0
    prev_xy = sim.data.qpos[sim.qadr_free : sim.qadr_free + 2].copy()
    survived = True
    t_crash: float | None = None
    step_i = 0
    push_applied = False

    def _one_step(apply_yaw: bool) -> dict:
        nonlocal path, prev_xy
        if apply_yaw and abs(yaw_rate_cmd) > 1e-9 and _HEADING_STIFFNESS > 1e-9:
            sim.heading_target = sim._heading() + float(yaw_rate_cmd) / _HEADING_STIFFNESS
        else:
            sim.heading_target = heading0
        info = sim.policy_step()
        xy = sim.data.qpos[sim.qadr_free : sim.qadr_free + 2]
        path += float(np.linalg.norm(xy - prev_xy))
        prev_xy = xy.copy()
        return info

    for _ in range(n_settle):
        info = _one_step(apply_yaw=False)
        step_i += 1
        rolls.append(abs(info["roll_deg"]))
        vxs.append(info["vx"])
        if abs(info["roll_deg"]) > lean_limit_deg:
            survived = False
            t_crash = step_i * _POLICY_DT
            break

    if survived and abs(domain.push_vy) > 1e-9 and not push_applied:
        sim.apply_lateral_push(domain.push_vy)
        push_applied = True

    if survived:
        for _ in range(n_cmd):
            info = _one_step(apply_yaw=True)
            step_i += 1
            rolls.append(abs(info["roll_deg"]))
            vxs.append(info["vx"])
            if abs(info["roll_deg"]) > lean_limit_deg:
                survived = False
                t_crash = step_i * _POLICY_DT
                break

    sim.restore_nominal_plant()

    rolls_a = np.asarray(rolls, dtype=float) if rolls else np.array([0.0])
    vxs_a = np.asarray(vxs, dtype=float) if vxs else np.array([vx_cmd])
    net_hdg = math.degrees(_wrap_pi(sim._heading() - heading0))

    return EpisodeResult(
        suite=suite,
        vx_cmd=float(vx_cmd),
        yaw_rate_cmd=float(yaw_rate_cmd),
        duration_s=float(duration_s),
        settle_s=float(settle_s),
        survived=survived,
        t_crash_s=t_crash,
        max_abs_roll_deg=float(rolls_a.max()),
        mean_abs_roll_deg=float(rolls_a.mean()),
        mean_vx=float(vxs_a.mean()),
        mean_vx_err=float(np.mean(np.abs(vxs_a - vx_cmd))),
        net_heading_deg=float(net_hdg),
        path_length_m=float(path),
        n_steps=step_i,
        domain=domain.to_dict(),
    )


def _domain_jobs(suite: str, vx: float) -> list[tuple[str, float, float, DomainCfg]]:
    """Build (suite, vx, yaw, domain) jobs for one domain axis."""
    jobs: list[tuple[str, float, float, DomainCfg]] = []
    if suite == "friction":
        for mu in (0.5, 0.7, 1.0, 1.2):
            jobs.append((suite, vx, 0.0, DomainCfg(friction=mu, label=f"mu={mu:.1f}")))
    elif suite == "mass":
        for s in (0.6, 1.0, 1.4, 1.8):
            jobs.append((suite, vx, 0.0, DomainCfg(mass_scale=s, label=f"mass×{s:.1f}")))
    elif suite == "com":
        for cy, cz, lab in (
            (0.0, 0.0, "com=0"),
            (0.01, 0.0, "com_y=+1cm"),
            (-0.01, 0.0, "com_y=-1cm"),
            (0.0, 0.015, "com_z=+1.5cm"),
            (0.0, -0.015, "com_z=-1.5cm"),
            (0.01, 0.015, "com_y+z"),
        ):
            jobs.append((suite, vx, 0.0, DomainCfg(com_y=cy, com_z=cz, label=lab)))
    elif suite == "gains":
        for g in (0.9, 1.0, 1.1):
            jobs.append((suite, vx, 0.0, DomainCfg(gain_scale=g, label=f"gain×{g:.1f}")))
    elif suite == "imu":
        for std in (0.0, 0.005, 0.01, 0.02, 0.05):
            jobs.append((suite, vx, 0.0, DomainCfg(imu_noise_std=std, label=f"imuσ={std:.3f}")))
    elif suite == "roll":
        for r in (0.0, 2.0, 5.0, 8.0, 12.0, -5.0, -8.0):
            jobs.append((suite, vx, 0.0, DomainCfg(roll_init_deg=r, label=f"roll0={r:+.0f}°")))
    elif suite == "push":
        for vy in (0.0, 0.15, 0.3, 0.45, 0.6, -0.3, -0.45):
            jobs.append((suite, vx, 0.0, DomainCfg(push_vy=vy, label=f"push_vy={vy:+.2f}")))
    return jobs


def _summarize(episodes: list[EpisodeResult]) -> dict:
    by_suite: dict[str, list[EpisodeResult]] = {}
    for ep in episodes:
        by_suite.setdefault(ep.suite, []).append(ep)

    out: dict = {}
    for suite, eps in by_suite.items():
        n = len(eps)
        n_ok = sum(1 for e in eps if e.survived)
        fails = [e for e in eps if not e.survived]
        out[suite] = {
            "n": n,
            "n_survived": n_ok,
            "survival_rate": n_ok / n if n else 0.0,
            "worst_max_roll_deg": max(e.max_abs_roll_deg for e in eps) if eps else 0.0,
            "fail_cases": [
                {
                    "vx_cmd": e.vx_cmd,
                    "yaw_rate_cmd": e.yaw_rate_cmd,
                    "domain": e.domain.get("label", ""),
                    "t_crash_s": e.t_crash_s,
                    "max_abs_roll_deg": e.max_abs_roll_deg,
                    "net_heading_deg": e.net_heading_deg,
                }
                for e in sorted(fails, key=lambda e: (e.t_crash_s or 0.0))
            ],
        }
        if suite == "speed":
            out[suite]["table"] = [
                {
                    "vx_cmd": e.vx_cmd,
                    "survived": e.survived,
                    "max_abs_roll_deg": e.max_abs_roll_deg,
                    "mean_vx_err": e.mean_vx_err,
                }
                for e in sorted(eps, key=lambda e: e.vx_cmd)
            ]
        elif suite == "heading":
            out[suite]["table"] = [
                {
                    "yaw_rate_cmd": e.yaw_rate_cmd,
                    "vx_cmd": e.vx_cmd,
                    "survived": e.survived,
                    "max_abs_roll_deg": e.max_abs_roll_deg,
                    "net_heading_deg": e.net_heading_deg,
                    "t_crash_s": e.t_crash_s,
                }
                for e in sorted(eps, key=lambda e: e.yaw_rate_cmd)
            ]
        elif suite == "combo":
            cells = {}
            for e in eps:
                cells[f"{e.vx_cmd:.2f}|{e.yaw_rate_cmd:.2f}"] = {
                    "vx_cmd": e.vx_cmd,
                    "yaw_rate_cmd": e.yaw_rate_cmd,
                    "survived": e.survived,
                    "max_abs_roll_deg": e.max_abs_roll_deg,
                    "t_crash_s": e.t_crash_s,
                    "net_heading_deg": e.net_heading_deg,
                }
            out[suite]["grid"] = cells
        elif suite in _DOMAIN_SUITES:
            out[suite]["table"] = [
                {
                    "label": e.domain.get("label", ""),
                    "vx_cmd": e.vx_cmd,
                    "survived": e.survived,
                    "max_abs_roll_deg": e.max_abs_roll_deg,
                    "mean_vx_err": e.mean_vx_err,
                    "t_crash_s": e.t_crash_s,
                    "domain": e.domain,
                }
                for e in eps
            ]
    return out


def _write_markdown(report: SweepReport, path: Path, title: str | None = None) -> None:
    lines = [
        f"# {title or 'MuJoCo Paper policy command + DR sweeps'}",
        "",
        f"- created: `{report.created}`",
        f"- policy: `{report.npz}`",
        f"- xml: `{report.xml}`",
        f"- suites: {', '.join(report.suites)}",
        "",
    ]
    for suite, s in report.summary.items():
        lines += [
            f"## {suite}",
            "",
            f"- episodes: **{s['n_survived']}/{s['n']}** survived "
            f"({100.0 * s['survival_rate']:.0f}%)",
            f"- worst max |roll|: **{s['worst_max_roll_deg']:.1f}°**",
            "",
        ]
        if suite == "speed" and "table" in s:
            lines.append("| vx* | survived | max|roll| | mean |vx err| |")
            lines.append("|-----|----------|-----------|--------------|")
            for r in s["table"]:
                lines.append(
                    f"| {r['vx_cmd']:.2f} | {'Y' if r['survived'] else 'N'} | "
                    f"{r['max_abs_roll_deg']:.1f} | {r['mean_vx_err']:.2f} |"
                )
            lines.append("")
        elif suite == "heading" and "table" in s:
            lines.append("| yaw_rate* | vx* | survived | max|roll| | Δheading | t_crash |")
            lines.append("|-----------|-----|----------|-----------|----------|---------|")
            for r in s["table"]:
                tc = f"{r['t_crash_s']:.2f}" if r["t_crash_s"] is not None else "-"
                lines.append(
                    f"| {r['yaw_rate_cmd']:+.2f} | {r['vx_cmd']:.2f} | "
                    f"{'Y' if r['survived'] else 'N'} | {r['max_abs_roll_deg']:.1f} | "
                    f"{r['net_heading_deg']:+.1f} | {tc} |"
                )
            lines.append("")
        elif suite == "combo" and "grid" in s:
            vxs = sorted({c["vx_cmd"] for c in s["grid"].values()})
            yaws = sorted({c["yaw_rate_cmd"] for c in s["grid"].values()})
            lines.append("Survival grid (Y/N), rows=vx*, cols=yaw_rate*:")
            lines.append("")
            header = "| vx* \\ yaw | " + " | ".join(f"{y:+.2f}" for y in yaws) + " |"
            sep = "|---|" + "|".join(["---"] * len(yaws)) + "|"
            lines += [header, sep]
            for vx in vxs:
                cells = []
                for yaw in yaws:
                    key = f"{vx:.2f}|{yaw:.2f}"
                    cell = s["grid"].get(key)
                    if cell is None:
                        cells.append("?")
                    elif cell["survived"]:
                        cells.append(f"Y({cell['max_abs_roll_deg']:.0f}°)")
                    else:
                        tc = cell["t_crash_s"] if cell["t_crash_s"] is not None else float("nan")
                        cells.append(f"N@{tc:.1f}s")
                lines.append(f"| {vx:.2f} | " + " | ".join(cells) + " |")
            lines.append("")
        elif suite in _DOMAIN_SUITES and "table" in s:
            lines.append("| case | vx* | survived | max|roll| | |vx err| | t_crash |")
            lines.append("|------|-----|----------|-----------|---------|---------|")
            for r in s["table"]:
                tc = f"{r['t_crash_s']:.2f}" if r["t_crash_s"] is not None else "-"
                lines.append(
                    f"| {r['label']} | {r['vx_cmd']:.2f} | "
                    f"{'Y' if r['survived'] else 'N'} | {r['max_abs_roll_deg']:.1f} | "
                    f"{r['mean_vx_err']:.2f} | {tc} |"
                )
            lines.append("")
        if s.get("fail_cases"):
            lines.append("### Failures")
            lines.append("")
            for f in s["fail_cases"][:30]:
                dom = f" [{f['domain']}]" if f.get("domain") else ""
                lines.append(
                    f"- vx*={f['vx_cmd']:.2f}, yaw*={f['yaw_rate_cmd']:+.2f}{dom} → "
                    f"crash@{f['t_crash_s']:.2f}s, max|roll|={f['max_abs_roll_deg']:.1f}°, "
                    f"Δhdg={f['net_heading_deg']:+.1f}°"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_compare_markdown(
    path: Path,
    name_a: str,
    report_a: SweepReport,
    name_b: str,
    report_b: SweepReport,
) -> None:
    lines = [
        f"# Compare: {name_a} vs {name_b}",
        "",
        f"- A: `{report_a.npz}`",
        f"- B: `{report_b.npz}`",
        f"- created: `{report_a.created}`",
        "",
        "| suite | A survived | B survived | A worst |roll| | B worst |roll| | winner |",
        "|-------|------------|------------|----------------|----------------|--------|",
    ]
    suites = list(dict.fromkeys([*report_a.summary.keys(), *report_b.summary.keys()]))
    score = {name_a: 0, name_b: 0, "tie": 0}
    for suite in suites:
        sa = report_a.summary.get(suite, {})
        sb = report_b.summary.get(suite, {})
        ra = sa.get("survival_rate", 0.0)
        rb = sb.get("survival_rate")
        if rb is None:
            continue
        na, nb = sa.get("n_survived", 0), sb.get("n_survived", 0)
        ta, tb = sa.get("n", 0), sb.get("n", 0)
        wa, wb = sa.get("worst_max_roll_deg", 0.0), sb.get("worst_max_roll_deg", 0.0)
        if ra > rb + 1e-9:
            winner = name_a
            score[name_a] += 1
        elif rb > ra + 1e-9:
            winner = name_b
            score[name_b] += 1
        elif wa < wb - 1e-9:
            winner = f"{name_a} (lean)"
            score[name_a] += 1
        elif wb < wa - 1e-9:
            winner = f"{name_b} (lean)"
            score[name_b] += 1
        else:
            winner = "tie"
            score["tie"] += 1
        lines.append(
            f"| {suite} | {na}/{ta} ({100 * ra:.0f}%) | {nb}/{tb} ({100 * rb:.0f}%) | "
            f"{wa:.1f}° | {wb:.1f}° | **{winner}** |"
        )
    lines += [
        "",
        f"**Suite wins:** {name_a}={score[name_a]}, {name_b}={score[name_b]}, ties={score['tie']}",
        "",
    ]

    # Domain detail side-by-side where both have tables
    for suite in _DOMAIN_SUITES:
        if suite not in report_a.summary or suite not in report_b.summary:
            continue
        ta = {r["label"]: r for r in report_a.summary[suite].get("table", [])}
        tb = {r["label"]: r for r in report_b.summary[suite].get("table", [])}
        labels = list(dict.fromkeys([*ta.keys(), *tb.keys()]))
        if not labels:
            continue
        lines += [
            f"## {suite} detail",
            "",
            f"| case | {name_a} | {name_b} |",
            "|------|---------|---------|",
        ]
        for lab in labels:
            a = ta.get(lab)
            b = tb.get(lab)
            def _cell(r: dict | None) -> str:
                if r is None:
                    return "-"
                if r["survived"]:
                    return f"Y ({r['max_abs_roll_deg']:.0f}°)"
                tc = r["t_crash_s"] if r["t_crash_s"] is not None else float("nan")
                return f"N@{tc:.1f}s"
            lines.append(f"| {lab} | {_cell(a)} | {_cell(b)} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_suites(arg: str) -> list[str]:
    if arg == "all":
        return list(_ALL_SUITES)
    if arg == "dr":
        return list(_DOMAIN_SUITES)
    if arg == "commands":
        return list(_COMMAND_SUITES)
    return [arg]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo Paper policy command + DR eval.")
    p.add_argument("--npz", type=Path, default=_DEFAULT_NPZ)
    p.add_argument("--xml", type=Path, default=_DEFAULT_XML)
    p.add_argument(
        "--suite",
        default="all",
        choices=("all", "commands", "dr", *_ALL_SUITES),
        help="Sweep set. 'dr' = domain suites only; 'all' = commands + domain.",
    )
    p.add_argument("--duration", type=float, default=6.0, help="Command phase duration (s).")
    p.add_argument("--settle", type=float, default=1.0, help="Straight settle before command (s).")
    p.add_argument("--speeds", type=str, default="1.0,1.4,1.8,2.0,2.2,2.5,2.8,3.0")
    p.add_argument(
        "--yaw-rates",
        type=str,
        default="-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8",
        help="Use --yaw-rates=-0.6,0,0.6 if values start with '-'.",
    )
    p.add_argument("--heading-vx", type=float, default=2.0)
    p.add_argument("--combo-speeds", type=str, default="1.5,2.0,2.5,3.0")
    p.add_argument(
        "--combo-yaw-rates",
        type=str,
        default="-0.6,-0.3,0.0,0.3,0.6",
    )
    p.add_argument(
        "--dr-vx",
        type=float,
        default=2.0,
        help="Fixed vx* for domain-randomization suites.",
    )
    p.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        default=None,
        help="One or more extra npz policies to eval + compare against --npz.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for IMU noise.")
    return p.parse_args()


def _run_jobs(
    *,
    npz: Path,
    xml: Path,
    suites: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> SweepReport:
    np.random.seed(args.seed)
    speeds = _parse_float_list(args.speeds)
    yaws = _parse_float_list(args.yaw_rates)
    combo_speeds = _parse_float_list(args.combo_speeds)
    combo_yaws = _parse_float_list(args.combo_yaw_rates)

    policy = PaperPolicy(npz)
    sim = PaperBikeSim(xml, policy, vx_cmd=float(args.dr_vx))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report = SweepReport(created=stamp, npz=str(npz), xml=str(xml), suites=suites)

    jobs: list[tuple[str, float, float, DomainCfg]] = []
    if "speed" in suites:
        for vx in speeds:
            jobs.append(("speed", vx, 0.0, DomainCfg(label="nominal")))
    if "heading" in suites:
        for yaw in yaws:
            jobs.append(("heading", float(args.heading_vx), yaw, DomainCfg(label="nominal")))
    if "combo" in suites:
        for vx in combo_speeds:
            for yaw in combo_yaws:
                jobs.append(("combo", vx, yaw, DomainCfg(label="nominal")))
    for suite in _DOMAIN_SUITES:
        if suite in suites:
            jobs.extend(_domain_jobs(suite, float(args.dr_vx)))

    print(f"[INFO] {npz.name}: {len(jobs)} episodes | suites={suites}")
    t0 = time.time()
    for i, (suite, vx, yaw, domain) in enumerate(jobs, 1):
        ep = run_episode(
            sim,
            suite=suite,
            vx_cmd=vx,
            yaw_rate_cmd=yaw,
            duration_s=args.duration,
            settle_s=args.settle,
            domain=domain,
        )
        report.episodes.append(ep)
        flag = "PASS" if ep.survived else f"FAIL@{ep.t_crash_s:.2f}s"
        tag = domain.label if domain.label != "nominal" else ""
        print(
            f"[{i:3d}/{len(jobs)}] {suite:8s} vx*={vx:.2f} yaw*={yaw:+.2f} "
            f"{tag:14s} {flag}  max|roll|={ep.max_abs_roll_deg:5.1f}°"
        )

    report.summary = _summarize(report.episodes)
    stem = npz.stem
    json_path = out_dir / f"{stem}_sweep_results.json"
    md_path = out_dir / f"{stem}_sweep_report.md"
    json_path.write_text(
        json.dumps(
            {
                "created": report.created,
                "npz": report.npz,
                "xml": report.xml,
                "suites": report.suites,
                "summary": report.summary,
                "episodes": [asdict(e) for e in report.episodes],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, md_path, title=f"MuJoCo sweeps — {stem}")
    print(f"[DONE] {npz.name} in {time.time() - t0:.1f}s → {md_path}")
    for suite, s in report.summary.items():
        print(
            f"  {suite}: {s['n_survived']}/{s['n']} "
            f"({100 * s['survival_rate']:.0f}%), worst|roll|={s['worst_max_roll_deg']:.1f}°"
        )
    return report


def main() -> None:
    args = _parse_args()
    if not args.npz.is_file():
        raise SystemExit(f"Missing policy npz: {args.npz}")
    if not args.xml.is_file():
        raise SystemExit(f"Missing MJCF: {args.xml}")

    suites = _resolve_suites(args.suite)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.out_dir or (_REPO / "mujoco_sim2sim/logs/locomotive/evals" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = [args.npz]
    if args.compare:
        for p in args.compare:
            if not p.is_file():
                raise SystemExit(f"Missing compare npz: {p}")
            policies.append(p)

    reports: list[tuple[str, SweepReport]] = []
    for npz in policies:
        rep = _run_jobs(npz=npz, xml=args.xml, suites=suites, args=args, out_dir=out_dir)
        reports.append((npz.stem, rep))

    if len(reports) >= 2:
        # Pairwise compare first vs each other, and a full multi summary vs first.
        name_a, rep_a = reports[0]
        for name_b, rep_b in reports[1:]:
            cmp_path = out_dir / f"compare_{name_a}_vs_{name_b}.md"
            _write_compare_markdown(cmp_path, name_a, rep_a, name_b, rep_b)
            print(f"[COMPARE] {cmp_path}")


if __name__ == "__main__":
    main()
