#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/eval/eval_all_metrics.py (imports/paths rewritten for trackstand_rl locomotive).
"""Cycle a checkpoint through all ``play_with_metrics.py`` eval modes.

Run this with **system Python** (not ``isaaclab.sh -p``). It spawns a fresh Kit
process per mode by default so one OOM/crash does not abort the suite.

Examples::

    # Post-train plane suite (default):
    python isaaclab_train/scripts/eval_locomotive/eval_all_metrics.py \\
        --task CycleRL-NewBike-Paper-SteeringOnly-v0 \\
        --checkpoint /path/to/best_agent.pt \\
        --num_envs 64 --headless

    # Explicit end suite:
    python .../eval_all_metrics.py --suite end --checkpoint ... --headless

    # Include staged terrain:
    python .../eval_all_metrics.py --suite full --checkpoint ... --headless

    # Explicit subset:
    python .../eval_all_metrics.py --modes nominal,yaw_bias_sweep,kick \\
        --checkpoint ... --headless

    # Single Kit process (faster; one crash kills the suite):
    python .../eval_all_metrics.py --one_shot --suite end --checkpoint ... --headless

The first suite lands under ``<run>/eval/<mode>/`` plus ``eval_all_metrics_status.json``.
A re-run that would replace an existing ``summary.json`` is written to
``<run>/eval_<UTC>/`` instead. ``--overwrite`` keeps the old in-place behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PLAY_METRICS = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "skrl", "play_locomotive_metrics.py"))

import eval_paths  # noqa: E402

# Keep in sync with ``eval_scenarios.EVAL_SUITE_ALIASES`` / ``EVAL_MODES``.
_LADDER_MODES = (
    "speed_sweep",
    "heading_sweep",
    "yaw_bias_sweep",
    "drive_accel_sweep",
    "kick_sweep",
    "mass_sweep",
    "com_sweep",
    "mass_com_sweep",
    "friction_sweep",
)
_PAPER_TABLE_VI = (
    "paper_low_speed",
    "paper_normal_speed",
    "paper_no_noise",
    "paper_max_noise",
    "paper_dropout",
    "paper_flat",
    "paper_rough",
    "paper_gravel",
    "paper_slope",
    "paper_steps",
)
_PAPER_FRONTIER = ("paper_roll_sweep", "paper_noise_sweep", "speed_sweep")
SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "paper": _PAPER_TABLE_VI,
    "paper_frontier": _PAPER_FRONTIER,
    "paper_full": _PAPER_TABLE_VI + _PAPER_FRONTIER,
    # Headline suite: paper Table VI + CAT/MNT/MSS ladders.
    "end": _PAPER_TABLE_VI + _PAPER_FRONTIER,
    "diagnostics": _LADDER_MODES,
    "ladder": _LADDER_MODES,
    "envelope": (
        "mass_sweep",
        "com_sweep",
        "mass_com_sweep",
        "friction_sweep",
        "kick_sweep",
    ),
    "complex": tuple(f"complex_L{i}" for i in range(5)),
    "stairs": tuple(f"stairs_L{i}" for i in range(3)),
}
SUITE_ALIASES["terrain"] = SUITE_ALIASES["complex"] + SUITE_ALIASES["stairs"]
SUITE_ALIASES["full"] = SUITE_ALIASES["end"] + _LADDER_MODES

DEFAULT_SUITE = "end"
DEFAULT_TASK = "Locomotive-Cylinder-Aligned-v0"


def _inside_kit() -> bool:
    try:
        import carb  # noqa: F401

        return True
    except ImportError:
        return False


def _find_isaaclab_sh() -> str | None:
    candidates: list[str] = []
    for key in ("ISAACLAB_PATH", "ISAACLAB_ROOT", "ISAAC_LAB_PATH"):
        root = os.environ.get(key)
        if root:
            candidates.append(os.path.join(root, "isaaclab.sh"))
    candidates.append(os.path.join(os.getcwd(), "isaaclab.sh"))
    home = os.path.expanduser("~")
    candidates.extend(
        [
            os.path.join(home, "IsaacLab", "isaaclab.sh"),
            "/home/lmbike2/IsaacLab/isaaclab.sh",
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def expand_suite(spec: str) -> list[str]:
    """Expand ``full`` / ``end`` / comma lists into concrete mode names."""
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    out: list[str] = []
    seen: set[str] = set()
    # Ladder+terrain suites plus optional fixed probes still selectable via --modes.
    known = set(SUITE_ALIASES["full"]) | {
        "nominal",
        "kick",
        "mass",
        "friction",
        "training",
    }
    for tok in tokens:
        modes = list(SUITE_ALIASES[tok]) if tok in SUITE_ALIASES else [tok]
        for m in modes:
            if m not in known and m not in SUITE_ALIASES:
                raise SystemExit(
                    f"Unknown eval mode/suite {m!r}. "
                    f"Choose from {sorted(known)} or aliases {sorted(SUITE_ALIASES)}"
                )
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
    return out


def _kit_prefix() -> list[str]:
    isaac = _find_isaaclab_sh()
    if isaac:
        return [isaac, "-p"]
    print(
        "[WARN] isaaclab.sh not found; using current Python. "
        "Set ISAACLAB_PATH if Kit fails to start.",
        file=sys.stderr,
    )
    return [sys.executable]


def _play_cmd(mode_args: Sequence[str], passthrough: Sequence[str]) -> list[str]:
    return [*_kit_prefix(), _PLAY_METRICS, *mode_args, *passthrough]


def _default_save_dir(checkpoint: str) -> str:
    path = os.path.abspath(checkpoint)
    if os.path.isdir(path):
        run_dir = path
    else:
        parent = os.path.dirname(path)
        run_dir = os.path.dirname(parent) if os.path.basename(parent) == "checkpoints" else parent
    return os.path.join(run_dir, "eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cycle a checkpoint through all play_with_metrics eval modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Suites: end (default, plane post-train), full (end+terrain), "
            "envelope, terrain, complex, stairs. Run with system Python, not isaaclab.sh -p."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint, or a run directory containing checkpoints/.",
    )
    p.add_argument("--task", type=str, default=DEFAULT_TASK)
    p.add_argument(
        "--suite",
        type=str,
        default=DEFAULT_SUITE,
        help=(
            f"Suite alias or comma list (default: {DEFAULT_SUITE}). "
            f"Aliases: {', '.join(sorted(SUITE_ALIASES))}."
        ),
    )
    p.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Override --suite with an explicit comma-separated mode list.",
    )
    p.add_argument(
        "--one_shot",
        action="store_true",
        default=False,
        help="Run the whole suite in one Kit process via play_with_metrics --eval_suite.",
    )
    p.add_argument(
        "--fail_fast",
        action="store_true",
        default=False,
        help="Stop after the first non-zero exit (per-mode mode only).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="Print expanded modes for --suite/--modes and exit.",
    )
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--save_dir", type=str, default=None, help="Eval output root (default: <run>/eval).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Replace summaries already in --save_dir. Default: write a dated sibling directory.",
    )
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to play_with_metrics (place after --).",
    )
    return p


def _common_play_args(args: argparse.Namespace) -> tuple[list[str], str]:
    out: list[str] = [
        "--task",
        args.task,
        "--checkpoint",
        os.path.abspath(args.checkpoint),
        "--num_envs",
        str(int(args.num_envs)),
    ]
    requested = Path(args.save_dir or _default_save_dir(args.checkpoint))
    save_path, redirected = eval_paths.allocate_eval_dir(requested, overwrite=bool(args.overwrite))
    eval_paths.note_eval_allocation(
        save_path,
        requested=requested,
        redirected=redirected,
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else None,
    )
    if redirected:
        print(
            f"[eval_all_metrics] {requested} already has summaries; "
            f"this re-run writes to {save_path}"
        )
    save_dir = str(save_path)
    out.extend(["--save_dir", save_dir])
    if args.overwrite:
        out.append("--overwrite")
    if args.eval_steps is not None:
        out.extend(["--eval_steps", str(int(args.eval_steps))])
    if args.headless:
        out.append("--headless")
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    out.extend(extra)
    return out, os.path.abspath(save_dir)


def run_one_shot(modes: list[str], common: Sequence[str]) -> int:
    suite = ",".join(modes)
    cmd = _play_cmd(["--eval_suite", suite], common)
    print("[eval_all_metrics] one-shot command:")
    print("  " + " ".join(cmd))
    return int(subprocess.run(cmd, check=False).returncode)


def run_per_mode(
    modes: list[str],
    common: Sequence[str],
    *,
    save_dir: str,
    fail_fast: bool,
) -> int:
    os.makedirs(save_dir, exist_ok=True)
    results: list[dict] = []
    t0 = time.time()

    print(f"[eval_all_metrics] cycling {len(modes)} modes → {save_dir}")
    for i, mode in enumerate(modes, start=1):
        print("\n" + "=" * 72)
        print(f"[eval_all_metrics] ({i}/{len(modes)}) mode={mode}")
        print("=" * 72)
        cmd = _play_cmd(["--eval_mode", mode], common)
        print("  " + " ".join(cmd))
        t_mode = time.time()
        rc = int(subprocess.run(cmd, check=False).returncode)
        elapsed = time.time() - t_mode
        summary_path = os.path.join(save_dir, mode, "summary.json")
        wrote_summary = os.path.isfile(summary_path)
        # Kit sometimes exits 0 after CUDA/init failure without writing artifacts.
        effective_rc = rc if rc != 0 else (0 if wrote_summary else 2)
        row = {
            "mode": mode,
            "returncode": effective_rc,
            "kit_returncode": rc,
            "wrote_summary": wrote_summary,
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        if effective_rc == 0:
            status = "OK"
        elif rc == 0 and not wrote_summary:
            status = "FAIL(no summary.json)"
        else:
            status = f"FAIL(rc={rc})"
        print(f"[eval_all_metrics] {mode}: {status} in {elapsed:.1f}s")
        if effective_rc != 0 and fail_fast:
            print("[eval_all_metrics] --fail_fast: stopping.")
            break

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": next(
            (common[i + 1] for i, a in enumerate(common) if a == "--checkpoint"),
            None,
        ),
        "task": next((common[i + 1] for i, a in enumerate(common) if a == "--task"), None),
        "modes": modes,
        "results": results,
        "n_ok": sum(1 for r in results if r["returncode"] == 0),
        "n_fail": sum(1 for r in results if r["returncode"] != 0),
        "elapsed_s": round(time.time() - t0, 1),
    }
    status_path = os.path.join(save_dir, "eval_all_metrics_status.json")
    with open(status_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== EVAL ALL METRICS ==========")
    for r in results:
        mark = "OK  " if r["returncode"] == 0 else "FAIL"
        print(f"  {r['mode']:<18s}  {mark}  rc={r['returncode']}  {r['elapsed_s']:.1f}s")
    print(
        f"\n[eval_all_metrics] {summary['n_ok']}/{len(results)} ok  "
        f"({summary['n_fail']} failed)  total {summary['elapsed_s']:.1f}s"
    )
    print(f"[eval_all_metrics] status → {status_path}")
    print(f"[eval_all_metrics] per-mode reports under → {save_dir}/<mode>/")
    return 0 if summary["n_fail"] == 0 and len(results) == len(modes) else 1


def main(argv: Sequence[str] | None = None) -> int:
    if not os.path.isfile(_PLAY_METRICS):
        raise FileNotFoundError(f"Missing play_with_metrics.py next to this script: {_PLAY_METRICS}")

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    modes = expand_suite(args.modes if args.modes else args.suite)

    if args.list:
        print(" ".join(modes))
        print(f"# {len(modes)} modes", file=sys.stderr)
        return 0

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required (unless using --list)")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint path not found: {args.checkpoint}")

    if _inside_kit() and not args.one_shot:
        print(
            "[WARN] Running inside Kit (isaaclab.sh -p). Nested Kit spawns are unsafe; "
            "forcing --one_shot. Prefer: python eval_all_metrics.py ...",
            file=sys.stderr,
        )
        args.one_shot = True

    common, save_dir = _common_play_args(args)
    print(f"[eval_all_metrics] task={args.task}")
    print(f"[eval_all_metrics] checkpoint={os.path.abspath(args.checkpoint)}")
    print(f"[eval_all_metrics] modes ({len(modes)}): {', '.join(modes)}")
    print(f"[eval_all_metrics] save_dir={save_dir}")

    try:
        if args.one_shot:
            return run_one_shot(modes, common)
        return run_per_mode(modes, common, save_dir=save_dir, fail_fast=args.fail_fast)
    finally:
        eval_paths.publish_eval_dir(save_dir)


if __name__ == "__main__":
    raise SystemExit(main())
