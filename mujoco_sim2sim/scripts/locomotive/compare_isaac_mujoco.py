#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/mujoco_paper/compare_isaac_mujoco.py (imports/paths rewritten for trackstand_rl locomotive).
"""Isaac → MuJoCo transfer table for the CycleRL paper suite (paper Table IX style).

Takes a JSON manifest::

    [{"label": "act_refine_lo (cylinder)",
      "isaac": ["<run>/eval_paper_full", "<run>/eval_paper_full_<stamp>"],
      "mujoco": "<run>/eval_mujoco"}, ...]

Per mode the newest Isaac ``summary.json`` across the listed directories is used.
CAT / MNT / MSS are recomputed on both sides from saved episodes with
``eval/paper_metrics.py`` so older Isaac summaries (previous frontier rule) are
scored the same way as MuJoCo. Transfer ratio = MuJoCo / Isaac for higher-is-better
metrics and Isaac / MuJoCo for lower-is-better, capped at 1.0 (paper Table IX).

    python3 mujoco_sim2sim/scripts/locomotive/compare_isaac_mujoco.py manifest.json --out report.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[3]  # trackstand_rl root
_PM_PATH = _REPO / "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/locomotive" / "eval/paper_metrics.py"
_spec = importlib.util.spec_from_file_location("cyclerl_paper_metrics", _PM_PATH)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)

TABLE_VI = (
    "paper_low_speed", "paper_normal_speed", "paper_no_noise", "paper_max_noise", "paper_dropout",
    "paper_flat", "paper_gravel", "paper_rough", "paper_slope", "paper_steps",
)
FRONTIER = {"cat_deg": "paper_roll_sweep", "mnt": "paper_noise_sweep", "mss_ms": "speed_sweep"}
# key: (label, higher_is_better, Froude target)
TABLE_II = (
    ("bsr", "BSR", True, 0.999),
    ("ste_deg", "STE °", False, 1.15),
    ("vte_ms", "VTE m/s", False, 0.09),
    ("brt_s", "BRT s", False, 0.64),
    ("cat_deg", "CAT °", True, 27.8),
    ("mss_ms", "MSS m/s", False, 0.53),
    ("mnt", "MNT", True, 0.32),
)
_EP_FIELDS = ("complete", "bsr_balanced", "spawn_roll_deg", "noise_rho", "spawn_cmd_vx_ms")


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None


def _mode_dirs(roots: list[Path]) -> dict[str, Path]:
    """Newest ``<root>/<mode>`` holding a summary.json, per mode."""
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"eval dir not found: {root}")
        for mode_dir in root.iterdir():
            summary = mode_dir / "summary.json"
            if summary.is_file():
                best = found.get(mode_dir.name)
                if best is None or summary.stat().st_mtime > (best / "summary.json").stat().st_mtime:
                    found[mode_dir.name] = mode_dir
    return found


def _episodes(mode_dir: Path) -> list[SimpleNamespace]:
    """Saved per-episode records (MuJoCo episodes.json or the newest Isaac eval_report)."""
    path = mode_dir / "episodes.json"
    if path.is_file():
        raw = json.loads(path.read_text())
    else:
        reports = sorted((mode_dir / "reports").glob("eval_report_*.json"))
        if not reports:
            return []
        raw = json.loads(reports[-1].read_text()).get("episodes", [])
    out = []
    for ep in raw:
        vals = {k: ep.get(k) for k in _EP_FIELDS}
        vals["complete"] = bool(vals["complete"]) if vals["complete"] is not None else True
        vals["bsr_balanced"] = bool(vals["bsr_balanced"])
        for k in ("spawn_roll_deg", "noise_rho", "spawn_cmd_vx_ms"):
            vals[k] = float(vals[k]) if isinstance(vals[k], (int, float)) else float("nan")
        out.append(SimpleNamespace(**vals))
    return out


def load_side(roots: list[Path]) -> dict:
    dirs = _mode_dirs(roots)
    side = {"modes": {}, "frontier": {}, "sources": {}}
    for mode, d in dirs.items():
        side["modes"][mode] = json.loads((d / "summary.json").read_text())
        side["sources"][mode] = str(d)
    frontier_fn = {
        "cat_deg": pm.analyze_critical_angle,
        "mnt": pm.analyze_noise_tolerance,
        "mss_ms": pm.analyze_min_sustaining_speed,
    }
    for key, mode in FRONTIER.items():
        eps = _episodes(dirs[mode]) if mode in dirs else []
        side["frontier"][key] = frontier_fn[key](eps) if eps else float("nan")
    return side


def table_ii_value(side: dict, key: str) -> float | None:
    if key in side["frontier"]:
        return _num(side["frontier"][key])
    return _num(side["modes"].get("paper_flat", {}).get(key))


def ratio(isaac: float | None, mujoco: float | None, higher: bool) -> str:
    if isaac is None or mujoco is None:
        return "—"
    if higher:
        r = mujoco / isaac if isaac > 0 else (1.0 if mujoco >= isaac else 0.0)
    else:
        r = isaac / mujoco if mujoco > 0 else 1.0
    return f"{min(1.0, r):.2f}"


def _fmt(key: str, v: float | None) -> str:
    if v is None:
        return "n/a"
    if key == "bsr":
        return f"{100 * v:.1f}%"
    return f"{v:.3f}" if key in ("vte_ms", "mnt") else f"{v:.2f}"


def build(manifest: list[dict]) -> tuple[str, dict]:
    lines = [
        "# Isaac → MuJoCo transfer (CycleRL paper suite)",
        "",
        "Ratio = MuJoCo/Isaac (higher-is-better) or Isaac/MuJoCo (lower-is-better), capped at 1.00 as in "
        "paper Table IX. CAT / MNT / MSS recomputed from episodes with `paper_metrics.py` on both sides. "
        "Terrain rows (rough / slope / steps) are Isaac-only.",
        "",
    ]
    data: dict = {}
    for entry in manifest:
        label = entry["label"]
        isaac = load_side([Path(p) for p in entry["isaac"]])
        mujo = load_side([Path(entry["mujoco"])])
        data[label] = {"isaac": isaac, "mujoco": mujo}
        lines += [f"## {label}", "", "| Table II | target | Isaac | MuJoCo | ratio |", "|---|---:|---:|---:|---:|"]
        for key, name, higher, target in TABLE_II:
            iv, mv = table_ii_value(isaac, key), table_ii_value(mujo, key)
            tgt = f"{'≥' if higher else '≤'} {target}"
            lines.append(f"| {name} | {tgt} | {_fmt(key, iv)} | {_fmt(key, mv)} | {ratio(iv, mv, higher)} |")
        lines += ["", "| Table VI BSR | Isaac | MuJoCo | Isaac STE° | MuJoCo STE° |", "|---|---:|---:|---:|---:|"]
        for mode in TABLE_VI:
            i, m = isaac["modes"].get(mode, {}), mujo["modes"].get(mode, {})
            lines.append(
                f"| {mode} | {_fmt('bsr', _num(i.get('bsr')))} | "
                f"{_fmt('bsr', _num(m.get('bsr'))) if m else 'not modelled' if mode in ('paper_rough', 'paper_slope', 'paper_steps') else 'n/a'} | "
                f"{_fmt('ste_deg', _num(i.get('ste_deg')))} | {_fmt('ste_deg', _num(m.get('ste_deg')))} |"
            )
        lines += ["", "Sources: Isaac " + ", ".join(f"`{p}`" for p in entry["isaac"]) + f"; MuJoCo `{entry['mujoco']}`", ""]
    return "\n".join(lines) + "\n", data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="Markdown report path (a .json is written next to it).")
    args = ap.parse_args()
    report, data = build(json.loads(args.manifest.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    def safe(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [safe(v) for v in o]
        return o

    args.out.with_suffix(".json").write_text(json.dumps(safe(data), indent=1) + "\n", encoding="utf-8")
    print(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
