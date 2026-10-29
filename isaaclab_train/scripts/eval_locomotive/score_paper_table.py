#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/eval/score_paper_table.py (imports/paths rewritten for trackstand_rl locomotive).
"""Score a Paper-v0 eval/ directory against Froude-scaled Table IV / VI targets.

    python scripts/eval_locomotive/score_paper_table.py logs/skrl/locomotive/Locomotive-Cylinder-Aligned-v0/<run>/eval

Reads each ``<mode>/summary.json`` (or ``eval_report_*.json`` summary block).
Headline suite modes are the ten Table VI conditions. Frontier metrics CAT / MNT /
MSS come from paper_roll_sweep / paper_noise_sweep / speed_sweep when present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TABLE_VI = (
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

# Froude-scaled Table IV / VI targets (see the paper-reproduction plan).
TARGETS = {
    "bsr": 0.999,
    "ste_deg": 1.15,
    "vte_ms": 0.09,
    "brt_s": 0.64,
    "cat_deg": 27.8,
    "mss_ms": 0.53,
    "mnt": 0.32,
}


def _load_summary(mode_dir: Path) -> dict:
    named = mode_dir / "summary.json"
    if named.is_file():
        return json.loads(named.read_text())
    reports = sorted(mode_dir.glob("eval_report_*.json"))
    if not reports:
        return {}
    data = json.loads(reports[-1].read_text())
    return data.get("summary") or data


def _fmt(value: float | None, spec: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return format(float(value), spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eval_dir", type=Path, help="Run eval/ directory")
    args = p.parse_args(argv)
    root = args.eval_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"eval dir not found: {root}")

    rows: list[tuple[str, dict]] = []
    for mode in TABLE_VI:
        path = root / mode
        rows.append((mode, _load_summary(path) if path.is_dir() else {}))

    print(f"Eval root: {root}")
    print(f"{'mode':<22} {'BSR':>8} {'STE°':>8} {'VTE':>8} {'BRT s':>8}")
    bsrs: list[float] = []
    for mode, s in rows:
        bsr = s.get("bsr")
        if isinstance(bsr, (int, float)) and math.isfinite(bsr):
            bsrs.append(float(bsr))
        print(
            f"{mode:<22} {_fmt(s.get('bsr'), '.1%'):>8} "
            f"{_fmt(s.get('ste_deg'), '.2f'):>8} "
            f"{_fmt(s.get('vte_ms'), '.3f'):>8} "
            f"{_fmt(s.get('brt_s'), '.2f'):>8}"
        )

    # Aggregate Table II from the clean flat row when present, else mean of suite.
    flat = next((s for m, s in rows if m in ("paper_no_noise", "paper_flat") and s), {})
    roll = _load_summary(root / "paper_roll_sweep") if (root / "paper_roll_sweep").is_dir() else {}
    noise = _load_summary(root / "paper_noise_sweep") if (root / "paper_noise_sweep").is_dir() else {}
    speed = _load_summary(root / "speed_sweep") if (root / "speed_sweep").is_dir() else {}

    measured = {
        "bsr": flat.get("bsr", (sum(bsrs) / len(bsrs)) if bsrs else float("nan")),
        "ste_deg": flat.get("ste_deg", float("nan")),
        "vte_ms": flat.get("vte_ms", float("nan")),
        "brt_s": flat.get("brt_s", float("nan")),
        "cat_deg": roll.get("cat_deg", float("nan")),
        "mnt": noise.get("mnt", float("nan")),
        "mss_ms": speed.get("mss_ms", float("nan")),
    }

    print()
    print("Table II vs Froude-scaled targets (flat / frontier ladders)")
    print(f"{'metric':<10} {'measured':>10} {'target':>10} {'pass':>6}")
    all_ok = True
    for key, target in TARGETS.items():
        val = measured.get(key)
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
            print(f"{key:<10} {'n/a':>10} {target:>10} {'--':>6}")
            all_ok = False
            continue
        higher = key in ("bsr", "cat_deg", "mnt")
        ok = val >= target if higher else val <= target
        all_ok = all_ok and ok
        print(f"{key:<10} {val:10.3f} {target:10.3f} {'yes' if ok else 'NO':>6}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
