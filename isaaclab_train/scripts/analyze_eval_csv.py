#!/usr/bin/env python3
"""Generate graphs for each eval_report_*.csv without launching Isaac Sim.

Each CSV gets a matching PNG/PDF set written next to it (or in --out-dir):

    survival_thresholds, fail_times, crash_vs_survive, term_reasons,
    spawn_scatter, metrics_overview, end_xy, plus a thresholds.txt summary.

Examples::

    python scripts/analyze_eval_csv.py path/to/eval_report_2026-08-13T12-00-00.csv
    python scripts/analyze_eval_csv.py logs/skrl/cyclerl_logs/.../eval/
    python scripts/analyze_eval_csv.py a.csv b.csv --out-dir ./graphs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the CSV plotter without loading CycleRL.tasks (gymnasium / Isaac).
_CYCLERL = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "CycleRL"
    / "CycleRL"
    / "tasks"
    / "manager_based"
    / "cyclerl"
)
sys.path.insert(0, str(_CYCLERL))

from eval_csv_plots import find_eval_csvs, plot_eval_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate graphs for each eval_report_*.csv.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSV file(s) or directories to search recursively for eval_report*.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Write all graphs here. Default: next to each CSV.",
    )
    args = parser.parse_args()

    csvs: list[str] = []
    for path in args.paths:
        csvs.extend(find_eval_csvs(path))
    csvs = sorted(set(csvs))
    if not csvs:
        print("No eval_report*.csv found.", file=sys.stderr)
        return 1

    n_ok = 0
    for csv_path in csvs:
        print(f"[analyze_eval_csv] {csv_path}")
        plots = plot_eval_csv(csv_path, out_dir=args.out_dir)
        if plots:
            n_ok += 1
            for name, dest in plots.items():
                print(f"  {name}: {dest}")
    print(f"[analyze_eval_csv] graphed {n_ok}/{len(csvs)} CSV files.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
