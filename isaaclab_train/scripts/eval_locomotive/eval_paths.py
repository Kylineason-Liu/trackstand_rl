# Ported from ~/bike_test/CycleRL/scripts/eval/eval_paths.py (imports/paths rewritten for trackstand_rl locomotive).
"""Eval output paths that keep an earlier suite when a re-run starts.

The first suite for a save directory still lands in that directory
(``<run>/eval/<mode>/summary.json``). A later invocation that would replace
any ``summary.json`` is written to a sibling ``<name>_<UTC>/`` instead, and
the run directory records both in ``eval_runs.json`` and ``last_eval_dir.txt``.

Comparison helpers in this module resolve ``baseline_run`` from a run's
``run_meta.json``. Callers must not keep a second copy of that id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

BASELINE_RUN_KEY = "baseline_run"


def has_eval_results(path: Path) -> bool:
    """True when ``path`` already holds a mode summary that a re-run would replace."""
    root = Path(path)
    if not root.is_dir():
        return False
    if (root / "summary.json").is_file():
        return True
    try:
        children = list(root.iterdir())
    except OSError:
        return False
    return any(child.is_dir() and (child / "summary.json").is_file() for child in children)


def modes_would_overwrite(path: Path, modes: Sequence[str]) -> bool:
    """True when any of ``modes`` already has ``summary.json`` under ``path``."""
    root = Path(path)
    return any((root / mode / "summary.json").is_file() for mode in modes)


def allocate_eval_dir(requested: Path, *, overwrite: bool = False, now: datetime | None = None) -> tuple[Path, bool]:
    """Return ``(directory, redirected)``.

    ``redirected`` is true when ``requested`` already had summaries and the
    caller did not pass ``overwrite``. The returned directory exists.
    """
    requested = Path(requested).expanduser().resolve()
    if overwrite or not has_eval_results(requested):
        requested.mkdir(parents=True, exist_ok=True)
        return requested, False

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d_%H-%M-%S")
    base = f"{requested.name}_{stamp}"
    candidate = requested.parent / base
    suffix = 2
    while candidate.exists():
        candidate = requested.parent / f"{base}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate, True


def note_eval_allocation(
    allocated: Path,
    *,
    requested: Path,
    redirected: bool,
    checkpoint: str | None = None,
) -> None:
    """Record this allocation next to the run and inside the eval directory."""
    allocated = Path(allocated).resolve()
    requested = Path(requested).expanduser().resolve()
    allocated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "dir": str(allocated),
        "name": allocated.name,
        "requested": str(requested),
        "redirected": bool(redirected),
        "allocated_utc": allocated_utc,
        "checkpoint": checkpoint,
    }
    provenance = allocated / "eval_run.json"
    provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    parent = allocated.parent
    index_path = parent / "eval_runs.json"
    runs: list[dict] = []
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                runs = [row for row in loaded["runs"] if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            runs = []
    runs.append(record)
    index_path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")


def publish_eval_dir(allocated: Path) -> None:
    """Point ``last_eval_dir.txt`` at ``allocated`` once it actually has summaries.

    A re-run that dies before writing a summary leaves the previous pointer in
    place, so readers keep the earlier suite.
    """
    allocated = Path(allocated).resolve()
    if not has_eval_results(allocated):
        return
    (allocated.parent / "last_eval_dir.txt").write_text(str(allocated) + "\n", encoding="utf-8")


def read_last_eval_dir(run_dir: Path) -> Path | None:
    """Latest published eval directory under ``run_dir`` that still has summaries."""
    pointer = Path(run_dir) / "last_eval_dir.txt"
    if not pointer.is_file():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_dir() and has_eval_results(path):
        return path
    return None


def load_run_meta(path: Path) -> dict:
    """Load a ``run_meta.json`` object. Missing or invalid files return ``{}``."""
    meta_path = Path(path)
    if meta_path.is_dir():
        meta_path = meta_path / "run_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def baseline_run_id(meta: dict, *, source: str) -> str:
    """Return ``baseline_run`` or exit-style ``KeyError`` with ``source`` in the message."""
    raw = meta.get(BASELINE_RUN_KEY)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    raise KeyError(f"{source} has no {BASELINE_RUN_KEY} in run_meta.json")


def resolve_logged_run(run_id: str, *, repo: Path, task: str) -> Path:
    """Resolve a run id or path to a log directory.

    Absolute paths and existing directories win. Otherwise the id is looked up
    under ``logs/skrl/locomotive/<task>/``. A unique prefix match is accepted
    so a stamp without the ``_ppo_torch_bicycle_ppo`` suffix still resolves.
    """
    raw = str(run_id).strip()
    direct = Path(raw).expanduser()
    if direct.is_dir():
        return direct.resolve()
    root = Path(repo) / "logs" / "skrl" / "locomotive" / task
    exact = root / raw
    if exact.is_dir():
        return exact.resolve()
    if root.is_dir():
        matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(raw))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise FileNotFoundError(f"baseline_run {raw!r} matches more than one run under {task}: {names}")
    raise FileNotFoundError(f"baseline_run {raw!r} not found under {root}")
