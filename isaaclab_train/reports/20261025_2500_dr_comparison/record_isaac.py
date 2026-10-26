"""Reuse the audited 75300 recorder for the DR-trained 2500 checkpoint.

Only this report-local process is adapted. Task/evaluator and old reports are
never rewritten. The reused recorder rejects non-matching saved configuration.
"""

import hashlib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "20261021_75300_actuation/record_eval.py"
source = TEMPLATE.read_text()
assert source.count("2026-10-21_00-41-38") == 1
assert source.count('"training_dr": False') == 1
assert "--dr" in sys.argv and "--pure_survival" not in sys.argv
# The training snapshot leaves this optional field None. Current source makes
# its resolved zero explicit; retain the exact saved spelling in this run only.
override = "env.scene.robot.actuators.drive.viscous_friction=null"
assert not any(arg.startswith("env.scene.robot.actuators.drive.viscous_friction=") for arg in sys.argv)
sys.argv.append(override)
source = source.replace("2026-10-21_00-41-38", "2026-10-21_11-18-29").replace("75300", "2500")
source = source.replace('"training_dr": False', '"training_dr": True')
source = source.replace('paths += [source_checkpoint,', 'paths += [Path(TEMPLATE_PATH), source_checkpoint,')
assert "Path(TEMPLATE_PATH)" in source
print(f"Reused recorder: {TEMPLATE}", flush=True)
print(f"Recorder SHA256: {hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()}", flush=True)
print(f"Runtime-only saved-configuration override: {override}", flush=True)
namespace = {"__name__": "__2500_capture_adapter__", "__file__": str(Path(__file__).resolve()),
             "TEMPLATE_PATH": str(TEMPLATE)}
exec(compile(source, str(TEMPLATE), "exec"), namespace)
