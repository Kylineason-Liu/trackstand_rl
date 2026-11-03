"""Native-log isolation and output preservation for the separate v2 CLIs."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco_v2.reporting import _evaluation_output_base
from newbike_mujoco_v2.runtime import preserve_working_directory


class V2RuntimeTests(unittest.TestCase):
    def test_working_directory_is_restored_after_success_and_exception(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            @preserve_working_directory
            def run(fail):
                os.chdir(temporary)
                if fail:
                    raise RuntimeError("fixture")
                return Path.cwd()

            self.assertEqual(run(False), Path(temporary))
            self.assertEqual(Path.cwd(), original)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                run(True)
            self.assertEqual(Path.cwd(), original)

    def test_runtime_directory_reserves_a_prefix_even_after_failed_startup(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
                "newbike_mujoco_v2.reporting.time.strftime", return_value="202611032100"):
            first = _evaluation_output_base(temporary, "model_26600.pt", False)
            Path(first + "_runtime").mkdir()
            second = _evaluation_output_base(temporary, "model_26600.pt", False)
            self.assertEqual(second, first + "_2")
            Path(second + "_summary.json").write_text("{}")
            self.assertEqual(_evaluation_output_base(temporary, "model_26600.pt", False), first + "_3")


if __name__ == "__main__":
    unittest.main()
