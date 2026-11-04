"""Keep this policy's ONNX graph and weights in one hashable, portable file."""

from pathlib import Path

import onnx

root = Path(__file__).resolve().parents[2]
path = root / "policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx"
model = onnx.load(path)
onnx.external_data_helper.convert_model_from_external_data(model)
onnx.save_model(model, path, save_as_external_data=False)
verified = onnx.load(path, load_external_data=False)
assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in verified.graph.initializer)
onnx.checker.check_model(verified)
path.with_suffix(".onnx.data").unlink(missing_ok=True)
print(f"Self-contained ONNX checker passed: {path}, {path.stat().st_size} bytes")
