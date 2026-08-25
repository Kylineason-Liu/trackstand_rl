#!/usr/bin/env python3
"""Convert an ONNX model → TensorFlow SavedModel → quantized INT8 TFLite.

Usage:
    # Step A — ONNX → TensorFlow SavedModel (via onnx2tf CLI):
    pip install onnx2tf
    onnx2tf -i bicycle_policy.onnx -o saved_model_output

    # Step B — SavedModel → quantized TFLite:
    python scripts/deployment/quantize_tflite.py \
        --saved-model saved_model_output \
        --obs-dim 13 \
        --output bicycle_policy_quantized.tflite

The representative dataset is drawn from the preprocessor's known range
(clipped at ±5σ) or from uniform randn if no preprocessor info is available.
"""

import argparse
import os

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="INT8-quantize a TF SavedModel.")
    parser.add_argument("--saved-model", type=str, required=True,
                        help="Path to the TF SavedModel directory (onnx2tf output).")
    parser.add_argument("--obs-dim", type=int, required=True,
                        help="Observation dimension.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tflite path. Default: derived from saved-model dir.")
    parser.add_argument("--num-calibration", type=int, default=200,
                        help="Number of representative samples (default 200).")
    parser.add_argument("--no-int8", action="store_true",
                        help="Skip INT8; produce a Float16 model instead.")
    args = parser.parse_args()

    # ---- 0. Derive output name from saved-model path -----------------------
    if args.output is None:
        sm = os.path.abspath(args.saved_model)
        suffix = "_int8.tflite" if not args.no_int8 else "_float16.tflite"
        args.output = os.path.basename(sm.rstrip("/")) + suffix
        print(f"[INFO] Output name derived from saved-model: {args.output}")

    # ---- 1. Representative dataset ----------------------------------------
    # For best results, use real observation samples collected during simulation.
    # Here we use a unit-normal distribution clipped to ±5σ as a safe default.
    def representative_dataset():
        for _ in range(args.num_calibration):
            data = np.random.randn(1, args.obs_dim).astype(np.float32).clip(-5.0, 5.0)
            yield [data]

    # ---- 2. Convert --------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.no_int8:
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"[INFO] TFLite model saved to: {args.output}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
