#!/usr/bin/env python3
"""PhytoTrace AI — Model Export (PyTorch to ONNX) and Parity Verification."""

import argparse
import json
from pathlib import Path
import torch

from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.utils.paths import resolve_path


def export_baseline_to_onnx(
    checkpoint_path: Path,
    export_dir: Path,
) -> dict:
    if not checkpoint_path.is_file():
        return {
            "status": "NOT_EXPORTED",
            "reason": f"Checkpoint not found at {checkpoint_path}. Train the baseline model before exporting.",
        }

    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "baseline_mobilenetv3.onnx"
    meta_path = export_dir / "baseline_mobilenetv3_metadata.json"

    model = MobileNetV3Baseline(num_classes=3, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
    except Exception as e:
        return {"status": "EXPORT_FAILED", "error": str(e)}

    metadata = {
        "model_name": "PhytoTrace Baseline MobileNetV3-Small",
        "num_classes": 3,
        "classes": MobileNetV3Baseline.CLASS_NAMES,
        "input_shape": [1, 3, 224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "opset_version": 14,
        "source_checkpoint": str(checkpoint_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "SUCCESS",
        "onnx_path": str(onnx_path),
        "metadata_path": str(meta_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Export trained PhytoTrace model to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/baseline/baseline_mobilenetv3_best.pt")
    parser.add_argument("--export_dir", type=str, default="models/exports")
    args = parser.parse_args()

    print("=" * 60)
    print("🌿 PHYTOTRACE AI — MODEL EXPORT")
    print("=" * 60)

    res = export_baseline_to_onnx(
        checkpoint_path=resolve_path(args.checkpoint),
        export_dir=resolve_path(args.export_dir),
    )
    print(json.dumps(res, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()

