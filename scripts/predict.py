#!/usr/bin/env python3
"""PhytoTrace AI — Single Image Prediction & Class Score Breakdown CLI."""

import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image

from phytotrace.data.augmentation import get_baseline_transforms
from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.device import get_device
from phytotrace.models.baseline import MobileNetV3Baseline
from phytotrace.preprocessing.quality import assess_image_quality
from phytotrace.utils.paths import resolve_path


def predict_single_image(
    image_path: Path,
    checkpoint_path: Path,
    confidence_threshold: float = 0.50,
    uncertainty_margin: float = 0.15,
) -> dict:
    start_time = time.time()
    image_path = resolve_path(image_path)
    checkpoint_path = resolve_path(checkpoint_path)

    if not image_path.is_file():
        return {
            "image_filename": str(image_path.name),
            "status": "ERROR",
            "message": f"Image file not found at {image_path}",
        }

    # 1. Load image and check quality
    bgr = check_and_fix_exif_orientation(image_path)
    if bgr is None:
        return {
            "image_filename": str(image_path.name),
            "status": "ERROR",
            "message": "Image file is corrupt or cannot be decoded.",
        }

    qc = assess_image_quality(bgr)

    # 2. Check model checkpoint
    if not checkpoint_path.is_file():
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "image_filename": str(image_path.name),
            "model_version": "PhytoTrace-v0.1.0-baseline",
            "checkpoint_status": "model_not_trained",
            "checkpoint_path": str(checkpoint_path),
            "predicted_class": None,
            "class_scores": None,
            "quality_warnings": qc.get("reasons", []),
            "processing_time_ms": elapsed_ms,
            "is_uncertain": True,
            "uncertainty_explanation": f"Model weights not found at {checkpoint_path}. Train the baseline model before running inference.",
            "scientific_disclaimer": "Classifier confidence alone does NOT establish a biologically confirmed diagnosis of insect species (Tuta vs Liriomyza).",
        }

    device = get_device("auto")
    model = MobileNetV3Baseline(num_classes=3, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    # 3. Preprocess and run inference
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    transforms = get_baseline_transforms(is_training=False)
    tensor = transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_probabilities(tensor).squeeze(0).cpu().numpy()

    class_names = MobileNetV3Baseline.CLASS_NAMES
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])

    scores = {name: round(float(probs[i]), 4) for i, name in enumerate(class_names)}

    # Uncertainty evaluation
    # Sort top 2 probabilities
    sorted_probs = sorted(probs, reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1])
    is_uncertain = bool((pred_prob < confidence_threshold) or (margin < uncertainty_margin))

    if not qc.get("is_acceptable", True):
        is_uncertain = True
        uncertainty_explanation = f"Quality warning: {', '.join(qc.get('reasons', []))}"
    elif is_uncertain:
        uncertainty_explanation = f"Low confidence ({pred_prob:.2f}) or narrow margin ({margin:.2f}) between top classes."
    else:
        uncertainty_explanation = "Prediction confidence exceeds decision threshold."

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "image_filename": str(image_path.name),
        "model_version": "PhytoTrace-v0.1.0-baseline",
        "checkpoint_status": "loaded",
        "checkpoint_path": str(checkpoint_path),
        "predicted_class": pred_class,
        "class_scores": scores,
        "quality_warnings": qc.get("reasons", []),
        "processing_time_ms": elapsed_ms,
        "is_uncertain": bool(is_uncertain),
        "uncertainty_explanation": uncertainty_explanation,
        "scientific_disclaimer": "Classifier confidence alone does NOT establish a biologically confirmed diagnosis of insect species (Tuta vs Liriomyza). Secondary expert examination is required.",
    }


def main():
    parser = argparse.ArgumentParser(description="Run PhytoTrace baseline classifier prediction on a leaf photograph.")
    parser.add_argument("--image", type=str, required=True, help="Path to input tomato leaf image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/baseline/baseline_mobilenetv3_best.pt",
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.50)
    args = parser.parse_args()

    res = predict_single_image(
        image_path=Path(args.image),
        checkpoint_path=Path(args.checkpoint),
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
