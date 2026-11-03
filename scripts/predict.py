#!/usr/bin/env python3
"""PhytoTrace AI — Single and Batch CLI Prediction Tool."""

import argparse
import json
from pathlib import Path
import cv2

from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.utils.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description="Run PhytoTrace diagnostic inference on a leaf photograph.")
    parser.add_argument("--image", type=str, required=True, help="Path to input tomato leaf image")
    parser.add_argument("--save_overlay", type=str, default=None, help="Optional output path for annotated overlay")
    args = parser.parse_args()

    img_path = resolve_path(args.image)
    if not img_path.is_file():
        print(f"Error: File not found at {img_path}")
        return

    pipeline = InferencePipeline()
    result = pipeline.predict(img_path, image_id=img_path.stem)

    overlay = result.pop("_annotated_image", None)
    if args.save_overlay and overlay is not None:
        out_p = resolve_path(args.save_overlay)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
        print(f"Saved diagnostic overlay to {out_p}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

