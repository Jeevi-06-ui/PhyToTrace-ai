"""Save the individual model outputs for the browser-selected WhatsApp leaflet."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.inference.pipeline import InferencePipeline

ROOT = Path("reports/evidence_repair")
SOURCE = Path("reports/debug_segmentation/benchmark_whatsapp_orig.jpg")


def main():
    torch.set_num_threads(4)
    image = check_and_fix_exif_orientation(SOURCE)
    points = json.loads((ROOT / "browser_results.json").read_text())["whatsapp-polygon"]["leaf_polygon"]
    pipe = InferencePipeline()
    deployed_floor = pipe.mine_segmenter.confidence_threshold
    cases = (
        ("automatic", None, deployed_floor),
        ("polygon", points, deployed_floor),
        ("polygon_low_score_diagnostics", points, 0.05),
    )
    for name, polygon, floor in cases:
        pipe.mine_segmenter.confidence_threshold = floor
        result = pipe.predict(image, leaf_polygon=polygon, diagnostics=True)
        output = ROOT / name
        output.mkdir(parents=True, exist_ok=True)
        h, w = image.shape[:2]
        x1, y1, x2, y2 = result["crop_box"] or (0, 0, w, h)
        roi = image[y1:y2, x1:x2]
        leaf = result["_leaf_diagnostics"]
        mine = result["_mine_diagnostics"]
        images = {
            "original.jpg": image,
            "inference_crop.jpg": roi,
            "unet_probability.png": np.rint(leaf["probability_map"] * 65535).astype(np.uint16),
            "unet_binary.png": leaf["unet_mask"],
            "colour_heuristic.png": pipe.leaf_segmenter.segment_leaf_heuristic(roi),
            "boundary_original.png": result["_leaf_mask"],
            "final_overlay.jpg": result["_annotated_image"],
        }
        for candidate_id, mask in mine["raw_masks"].items():
            canvas = np.zeros((h, w), dtype=np.uint8)
            canvas[y1:y2, x1:x2] = mask
            images[f"{candidate_id}_native.png"] = mask
            images[f"{candidate_id}_reprojected.png"] = canvas
            record = next(c for c in result["_candidates"] if c["candidate_id"] == candidate_id)
            assert cv2.countNonZero(canvas) == record["raw_pixels"]
            assert cv2.countNonZero(cv2.bitwise_and(canvas, result["_leaf_mask"])) == record["in_leaf_pixels"]
        for filename, raster in images.items():
            assert cv2.imwrite(str(output / filename), raster)
        rows = result["_candidates"]
        fields = ["candidate_id", "class_name", "confidence", "bbox", "raw_pixels",
                  "in_leaf_pixels", "leaf_containment_pct", "decision", "reasons"]
        with (output / "candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary = {key: value for key, value in result.items() if not key.startswith("_")}
        (output / "result.json").write_text(json.dumps(summary, indent=2))
        print(name, floor, result["failure_classification"],
              [(c["confidence"], c["raw_pixels"], c["leaf_containment_pct"], c["reasons"]) for c in rows])


if __name__ == "__main__":
    main()
