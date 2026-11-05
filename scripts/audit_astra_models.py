#!/usr/bin/env python3
"""Generate real inference artifacts without reading the frozen test set for tuning."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from phytotrace.data.validators import check_and_fix_exif_orientation
from phytotrace.evaluation.metrics import compute_binary_segmentation_metrics
from phytotrace.inference.pipeline import InferencePipeline
from phytotrace.models.leaf_segmenter import LeafSegmenter
from phytotrace.models.mine_segmenter import MineSegmenter


def audit_image(pipe, filename, output, crop=None):
    image = check_and_fix_exif_orientation(filename)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = crop or (0, 0, w, h)
    roi = image[y1:y2, x1:x2]
    result = pipe.predict(image, crop_box=crop, diagnostics=True)
    leaf = result["_leaf_diagnostics"]
    mine = result["_mine_diagnostics"]
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "01_original.jpg"), image)
    if leaf and leaf.get("probability_map") is not None:
        # Fixed [0, 1] intensity scale preserves evidence of a near-constant output.
        cv2.imwrite(str(output / "02_unet_probability.png"), np.rint(leaf["probability_map"] * 65535).astype(np.uint16))
        cv2.imwrite(str(output / "03_unet_mask.png"), leaf["unet_mask"])
    cv2.imwrite(str(output / "04_colour_heuristic.png"), pipe.leaf_segmenter.segment_leaf_heuristic(roi))
    raw_union = np.zeros(roi.shape[:2], dtype=np.uint8)
    for candidate_id, mask in (mine or {}).get("raw_masks", {}).items():
        cv2.imwrite(str(output / f"05_{candidate_id}.png"), mask)
        cv2.bitwise_or(raw_union, mask, dst=raw_union)
    cv2.imwrite(str(output / "05_yolo_raw_union.png"), raw_union)
    if result["_mine_mask"] is not None:
        cv2.imwrite(str(output / "06_yolo_accepted.png"), result["_mine_mask"])
    cv2.imwrite(str(output / "07_final_overlay.jpg"), result["_annotated_image"])
    candidates = result["_candidates"]
    (output / "08_candidates.json").write_text(json.dumps(candidates, indent=2))
    fields = ["candidate_id", "class_name", "confidence", "bbox", "raw_pixels",
              "in_leaf_pixels", "leaf_containment_pct", "decision", "reasons"]
    with (output / "08_candidates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/astra_cv"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    original_leaf = "models/checkpoints/leaf_segmenter/leaf_unet_best.pt"
    experiment_leaf = "models/checkpoints/astra_leaf_experiment/leaf_unet_best.pt"
    original_mine = "models/checkpoints/mine_segmenter/best.pt"
    experiment_mine = "models/checkpoints/astra_mine_experiment/weights/best.pt"
    pipe = InferencePipeline()
    photos = Path("data/raw/original_photos")
    phone = next(Path("data/inspections/raw").glob("*024600*"))
    cases = [("reviewed_positive", photos / "5558326.jpg", None),
             ("reviewed_negative", photos / "BLK_3_1002_PL010_H.JPG", None),
             ("greenhouse_automatic", phone, None),
             ("greenhouse_manual", phone, [816, 1169, 2268, 2943])]
    summary = {}
    for name, leaf, mine in (("original", original_leaf, original_mine),
                             ("new_leaf_original_yolo", experiment_leaf, original_mine),
                             ("new_leaf_new_yolo", experiment_leaf, experiment_mine),
                             ("new_leaf_last_yolo", experiment_leaf, experiment_mine.replace("best.pt", "last.pt"))):
        pipe.leaf_segmenter = LeafSegmenter(leaf, "cpu")
        pipe.mine_segmenter = MineSegmenter(mine, pipe.mine_segmenter.confidence_threshold, device="cpu")
        summary[name] = {
            label: audit_image(pipe, filename, args.out / name / label, crop)
            for label, filename, crop in cases
        }
    val = set(json.loads(Path("data/splits/splits.json").read_text())["val"])
    catalog = csv.DictReader(Path("data/segmentation/catalog.csv").open())
    old, new = (LeafSegmenter(path, "cpu") for path in (original_leaf, experiment_leaf))
    comparisons = {}
    for row in catalog:
        if row["image_id"] not in val or row["ready_for_training"].lower() != "true":
            continue
        im = check_and_fix_exif_orientation(photos / row["original_filename"])
        truth = cv2.imread(row["leaf_mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        comparisons[row["image_id"]] = {
            "legacy": compute_binary_segmentation_metrics(old.segment(im, diagnostics=True)["unet_mask"] > 0, truth),
            "retrained": compute_binary_segmentation_metrics(new.segment(im, diagnostics=True)["unet_mask"] > 0, truth),
            "colour_heuristic": compute_binary_segmentation_metrics(new.segment_leaf_heuristic(im) > 0, truth),
        }
    summary["validation_leaf"] = comparisons
    (args.out / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
