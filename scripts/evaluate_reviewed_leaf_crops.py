"""Compare the fixed baseline and crop-trained YOLO on matched development inputs."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from phytotrace.models.mine_segmenter import MineSegmenter

ROOT = Path("reports/yolo_crop_experiment")
BASELINE = Path("models/checkpoints/mine_segmenter/best.pt")
CANDIDATE = Path("runs/segment/models/checkpoints/astra_reviewed_leaf_crop/weights/best.pt")


def main():
    torch.set_num_threads(4)
    comparison = {}
    for name, checkpoint in (("baseline", BASELINE), ("crop_trained", CANDIDATE)):
        metrics = {}
        for view, data in (("full", "astra_crop_source"), ("crop", "astra_crop_experiment")):
            model = YOLO(str(checkpoint))
            result = model.val(data=str(Path("data", data, "dataset.yaml").resolve()), device="cpu",
                               imgsz=640, batch=1, workers=0, plots=False, verbose=False,
                               project=str(ROOT.resolve()), name=f"{name}_{view}")
            metrics[view] = {key: float(value) for key, value in result.results_dict.items()}
        pipe = MineSegmenter(str(checkpoint), confidence_threshold=0.20, device="cpu")
        negatives = {}
        for file in sorted(Path("data/astra_crop_experiment/labels/train").glob("*.txt")):
            if file.stat().st_size:
                continue
            image = next(Path("data/astra_crop_experiment/images/train").glob(file.stem + ".*"))
            results = pipe.predict(cv2.imread(str(image)))
            negatives[file.stem] = results["raw_candidates_count"]
        phone = cv2.imread("reports/debug_segmentation/benchmark_whatsapp_orig.jpg")
        polygon = json.loads(Path("reports/evidence_repair/browser_results.json").read_text())[
            "whatsapp-polygon"]["leaf_polygon"]
        points = np.asarray(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        crop = phone[y:y+h, x:x+w]
        selection = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(selection, [points - [x, y]], 255)
        result = pipe.predict(crop, leaf_mask=selection, diagnostics=True)
        overlay = phone.copy()
        cv2.polylines(overlay, [points], True, (100, 255, 130), 6)
        for candidate in result["candidates"]:
            for polygon in candidate["raw_polygons"]:
                contour = np.asarray(polygon, dtype=np.int32) + [x, y]
                cv2.polylines(overlay, [contour.astype(np.int32)], True, (0, 155, 255), 5)
        cv2.imwrite(str(ROOT / f"whatsapp_{name}_raw.jpg"), overlay)
        comparison[name] = {"validation": metrics, "training_negative_proposal_counts": negatives,
                            "whatsapp_raw_candidates": result["candidates"]}
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
