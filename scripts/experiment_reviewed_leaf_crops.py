"""Build development-only YOLO crops from reviewed leaf masks and exported labels."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Fresh reviewed YOLO export")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    catalog = {r["original_filename"]: r for r in csv.DictReader(open("data/segmentation/catalog.csv"))}
    splits = json.loads(Path("data/splits/splits.json").read_text())
    args.destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for partition in ("train", "val"):
        images = args.destination / "images" / partition
        labels = args.destination / "labels" / partition
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for file in sorted((args.source / "images" / partition).iterdir()):
            row = catalog[file.name]
            if row["image_id"] not in splits[partition] or row["image_id"] in splits["test"]:
                raise ValueError(f"Unexpected partition for {file.name}")
            image = cv2.imread(str(file))
            mask = cv2.imread(row["leaf_mask_path"], cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or mask.shape != image.shape[:2] or not np.any(mask):
                raise ValueError(f"Missing or misaligned leaf mask: {file.name}")
            h, w = mask.shape
            x, y, width, height = cv2.boundingRect(mask)
            target = (args.source / "labels" / partition / file.with_suffix(".txt").name)
            polygons = [np.asarray(line.split()[1:], dtype=float).reshape(-1, 2) * [w, h]
                        for line in target.read_text().splitlines() if line.strip()]
            # Include every annotated mine completely, even with minor leaf-mask disagreements.
            points = np.vstack([[[x, y], [x + width, y + height]], *polygons])
            x1, y1 = np.maximum(0, np.floor(points.min(axis=0))).astype(int)
            x2, y2 = np.minimum([w, h], np.ceil(points.max(axis=0))).astype(int)
            encoded = [(p - [x1, y1]) / [x2 - x1, y2 - y1] for p in polygons]
            if any(np.any((p < 0) | (p > 1)) for p in encoded):
                raise ValueError(f"Annotation outside crop: {file.name}")
            if not cv2.imwrite(str(images / file.name), image[y1:y2, x1:x2]):
                raise OSError(f"Could not write {file.name}")
            (labels / target.name).write_text("".join(
                "0 " + " ".join(f"{v:.8f}" for v in p.ravel()) + "\n" for p in encoded
            ))
            manifest.append({"id": row["image_id"], "partition": partition, "file": file.name,
                             "crop": list(map(int, (x1, y1, x2, y2))), "mines": len(polygons)})
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.destination / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(args.destination.resolve()), "train": "images/train",
        "val": "images/val", "names": {0: "mine_damage"},
    }))


if __name__ == "__main__":
    main()
