"""YOLO instance segmentation dataset exporter for PhytoTrace Phase 3.

Converts verified CVAT COCO polygon annotations and reviewer-audited catalog records
into Ultralytics YOLO segmentation format:
- images/train, images/val
- labels/train, labels/val (normalized polygon coordinates [0, 1])
- Empty label files for confirmed negative (healthy/no-mine) samples
- Strict exclusion of the frozen held-out test partition
"""

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import yaml
from PIL import Image, ImageOps

from phytotrace.segmentation.io import parse_coco_segmentation_json, resolve_filename_match
from phytotrace.segmentation.schemas import ReviewerDecision
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.segmentation.yolo_exporter")


def export_yolo_segmentation_dataset(
    catalog_path: Path,
    coco_json_path: Path,
    splits_json_path: Path,
    photos_dir: Path,
    output_dir: Path,
    yaml_output_path: Path,
    class_id: int = 0,
    class_name: str = "mine_damage",
) -> Dict[str, Any]:
    """Export verified segmentation annotations to YOLO segmentation dataset structure.

    Args:
        catalog_path: Path to data/segmentation/catalog.csv.
        coco_json_path: Path to instances_default.json.
        splits_json_path: Path to data/splits/splits.json.
        photos_dir: Path to raw photos directory.
        output_dir: Path to target yolo_segmentation directory.
        yaml_output_path: Path to output yolo_dataset.yaml.
        class_id: Class integer index for YOLO (default 0).
        class_name: Name of target class (default 'mine_damage').

    Returns:
        Dict summarizing export statistics and integrity verification.
    """
    catalog_p = resolve_path(catalog_path)
    coco_p = resolve_path(coco_json_path)
    splits_p = resolve_path(splits_json_path)
    photos_p = resolve_path(photos_dir)
    out_p = resolve_path(output_dir)
    yaml_p = resolve_path(yaml_output_path)

    if not catalog_p.is_file():
        raise FileNotFoundError(f"Catalog file not found: {catalog_p}")
    if not coco_p.is_file():
        raise FileNotFoundError(f"COCO annotations file not found: {coco_p}")
    if not splits_p.is_file():
        raise FileNotFoundError(f"Splits JSON not found: {splits_p}")
    if not photos_p.is_dir():
        raise FileNotFoundError(f"Photos directory not found: {photos_p}")

    # Load splits
    with open(splits_p, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids: Set[str] = set(splits.get("train", []))
    val_ids: Set[str] = set(splits.get("val", []))
    test_ids: Set[str] = set(splits.get("test", []))

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Frozen split partitions overlap")

    # Load catalog
    cat_df = pd.read_csv(catalog_p, dtype=str).fillna("")
    ready_mask = cat_df["ready_for_training"].astype(str).str.lower().isin(["true", "1"])
    ready_df = cat_df[ready_mask]

    # Load parsed COCO annotations
    parsed_coco = parse_coco_segmentation_json(coco_p)
    coco_filenames = list(parsed_coco.keys())

    # Create destination directories
    for split_name in ("train", "val"):
        (out_p / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_p / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # Clean existing files in split dirs to avoid stale remnants
    for split_name in ("train", "val"):
        for f in (out_p / "images" / split_name).glob("*"):
            if f.is_file():
                f.unlink()
        for f in (out_p / "labels" / split_name).glob("*"):
            if f.is_file():
                f.unlink()

    stats: Dict[str, Any] = {
        "train": {
            "total_images": 0,
            "positive_images": 0,
            "verified_negative_images": 0,
            "total_polygons": 0,
            "image_ids": [],
            "filenames": [],
        },
        "val": {
            "total_images": 0,
            "positive_images": 0,
            "verified_negative_images": 0,
            "total_polygons": 0,
            "image_ids": [],
            "filenames": [],
        },
        "excluded_test_images": 0,
        "excluded_unready_images": 0,
    }

    # Count exclusions
    stats["excluded_test_images"] = int(ready_df["image_id"].isin(test_ids).sum())
    stats["excluded_unready_images"] = int((~ready_mask).sum())

    # Process train and val splits
    for split_name, target_ids in (("train", train_ids), ("val", val_ids)):
        split_records = ready_df[ready_df["image_id"].isin(target_ids)]

        for _, row in split_records.iterrows():
            img_id = row["image_id"]
            orig_filename = row["original_filename"]
            is_neg = str(row["is_verified_negative"]).strip().lower() in ("true", "1")
            rev_dec = str(row["reviewer_decision"]).strip().lower()

            # Find matching COCO record
            match_res = resolve_filename_match(orig_filename, coco_filenames)
            if not match_res.matched_target:
                raise ValueError(
                    f"Failed to match catalog image '{orig_filename}' ({img_id}) to COCO export: {match_res.message}"
                )

            coco_entry = parsed_coco[match_res.matched_target]
            img_w = coco_entry["image_info"]["width"]
            img_h = coco_entry["image_info"]["height"]
            mines = coco_entry["mine_damage"]

            # Source photo
            src_photo = photos_p / orig_filename
            if not src_photo.is_file():
                raise FileNotFoundError(f"Source photograph missing: {src_photo}")

            if is_neg != (rev_dec == ReviewerDecision.CONFIRMED_NEGATIVE.value):
                raise ValueError(f"Conflicting review and verified-negative flag: {img_id}")
            if is_neg and mines:
                raise ValueError(f"Verified-negative image still has mine polygons: {img_id}")
            if not is_neg and (rev_dec != ReviewerDecision.CONFIRMED_MINE.value or not mines):
                raise ValueError(f"Missing positive review or mine polygons: {img_id}")
            with Image.open(src_photo) as photo:
                if ImageOps.exif_transpose(photo).size != (img_w, img_h):
                    raise ValueError(f"COCO coordinates do not match the oriented photo: {img_id}")

            dst_photo = out_p / "images" / split_name / orig_filename
            shutil.copy2(src_photo, dst_photo)

            # Destination label
            stem = Path(orig_filename).stem
            label_file = out_p / "labels" / split_name / f"{stem}.txt"

            # Check if verified negative
            if is_neg:
                # Verified negative -> empty label file
                label_file.write_text("", encoding="utf-8")
                stats[split_name]["verified_negative_images"] += 1
            else:
                # Positive image with mine polygons
                lines: List[str] = []
                for poly in mines:
                    if not poly.points or len(poly.points) < 3:
                        continue
                    coords: List[str] = []
                    for px, py in poly.points:
                        norm_x = min(max(float(px) / img_w, 0.0), 1.0)
                        norm_y = min(max(float(py) / img_h, 0.0), 1.0)
                        coords.append(f"{norm_x:.6f} {norm_y:.6f}")
                    lines.append(f"{class_id} {' '.join(coords)}")

                label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                stats[split_name]["positive_images"] += 1
                stats[split_name]["total_polygons"] += len(lines)

            stats[split_name]["total_images"] += 1
            stats[split_name]["image_ids"].append(img_id)
            stats[split_name]["filenames"].append(orig_filename)

    # Generate dataset YAML
    yaml_p.parent.mkdir(parents=True, exist_ok=True)
    yaml_data = {
        "path": str(out_p.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            class_id: class_name,
        },
    }

    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write("# PhytoTrace AI — Phase 3 YOLO Instance Segmentation Dataset\n")
        f.write("# Generated automatically by export_yolo_segmentation_dataset\n")
        f.write("# Held-out test partition is strictly excluded and frozen.\n\n")
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    # Perform integrity verification
    _verify_dataset_integrity(out_p, stats)

    logger.info(
        "Successfully exported YOLO dataset: %d train images (%d pos, %d neg), %d val images.",
        stats["train"]["total_images"],
        stats["train"]["positive_images"],
        stats["train"]["verified_negative_images"],
        stats["val"]["total_images"],
    )

    return stats


def _verify_dataset_integrity(dataset_dir: Path, stats: Dict[str, Any]) -> None:
    """Validate exported YOLO segmentation dataset integrity and coordinate boundaries."""
    for split_name in ("train", "val"):
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name

        images = list(img_dir.glob("*"))
        labels = list(lbl_dir.glob("*.txt"))

        expected_count = stats[split_name]["total_images"]
        if len(images) != expected_count:
            raise ValueError(
                f"Mismatch in {split_name} images count: expected {expected_count}, found {len(images)}"
            )
        if len(labels) != expected_count:
            raise ValueError(
                f"Mismatch in {split_name} labels count: expected {expected_count}, found {len(labels)}"
            )

        # Validate label contents and coordinates [0, 1]
        for lbl_file in labels:
            content = lbl_file.read_text(encoding="utf-8").strip()
            if not content:
                continue  # Valid empty file for negative sample
            for line_idx, line in enumerate(content.splitlines()):
                tokens = line.strip().split()
                if len(tokens) < 7:  # class_id + at least 3 points (6 coords)
                    raise ValueError(
                        f"Degenerate polygon in {lbl_file.name}:{line_idx + 1}: only {len(tokens)} tokens"
                    )
                # Verify class index
                _cls = int(tokens[0])
                coords = [float(t) for t in tokens[1:]]
                if len(coords) % 2 != 0:
                    raise ValueError(
                        f"Odd number of coordinates in {lbl_file.name}:{line_idx + 1}"
                    )
                for c in coords:
                    if c < 0.0 or c > 1.0:
                        raise ValueError(
                            f"Coordinate out of range [0, 1] in {lbl_file.name}:{line_idx + 1}: {c}"
                        )

