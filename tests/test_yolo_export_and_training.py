"""Tests for YOLO segmentation dataset export, data integrity, and training pipeline."""

import json
from pathlib import Path
import pytest
import yaml

from phytotrace.segmentation.yolo_exporter import export_yolo_segmentation_dataset
from phytotrace.training.train_mine_segmenter import train_mine_segmenter_pipeline
from phytotrace.utils.paths import resolve_path


def test_yolo_dataset_export_integrity(tmp_path):
    """Verify that YOLO dataset export produces exact expected file structure and zero leakage."""
    out_dir = tmp_path / "yolo_segmentation"
    yaml_out = tmp_path / "annotations" / "yolo_dataset.yaml"

    stats = export_yolo_segmentation_dataset(
        catalog_path=resolve_path("data/segmentation/catalog.csv"),
        coco_json_path=resolve_path("data/segmentation/annotations/instances_default.json"),
        splits_json_path=resolve_path("data/splits/splits.json"),
        photos_dir=resolve_path("data/raw/original_photos"),
        output_dir=out_dir,
        yaml_output_path=yaml_out,
    )

    # 1. Verification of train split counts
    assert stats["train"]["total_images"] == 10
    assert stats["train"]["positive_images"] == 7
    assert stats["train"]["verified_negative_images"] == 3
    assert stats["train"]["total_polygons"] == 82

    # 2. Verification of val split counts
    assert stats["val"]["total_images"] == 3
    assert stats["val"]["positive_images"] == 3
    assert stats["val"]["verified_negative_images"] == 0
    assert stats["val"]["total_polygons"] == 15

    # 3. Verification of file system existence
    train_imgs = list((out_dir / "images" / "train").glob("*"))
    train_lbls = list((out_dir / "labels" / "train").glob("*.txt"))
    val_imgs = list((out_dir / "images" / "val").glob("*"))
    val_lbls = list((out_dir / "labels" / "val").glob("*.txt"))

    assert len(train_imgs) == 10
    assert len(train_lbls) == 10
    assert len(val_imgs) == 3
    assert len(val_lbls) == 3

    # 4. Verification of verified negative empty label files
    expected_empty = {"BLK_3_1002_PL010_H.txt", "BLK_3_1002_PL021_H.txt", "BLK_3_1002_PL029_H.txt"}
    actual_empty = {f.name for f in train_lbls if f.stat().st_size == 0}
    assert actual_empty == expected_empty

    # 5. Verification of positive label formatting and normalized coords [0, 1]
    for lbl in train_lbls:
        if lbl.name in expected_empty:
            continue
        content = lbl.read_text(encoding="utf-8").strip()
        assert len(content) > 0
        for line in content.splitlines():
            tokens = line.strip().split()
            assert tokens[0] == "0"  # class_id
            coords = [float(c) for c in tokens[1:]]
            assert len(coords) >= 6
            assert len(coords) % 2 == 0
            for c in coords:
                assert 0.0 <= c <= 1.0

    # 6. Verification of test isolation: zero test images in dataset
    splits = json.load(open(resolve_path("data/splits/splits.json")))
    test_ids = set(splits.get("test", []))
    assert not any(img_id in test_ids for img_id in stats["train"]["image_ids"])
    assert not any(img_id in test_ids for img_id in stats["val"]["image_ids"])

    # 7. Verification of unready image exclusion (e.g. IMG_0020 with needs_correction)
    assert "IMG_0020" not in stats["train"]["image_ids"]
    assert "IMG_0020" not in stats["val"]["image_ids"]

    # 8. Verification of dataset YAML file
    assert yaml_out.is_file()
    with open(yaml_out, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["names"] == {0: "mine_damage"}


def test_train_mine_segmenter_missing_yaml_safely_fails(tmp_path):
    """Ensure training pipeline reports DATA_NOT_READY when yaml is missing without crash."""
    non_existent_yaml = tmp_path / "missing.yaml"
    res = train_mine_segmenter_pipeline(
        data_yaml_path=str(non_existent_yaml),
        epochs=1,
    )
    assert res["status"] == "DATA_NOT_READY"
    assert "not found" in res["message"]

