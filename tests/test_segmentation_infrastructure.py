"""Unit tests for Phase 3 segmentation utilities, rasterization, containment checks, and COCO parsing."""

import json
from pathlib import Path
import numpy as np
import pytest

from phytotrace.segmentation.io import (
    binary_mask_to_polygons,
    parse_coco_segmentation_json,
    polygons_to_binary_mask,
)
from phytotrace.segmentation.schemas import (
    AnnotationStatus,
    ImageSegmentationCatalogRecord,
    PolygonAnnotation,
    SegmentationClass,
)
from phytotrace.segmentation.validators import (
    validate_mine_within_leaf_mask,
    validate_polygon_geometry,
)
from phytotrace.segmentation.visualization import render_segmentation_overlay


def test_polygon_geometry_validation():
    """Verify polygon geometric bounds and vertex requirements."""
    # Valid polygon
    valid_poly = PolygonAnnotation(
        points=[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        label=SegmentationClass.LEAF,
        area_pixels=1600.0,
    )
    is_valid, errors = validate_polygon_geometry(valid_poly, image_width=100, image_height=100)
    assert is_valid
    assert len(errors) == 0

    # Polygon outside image bounds
    oob_poly = PolygonAnnotation(
        points=[(10.0, 10.0), (150.0, 10.0), (150.0, 50.0)],
        label=SegmentationClass.LEAF,
    )
    is_valid, errors = validate_polygon_geometry(oob_poly, image_width=100, image_height=100)
    assert not is_valid
    assert any("outside image bounds" in e for e in errors)

    # Degenerate polygon (< 3 points)
    degen_poly = PolygonAnnotation(
        points=[(10.0, 10.0), (20.0, 20.0)],
        label=SegmentationClass.MINE_DAMAGE,
    )
    is_valid, errors = validate_polygon_geometry(degen_poly, image_width=100, image_height=100)
    assert not is_valid
    assert any("fewer than 3 vertices" in e for e in errors)


def test_polygons_to_binary_mask_and_inverse():
    """Verify rasterization from polygon to binary mask and back to contour."""
    h, w = 100, 100
    poly = PolygonAnnotation(
        points=[(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)],
        label=SegmentationClass.LEAF,
        area_pixels=3600.0,
    )
    mask = polygons_to_binary_mask([poly], height=h, width=w)
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    assert mask[50, 50] == 255  # Inside
    assert mask[5, 5] == 0      # Outside
    assert 3400 <= np.sum(mask > 0) <= 3800

    # Inverse back to polygon
    extracted_polys = binary_mask_to_polygons(mask, min_area=100.0)
    assert len(extracted_polys) == 1
    assert len(extracted_polys[0]) >= 4


def test_mine_within_leaf_containment_validation():
    """Verify biological rule that mine damage must reside inside the leaf boundary."""
    h, w = 100, 100
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[20:80, 20:80] = 255  # Leaf box

    # Mine completely inside leaf
    inside_mine = np.zeros((h, w), dtype=np.uint8)
    inside_mine[30:40, 30:40] = 255
    is_valid, out_ratio, errors = validate_mine_within_leaf_mask(inside_mine, leaf_mask)
    assert is_valid
    assert out_ratio == 0.0
    assert len(errors) == 0

    # Mine spilling heavily outside leaf (> 5%)
    spilling_mine = np.zeros((h, w), dtype=np.uint8)
    spilling_mine[10:30, 10:30] = 255  # Partly outside (10:20 is outside)
    is_valid, out_ratio, errors = validate_mine_within_leaf_mask(spilling_mine, leaf_mask, max_allowed_outside_ratio=0.05)
    assert not is_valid
    assert out_ratio > 0.05
    assert len(errors) > 0


def test_segmentation_overlay_rendering():
    """Verify semi-transparent overlay rendering on an RGB image."""
    h, w = 100, 100
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[20:80, 20:80] = 255
    mine_mask = np.zeros((h, w), dtype=np.uint8)
    mine_mask[40:50, 40:50] = 255

    overlay = render_segmentation_overlay(img, leaf_mask, mine_mask)
    assert overlay.shape == (h, w, 3)
    assert not np.array_equal(overlay, img)


def test_coco_segmentation_json_parsing(tmp_path):
    """Verify parsing of CVAT-exported COCO instances_default.json."""
    coco_data = {
        "categories": [
            {"id": 1, "name": "leaf"},
            {"id": 2, "name": "mine_damage"},
        ],
        "images": [
            {"id": 101, "file_name": "test_photo_01.jpg", "width": 200, "height": 200},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 101,
                "category_id": 1,
                "segmentation": [[20, 20, 180, 20, 180, 180, 20, 180]],
                "area": 25600.0,
                "bbox": [20, 20, 160, 160],
            },
            {
                "id": 2,
                "image_id": 101,
                "category_id": 2,
                "segmentation": [[40, 40, 60, 40, 60, 60, 40, 60]],
                "area": 400.0,
                "bbox": [40, 40, 20, 20],
            },
        ],
    }
    json_path = tmp_path / "instances_default.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    parsed = parse_coco_segmentation_json(json_path)
    assert "test_photo_01.jpg" in parsed
    assert len(parsed["test_photo_01.jpg"]["leaf"]) == 1
    assert len(parsed["test_photo_01.jpg"]["mine_damage"]) == 1
    assert parsed["test_photo_01.jpg"]["leaf"][0].label == SegmentationClass.LEAF
    assert parsed["test_photo_01.jpg"]["mine_damage"][0].label == SegmentationClass.MINE_DAMAGE

