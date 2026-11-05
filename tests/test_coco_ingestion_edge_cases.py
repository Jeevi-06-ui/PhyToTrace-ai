"""Tests for CVAT COCO segmentation ingestion, edge cases, negative verification, and discrepancy flagging."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from phytotrace.segmentation.io import (
    MatchResolutionStatus,
    MatchResult,
    find_coco_annotation_entry,
    parse_coco_segmentation_json,
    polygons_to_binary_mask,
    resolve_filename_match,
)
from phytotrace.segmentation.schemas import (
    AnnotationStatus,
    ImageSegmentationCatalogRecord,
    PolygonAnnotation,
    ReviewerDecision,
    SegmentationClass,
)
from phytotrace.segmentation.validators import validate_mine_within_leaf_mask


@pytest.fixture
def sample_coco_data(tmp_path: Path):
    """Create a realistic sample COCO JSON with diverse annotation scenarios."""
    coco = {
        "images": [
            {"id": 1, "file_name": "leaf_with_mines.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "healthy_leaf_neg.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "unannotated_img.jpg", "width": 100, "height": 100},
            {"id": 4, "file_name": "discrepancy_miner.jpg", "width": 100, "height": 100},
            {"id": 5, "file_name": "containment_viol.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "leaf", "supercategory": ""},
            {"id": 2, "name": "mine_damage", "supercategory": ""},
        ],
        "annotations": [
            # Image 1: Leaf + 2 distinct mine damage regions
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                "area": 6400.0,
                "bbox": [10, 10, 80, 80],
            },
            {
                "id": 102,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[20, 20, 40, 20, 40, 30, 20, 30]],
                "area": 200.0,
                "bbox": [20, 20, 20, 10],
            },
            {
                "id": 103,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[50, 50, 70, 50, 70, 70, 50, 70]],
                "area": 400.0,
                "bbox": [50, 50, 20, 20],
            },
            # Image 2: Healthy Leaf (Leaf polygon only, 0 mine damage)
            {
                "id": 201,
                "image_id": 2,
                "category_id": 1,
                "segmentation": [[15, 15, 85, 15, 85, 85, 15, 85]],
                "area": 4900.0,
                "bbox": [15, 15, 70, 70],
            },
            # Image 4: Leaf polygon present, but 0 mines (Discrepancy test if marked as leaf_miner in metadata)
            {
                "id": 401,
                "image_id": 4,
                "category_id": 1,
                "segmentation": [[10, 10, 80, 10, 80, 80, 10, 80]],
                "area": 4900.0,
                "bbox": [10, 10, 70, 70],
            },
            # Image 5: Leaf + mine outside leaf (Containment violation: leaf 20..80, mine 0..15)
            {
                "id": 501,
                "image_id": 5,
                "category_id": 1,
                "segmentation": [[20, 20, 80, 20, 80, 80, 20, 80]],
                "area": 3600.0,
                "bbox": [20, 20, 60, 60],
            },
            {
                "id": 502,
                "image_id": 5,
                "category_id": 2,
                "segmentation": [[2, 2, 15, 2, 15, 15, 2, 15]],  # Completely outside leaf!
                "area": 169.0,
                "bbox": [2, 2, 13, 13],
            },
        ],
    }

    coco_file = tmp_path / "test_coco_instances.json"
    with open(coco_file, "w") as f:
        json.dump(coco, f, indent=2)
    return coco_file


def test_coco_parser_multi_mine_and_negative(sample_coco_data: Path):
    """Verify that COCO parser extracts multiple mine regions and tracks negative leaves."""
    results = parse_coco_segmentation_json(sample_coco_data)

    assert "leaf_with_mines.jpg" in results
    assert len(results["leaf_with_mines.jpg"]["leaf"]) == 1
    assert len(results["leaf_with_mines.jpg"]["mine_damage"]) == 2

    # Verify healthy negative leaf has leaf polygon but zero mine polygons
    assert "healthy_leaf_neg.jpg" in results
    assert len(results["healthy_leaf_neg.jpg"]["leaf"]) == 1
    assert len(results["healthy_leaf_neg.jpg"]["mine_damage"]) == 0

    # Verify unannotated image exists in COCO images list but has 0 polygons
    assert "unannotated_img.jpg" in results
    assert len(results["unannotated_img.jpg"]["leaf"]) == 0
    assert len(results["unannotated_img.jpg"]["mine_damage"]) == 0


def test_polygons_to_mask_empty_negative():
    """Verify that empty polygon list rasterizes to an exact zero-filled binary mask."""
    empty_mask = polygons_to_binary_mask([], height=100, width=100)
    assert empty_mask.shape == (100, 100)
    assert np.all(empty_mask == 0)
    assert np.count_nonzero(empty_mask) == 0


def test_polygons_to_mask_multi_region_union():
    """Verify that multiple mine polygons rasterize into a unified binary mask."""
    p1 = PolygonAnnotation(points=[(10, 10), (30, 10), (30, 30), (10, 30)], label=SegmentationClass.MINE_DAMAGE)
    p2 = PolygonAnnotation(points=[(50, 50), (70, 50), (70, 70), (50, 70)], label=SegmentationClass.MINE_DAMAGE)

    mask = polygons_to_binary_mask([p1, p2], height=100, width=100)
    assert mask.shape == (100, 100)
    assert mask[20, 20] == 255
    assert mask[60, 60] == 255
    assert mask[40, 40] == 0  # Gap between polygons


def test_containment_violation_detection():
    """Verify that mines placed outside leaf margins are flagged by the containment validator."""
    leaf_p = PolygonAnnotation(points=[(20, 20), (80, 20), (80, 80), (20, 80)], label=SegmentationClass.LEAF)
    outside_mine = PolygonAnnotation(points=[(2, 2), (15, 2), (15, 15), (2, 15)], label=SegmentationClass.MINE_DAMAGE)

    leaf_mask = polygons_to_binary_mask([leaf_p], 100, 100)
    mine_mask = polygons_to_binary_mask([outside_mine], 100, 100)

    is_valid, out_ratio, errors = validate_mine_within_leaf_mask(mine_mask, leaf_mask)
    assert not is_valid
    assert out_ratio > 0.90  # ~100% of the mine is outside the leaf
    assert len(errors) > 0


def test_discrepancy_flagging_logic():
    """Verify that metadata leaf_miner_damage with 0 mine polygons is NOT silently treated as negative."""
    # Simulation of metadata row marked as leaf_miner_damage but 0 mine polygons in CVAT
    record = ImageSegmentationCatalogRecord(
        image_id="IMG_TEST_DISC",
        original_filename="discrepancy_miner.jpg",
        plant_id="PL_TEST",
        farm_id="BLOCK_TEST",
        visible_damage_type="leaf_miner_damage",
        leaf_mask_status=AnnotationStatus.COMPLETED,
        mine_annotation_status=AnnotationStatus.FLAGGED,
        ready_for_training=False,
        is_verified_negative=False,
        notes="DISCREPANCY: Labeled leaf_miner_damage, but 0 mines drawn.",
    )

    d = record.to_dict()
    assert d["mine_annotation_status"] == "flagged"
    assert d["ready_for_training"] is False
    assert d["is_verified_negative"] is False
    assert "DISCREPANCY" in d["notes"]


def test_find_coco_annotation_entry_case_insensitive_and_stem():
    """Regression test: verify case-insensitive and extension-tolerant filename matching.
    
    Protects against dropping images when metadata has uppercase .JPG and CVAT exports lowercase .jpg.
    """
    mock_parsed = {
        "BLK_3_1002_PL009_NH.jpg": {"present_in_coco": True, "leaf": [1], "mine_damage": [1, 2]},
        "IMG_2162.jpg": {"present_in_coco": True, "leaf": [1], "mine_damage": []},
    }

    # Uppercase .JPG query should resolve to lowercase .jpg entry
    res1, match1 = find_coco_annotation_entry(mock_parsed, "BLK_3_1002_PL009_NH.JPG")
    assert res1 is not None
    assert match1.status == MatchResolutionStatus.NORMALIZED
    assert match1.matched_target == "BLK_3_1002_PL009_NH.jpg"
    assert res1["present_in_coco"] is True
    assert len(res1["mine_damage"]) == 2

    # Different case query
    res2, match2 = find_coco_annotation_entry(mock_parsed, "img_2162.JPG")
    assert res2 is not None
    assert match2.status == MatchResolutionStatus.NORMALIZED
    assert res2["present_in_coco"] is True

    # Path object / with directory prefix
    res3, match3 = find_coco_annotation_entry(mock_parsed, "data/raw/original_photos/BLK_3_1002_PL009_NH.JPG")
    assert res3 is not None
    assert match3.status == MatchResolutionStatus.NORMALIZED

    # Non-existent file
    res4, match4 = find_coco_annotation_entry(mock_parsed, "non_existent.jpg")
    assert res4 is None
    assert match4.status == MatchResolutionStatus.UNMATCHED


def test_resolve_filename_match_all_conditions():
    """Comprehensive regression tests covering all 8 specified matching conditions:
    - .jpg versus .JPG
    - .jpeg versus .JPEG
    - exact filenames
    - mixed-case filenames
    - unknown filenames
    - case-insensitive collisions (ambiguous rejection)
    - extension collisions
    - empty pool
    """
    pool = [
        "leaf_01.jpg",
        "leaf_02.jpeg",
        "BLK_3_PL001.JPG",
        "Sample_Tomato.PNG",
    ]

    # 1. Exact filename match
    r_exact = resolve_filename_match("leaf_01.jpg", pool)
    assert r_exact.status == MatchResolutionStatus.EXACT
    assert r_exact.matched_target == "leaf_01.jpg"

    # 2. .jpg versus .JPG
    r_jpg = resolve_filename_match("leaf_01.JPG", pool)
    assert r_jpg.status == MatchResolutionStatus.NORMALIZED
    assert r_jpg.matched_target == "leaf_01.jpg"

    # 3. .jpeg versus .JPEG
    r_jpeg = resolve_filename_match("leaf_02.JPEG", pool)
    assert r_jpeg.status == MatchResolutionStatus.NORMALIZED
    assert r_jpeg.matched_target == "leaf_02.jpeg"

    # 4. Mixed-case filename
    r_mixed = resolve_filename_match("blk_3_pl001.jpg", pool)
    assert r_mixed.status == MatchResolutionStatus.NORMALIZED
    assert r_mixed.matched_target == "BLK_3_PL001.JPG"

    # 5. Unknown filename
    r_unknown = resolve_filename_match("unregistered_photo.jpg", pool)
    assert r_unknown.status == MatchResolutionStatus.UNMATCHED
    assert r_unknown.matched_target is None

    # 6. Case-insensitive collision rejection (ambiguous)
    collision_pool = ["Tomato_Leaf.jpg", "TOMATO_LEAF.jpg"]
    r_collision = resolve_filename_match("tomato_leaf.jpg", collision_pool)
    assert r_collision.status == MatchResolutionStatus.AMBIGUOUS
    assert r_collision.matched_target is None
    assert len(r_collision.candidates) == 2

    # 7. Extension stem-equivalence collision rejection
    stem_collision_pool = ["plant_01.jpg", "plant_01.png"]
    r_stem_collision = resolve_filename_match("plant_01.jpeg", stem_collision_pool)
    assert r_stem_collision.status == MatchResolutionStatus.AMBIGUOUS
    assert r_stem_collision.matched_target is None
    assert len(r_stem_collision.candidates) == 2

    # 8. Empty pool
    r_empty = resolve_filename_match("leaf_01.jpg", [])
    assert r_empty.status == MatchResolutionStatus.UNMATCHED


def test_coco_category_id_mapping(tmp_path: Path):
    """Verify that COCO parser correctly maps non-standard or swapped category IDs."""
    custom_coco = {
        "images": [
            {"id": 10, "file_name": "test_leaf.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 42, "name": "leaf_boundary", "supercategory": ""},
            {"id": 99, "name": "leaf_miner_damage_lesion", "supercategory": ""},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 42,
                "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                "area": 6400.0,
                "bbox": [10, 10, 80, 80],
            },
            {
                "id": 2,
                "image_id": 10,
                "category_id": 99,
                "segmentation": [[20, 20, 30, 20, 30, 30, 20, 30]],
                "area": 100.0,
                "bbox": [20, 20, 10, 10],
            },
        ],
    }
    p = tmp_path / "custom_cat_coco.json"
    with open(p, "w") as f:
        json.dump(custom_coco, f)

    parsed = parse_coco_segmentation_json(p)
    assert "test_leaf.jpg" in parsed
    assert len(parsed["test_leaf.jpg"]["leaf"]) == 1
    assert parsed["test_leaf.jpg"]["leaf"][0].label == SegmentationClass.LEAF
    assert len(parsed["test_leaf.jpg"]["mine_damage"]) == 1
    assert parsed["test_leaf.jpg"]["mine_damage"][0].label == SegmentationClass.MINE_DAMAGE


def test_compound_leaflets_multi_leaf_polygons():
    """Verify that multiple leaf polygons (e.g. compound leaflets in IMG_0004) rasterize into 1 mask."""
    p_leaf1 = PolygonAnnotation(points=[(10, 10), (40, 10), (40, 40), (10, 40)], label=SegmentationClass.LEAF)
    p_leaf2 = PolygonAnnotation(points=[(50, 50), (80, 50), (80, 80), (50, 80)], label=SegmentationClass.LEAF)

    mask = polygons_to_binary_mask([p_leaf1, p_leaf2], height=100, width=100)
    assert mask.shape == (100, 100)
    assert mask[20, 20] == 255
    assert mask[65, 65] == 255
    assert mask[45, 45] == 0


def test_biological_conflict_flagging():
    """Regression test: verify that healthy leaves with mine polygons are flagged and not ready."""
    record = ImageSegmentationCatalogRecord(
        image_id="IMG_HEALTHY_CONFLICT",
        original_filename="healthy_leaf.jpg",
        plant_id="PL_001",
        farm_id="BLOCK_1",
        visible_damage_type="healthy",
        leaf_mask_status=AnnotationStatus.COMPLETED,
        mine_annotation_status=AnnotationStatus.FLAGGED,
        num_damage_regions=31,
        ready_for_training=False,
        is_verified_negative=False,
        notes="CONFLICT: Healthy leaf has 31 mine polygons! Delete mines in CVAT to verify negative.",
    )
    d = record.to_dict()
    assert d["mine_annotation_status"] == "flagged"
    assert d["ready_for_training"] is False
    assert d["is_verified_negative"] is False
    assert "CONFLICT" in d["notes"]


def test_reviewer_decision_schema_and_states():
    """Verify that all 4 formal reviewer decision states are supported."""
    assert ReviewerDecision.CONFIRMED_NEGATIVE.value == "confirmed_negative"
    assert ReviewerDecision.CONFIRMED_MINE.value == "confirmed_mine"
    assert ReviewerDecision.NEEDS_CORRECTION.value == "needs_correction"
    assert ReviewerDecision.UNCERTAIN_EXCLUDE.value == "uncertain_exclude"
    assert ReviewerDecision.PENDING_REVIEW.value == "pending_review"


def test_contradiction_confirmed_negative_with_positive_mine_polygons_rejected():
    """Regression test: verify that CONFIRMED_NEGATIVE is rejected when COCO has >=1 mine polygons.
    
    Protects against data integrity bug where positive polygons were erroneously overridden
    with empty masks, prematurely unlocking training gates.
    """
    record = ImageSegmentationCatalogRecord(
        image_id="IMG_CONTRADICTORY",
        original_filename="BLK_3_1002_PL010_H.JPG",
        plant_id="PL_010",
        farm_id="BLOCK_3",
        visible_damage_type="healthy",
        leaf_mask_status=AnnotationStatus.COMPLETED,
        mine_annotation_status=AnnotationStatus.FLAGGED,
        annotation_reviewer="jeevithp",
        reviewer_decision=ReviewerDecision.CONFIRMED_NEGATIVE,
        reviewer_notes="Reviewer claimed negative, but COCO contains 30 mine polygons.",
        num_damage_regions=30,
        ready_for_training=False,
        is_verified_negative=False,
        notes="CONTRADICTION: Reviewer decision is CONFIRMED_NEGATIVE, but COCO contains 30 mine_damage polygons!",
    )
    d = record.to_dict()
    assert d["reviewer_decision"] == "confirmed_negative"
    assert d["mine_annotation_status"] == "flagged"
    assert d["ready_for_training"] is False
    assert d["is_verified_negative"] is False
    assert "CONTRADICTION" in d["notes"]
    assert d["num_damage_regions"] == 30


def test_consistent_confirmed_negative_with_zero_mine_polygons_accepted():
    """Verify that CONFIRMED_NEGATIVE is accepted ONLY when COCO has exactly 0 mine polygons."""
    record = ImageSegmentationCatalogRecord(
        image_id="IMG_CONSISTENT_NEG",
        original_filename="BLK_3_1002_PL010_H.JPG",
        plant_id="PL_010",
        farm_id="BLOCK_3",
        visible_damage_type="healthy",
        leaf_mask_status=AnnotationStatus.COMPLETED,
        mine_annotation_status=AnnotationStatus.COMPLETED,
        annotation_reviewer="jeevithp",
        reviewer_decision=ReviewerDecision.CONFIRMED_NEGATIVE,
        reviewer_notes="Verified 0 mines in CVAT; true negative leaf.",
        num_damage_regions=0,
        ready_for_training=True,
        is_verified_negative=True,
        leaf_mask_path="data/segmentation/masks/leaf/IMG_CONSISTENT_NEG_leaf_mask.png",
        mine_mask_path="data/segmentation/masks/mine/IMG_CONSISTENT_NEG_mine_mask.png",
        notes="REVIEW [jeevithp]: CONFIRMED_NEGATIVE — Verified 0 mines in COCO",
    )
    d = record.to_dict()
    assert d["reviewer_decision"] == "confirmed_negative"
    assert d["mine_annotation_status"] == "completed"
    assert d["ready_for_training"] is True
    assert d["is_verified_negative"] is True
    assert d["num_damage_regions"] == 0



def test_reviewer_decision_needs_correction_record():
    """Verify that NEEDS_CORRECTION decision correctly locks training readiness."""
    record = ImageSegmentationCatalogRecord(
        image_id="IMG_NEEDS_CORR",
        original_filename="BLK_3_1002_PL010_H.JPG",
        plant_id="PL_010",
        farm_id="BLOCK_3",
        visible_damage_type="healthy",
        leaf_mask_status=AnnotationStatus.COMPLETED,
        mine_annotation_status=AnnotationStatus.FLAGGED,
        annotation_reviewer="expert_pathologist_01",
        reviewer_decision=ReviewerDecision.NEEDS_CORRECTION,
        reviewer_notes="Spurious mine polygons must be deleted in CVAT.",
        ready_for_training=False,
        is_verified_negative=False,
        notes="REVIEW [expert_pathologist_01]: NEEDS_CORRECTION in CVAT",
    )
    d = record.to_dict()
    assert d["reviewer_decision"] == "needs_correction"
    assert d["ready_for_training"] is False
    assert d["is_verified_negative"] is False
    assert "NEEDS_CORRECTION" in d["notes"]




