"""Unit tests for damage percentage calculation and severity assessment."""

import numpy as np
import pytest

from phytotrace.inference.severity import compute_damage_severity


def test_empty_mask_handling():
    leaf_mask = np.zeros((100, 100), dtype=np.uint8)
    mine_mask = np.zeros((100, 100), dtype=np.uint8)

    res = compute_damage_severity(leaf_mask, mine_mask, min_leaf_pixels=500)
    assert not res["is_valid"]
    assert res["damage_percentage"] is None


def test_accurate_damage_percentage():
    # 1000 leaf pixels
    leaf_mask = np.zeros((100, 100), dtype=np.uint8)
    leaf_mask[0:10, 0:100] = 255  # 10 * 100 = 1000 pixels

    # 200 mined pixels strictly inside the leaf
    mine_mask = np.zeros((100, 100), dtype=np.uint8)
    mine_mask[0:10, 0:20] = 255   # 10 * 20 = 200 pixels

    res = compute_damage_severity(leaf_mask, mine_mask, min_leaf_pixels=500)
    assert res["is_valid"]
    assert res["leaf_pixels"] == 1000
    assert res["mined_pixels"] == 200
    assert res["damage_percentage"] == 20.0
    assert res["severity_tier"] == "severe_infestation"


def test_mine_pixels_outside_leaf_are_excluded():
    # 1000 leaf pixels
    leaf_mask = np.zeros((100, 100), dtype=np.uint8)
    leaf_mask[0:10, 0:100] = 255

    # 500 mined pixels, but 400 are outside the leaf in background
    mine_mask = np.zeros((100, 100), dtype=np.uint8)
    mine_mask[0:10, 0:10] = 255   # 100 pixels inside leaf
    mine_mask[50:90, 0:10] = 255  # 400 pixels outside leaf

    res = compute_damage_severity(leaf_mask, mine_mask, min_leaf_pixels=500)
    assert res["is_valid"]
    assert res["mined_pixels"] == 100
    assert res["damage_percentage"] == 10.0


def test_overlapping_mine_masks_union():
    # Test that overlapping mine regions do not double count
    leaf_mask = np.full((100, 100), 255, dtype=np.uint8)  # 10,000 leaf pixels

    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:20, 0:20] = 255  # 400 px

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[10:30, 10:30] = 255  # 400 px, overlaps with mask1 by 10x10=100 px

    # Combined union mask
    union_mask = np.bitwise_or(mask1, mask2)
    expected_mined_pixels = 400 + 400 - 100  # 700 px

    res = compute_damage_severity(leaf_mask, union_mask, min_leaf_pixels=500)
    assert res["mined_pixels"] == expected_mined_pixels
    assert res["damage_percentage"] == 7.0

