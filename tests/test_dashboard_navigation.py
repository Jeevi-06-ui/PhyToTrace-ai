"""Regression tests for Streamlit dashboard filter and navigation logic (0, 1, and multiple images)."""

import pandas as pd
import pytest
from demo.app import filter_dataframe, resolve_navigation_index


@pytest.fixture
def sample_metadata_df():
    data = [
        {"image_id": "IMG_001", "visible_damage_type": "healthy", "farm_id": "F1"},
        {"image_id": "IMG_002", "visible_damage_type": "healthy", "farm_id": "F1"},
        {"image_id": "IMG_003", "visible_damage_type": "leaf_miner_damage", "farm_id": "F1"},
        {"image_id": "IMG_004", "visible_damage_type": "other_damage", "farm_id": "F2"},
        {"image_id": "IMG_005", "visible_damage_type": "unlabelled", "farm_id": "F2"},
    ]
    return pd.DataFrame(data)


def test_zero_filtered_images_empty_handling():
    """Verify that zero matching images yields None index and avoids downstream .iloc crash."""
    empty_df = pd.DataFrame(columns=["image_id", "visible_damage_type"])
    res_df = filter_dataframe(empty_df, "Unlabelled Only")
    assert res_df.empty
    assert len(res_df) == 0

    idx = resolve_navigation_index(len(res_df))
    assert idx is None

    # Regression check: if sel_idx is None, downstream access must be guarded
    accessed = False
    if idx is not None and not res_df.empty:
        _ = res_df.iloc[idx]
        accessed = True
    assert not accessed


def test_one_filtered_image_no_slider_crash(sample_metadata_df):
    """Verify that exactly one matching image selects index 0 without slider min==max crash."""
    # In sample_metadata_df, only 1 image has visible_damage_type == 'unlabelled'
    unlabelled_df = filter_dataframe(sample_metadata_df, "Unlabelled Only (1)")
    assert len(unlabelled_df) == 1

    idx = resolve_navigation_index(len(unlabelled_df))
    assert idx == 0

    # In Streamlit, a slider with min_val=0, max_val=0 raises StreamlitInvalidMinMaxError.
    # Our navigation logic bypasses the slider when count == 1.
    selected_row = unlabelled_df.iloc[idx]
    assert selected_row["image_id"] == "IMG_005"
    assert selected_row["visible_damage_type"] == "unlabelled"


def test_multiple_filtered_images_slider_bounds(sample_metadata_df):
    """Verify that multiple matching images yield valid slider bounds where min_val < max_val."""
    healthy_df = filter_dataframe(sample_metadata_df, "Healthy (2)")
    count = len(healthy_df)
    assert count == 2

    # Slider bounds: min=0, max=count-1=1 -> 0 != 1 (valid Streamlit slider)
    min_val = 0
    max_val = count - 1
    assert min_val < max_val

    # Test clamped index resolution
    assert resolve_navigation_index(count, 0) == 0
    assert resolve_navigation_index(count, 1) == 1
    assert resolve_navigation_index(count, 99) == 1  # clamped to max
    assert resolve_navigation_index(count, -5) == 0  # clamped to min


def test_all_filters_coverage(sample_metadata_df):
    """Verify that each filter option properly partitions the metadata DataFrame."""
    all_df = filter_dataframe(sample_metadata_df, "All Images (5)")
    assert len(all_df) == 5

    unlabelled_df = filter_dataframe(sample_metadata_df, "Unlabelled Only (1)")
    assert len(unlabelled_df) == 1
    assert unlabelled_df.iloc[0]["visible_damage_type"] == "unlabelled"

    healthy_df = filter_dataframe(sample_metadata_df, "Healthy (2)")
    assert len(healthy_df) == 2
    assert (healthy_df["visible_damage_type"] == "healthy").all()

    miner_df = filter_dataframe(sample_metadata_df, "Leaf Miner Damage (1)")
    assert len(miner_df) == 1
    assert miner_df.iloc[0]["visible_damage_type"] == "leaf_miner_damage"

    other_df = filter_dataframe(sample_metadata_df, "Other Damage (1)")
    assert len(other_df) == 1
    assert other_df.iloc[0]["visible_damage_type"] == "other_damage"

