"""Unit tests for group-aware dataset splitting and leakage prevention."""

import pandas as pd
import pytest

from phytotrace.data.split import make_group_splits, verify_no_split_leakage


@pytest.fixture
def multi_group_df():
    """Create a DataFrame simulating multiple farms, plants, and leaves."""
    rows = []
    # 3 distinct farms, 6 plants, 18 leaves
    for farm_idx in range(3):
        farm_id = f"FARM_{farm_idx:02d}"
        for plant_idx in range(2):
            plant_id = f"P_{farm_idx}_{plant_idx}"
            for leaf_idx in range(3):
                img_id = f"IMG_{farm_idx}_{plant_idx}_{leaf_idx}"
                rows.append({
                    "image_id": img_id,
                    "farm_id": farm_id,
                    "plant_id": plant_id,
                    "leaf_id": f"L_{leaf_idx}",
                })
    return pd.DataFrame(rows)


def test_group_aware_splitting_proportions(multi_group_df):
    splits = make_group_splits(
        df=multi_group_df,
        group_col="farm_id",
        fallback_group_col="plant_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    total_samples = len(multi_group_df)
    train_count = len(splits["train"])
    val_count = len(splits["val"])
    test_count = len(splits["test"])

    assert train_count + val_count + test_count == total_samples
    assert train_count > 0
    assert val_count > 0
    assert test_count > 0


def test_zero_leakage_across_partitions(multi_group_df):
    splits = make_group_splits(
        df=multi_group_df,
        group_col="farm_id",
        fallback_group_col="plant_id",
        seed=42,
    )
    # Check image ID disjointness
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Check strict group non-leakage
    no_leakage, errors = verify_no_split_leakage(splits, multi_group_df, ["farm_id", "plant_id"])
    assert no_leakage, f"Data leakage detected: {errors}"


def test_reproducibility_with_seed(multi_group_df):
    s1 = make_group_splits(multi_group_df, seed=123)
    s2 = make_group_splits(multi_group_df, seed=123)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]

