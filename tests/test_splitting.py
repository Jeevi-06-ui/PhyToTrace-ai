"""Unit tests for group-aware dataset splitting and leakage prevention."""

import pandas as pd
import pytest

from phytotrace.data.split import make_group_splits, verify_no_split_leakage


@pytest.fixture
def multi_group_df():
    """Create a DataFrame simulating multiple farms, plants, and leaves with class labels."""
    rows = []
    classes = ["healthy", "leaf_miner_damage", "other_damage"]
    # 3 distinct farms, 6 plants, 18 leaves
    for farm_idx in range(3):
        farm_id = f"FARM_{farm_idx:02d}"
        for plant_idx in range(2):
            plant_id = f"P_{farm_idx}_{plant_idx}"
            cls = classes[(farm_idx + plant_idx) % 3]
            for leaf_idx in range(3):
                img_id = f"IMG_{farm_idx}_{plant_idx}_{leaf_idx}"
                rows.append({
                    "image_id": img_id,
                    "farm_id": farm_id,
                    "plant_id": plant_id,
                    "leaf_id": f"L_{leaf_idx}",
                    "visible_damage_type": cls,
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
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    no_leakage, errors = verify_no_split_leakage(splits, multi_group_df, ["farm_id", "plant_id"])
    assert no_leakage, f"Data leakage detected: {errors}"


def test_reproducibility_with_seed(multi_group_df):
    s1 = make_group_splits(multi_group_df, seed=123)
    s2 = make_group_splits(multi_group_df, seed=123)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]


# =========================================================================
# REQUIRED REGRESSION TESTS (URGENT SPLITTING AUDIT)
# =========================================================================

def test_regression_same_leaf_id_on_different_plants_no_false_leakage():
    """1. The same leaf_id (e.g. 'L01') on different physical plants must NOT trigger false leakage."""
    df = pd.DataFrame([
        {"image_id": "IMG_1", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_2", "farm_id": "F1", "plant_id": "P2", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_3", "farm_id": "F1", "plant_id": "P3", "leaf_id": "L01", "visible_damage_type": "healthy"},
    ])
    # Partition P1 in train, P2 in val, P3 in test
    manifest = {"train": ["IMG_1"], "val": ["IMG_2"], "test": ["IMG_3"]}
    no_leakage, errors = verify_no_split_leakage(manifest, df, group_cols=["leaf_id"], strict_classes=False)
    assert no_leakage, f"Expected no false leakage for repeated local leaf_id, got: {errors}"


def test_regression_same_physical_leaf_in_two_partitions_detected():
    """2. The same physical leaf occurring in two partitions must be detected as genuine leakage."""
    df = pd.DataFrame([
        {"image_id": "IMG_1a", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_1b", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_2", "farm_id": "F1", "plant_id": "P2", "leaf_id": "L01", "visible_damage_type": "healthy"},
    ])
    # Put two photos of the SAME physical leaf (P1_L01) in train and val
    manifest = {"train": ["IMG_1a"], "val": ["IMG_1b"], "test": ["IMG_2"]}
    no_leakage, errors = verify_no_split_leakage(manifest, df, strict_classes=False)
    assert not no_leakage
    assert any("Physical leaf leakage" in e for e in errors)


def test_regression_same_physical_plant_in_two_partitions_detected():
    """3. Images from one physical plant appearing in two partitions must be detected as leakage."""
    df = pd.DataFrame([
        {"image_id": "IMG_1_L1", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_1_L2", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L02", "visible_damage_type": "healthy"},
        {"image_id": "IMG_2", "farm_id": "F1", "plant_id": "P2", "leaf_id": "L01", "visible_damage_type": "healthy"},
    ])
    # Put Leaf 1 of Plant 1 in train, and Leaf 2 of Plant 1 in test
    manifest = {"train": ["IMG_1_L1"], "val": ["IMG_2"], "test": ["IMG_1_L2"]}
    no_leakage, errors = verify_no_split_leakage(manifest, df, strict_classes=False)
    assert not no_leakage
    assert any("Physical plant leakage" in e for e in errors)


def test_regression_duplicate_image_clusters_crossing_partitions_detected():
    """4. Duplicate image clusters crossing partitions must be detected."""
    df = pd.DataFrame([
        {"image_id": "IMG_A", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_B", "farm_id": "F2", "plant_id": "P2", "leaf_id": "L01", "visible_damage_type": "healthy"},
        {"image_id": "IMG_C", "farm_id": "F3", "plant_id": "P3", "leaf_id": "L01", "visible_damage_type": "healthy"},
    ])
    manifest = {"train": ["IMG_A"], "val": ["IMG_B"], "test": ["IMG_C"]}
    # Cluster says IMG_A and IMG_B are near duplicates
    no_leakage, errors = verify_no_split_leakage(
        manifest, df, duplicate_clusters=[["IMG_A", "IMG_B"]], strict_classes=False
    )
    assert not no_leakage
    assert any("Duplicate image cluster crossing partitions" in e for e in errors)


def test_regression_missing_group_ids_handled_safely():
    """5. Missing/blank group IDs must not cause crash or false matching."""
    df = pd.DataFrame([
        {"image_id": "IMG_1", "farm_id": "", "plant_id": "unknown", "leaf_id": "", "visible_damage_type": "healthy"},
        {"image_id": "IMG_2", "farm_id": "none", "plant_id": "", "leaf_id": "none", "visible_damage_type": "healthy"},
        {"image_id": "IMG_3", "farm_id": "F1", "plant_id": "P1", "leaf_id": "L01", "visible_damage_type": "healthy"},
    ])
    splits = make_group_splits(df, seed=42)
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 3
    no_leakage, errors = verify_no_split_leakage(splits, df, strict_classes=False)
    assert no_leakage, f"Expected no leakage for missing group IDs, got: {errors}"


def test_regression_extremely_imbalanced_group_sizes_detected():
    """6. Extremely imbalanced group sizes (e.g., 6/8/81) must be flagged as partition imbalance."""
    df = pd.DataFrame([{"image_id": f"IMG_{i}", "visible_damage_type": "healthy"} for i in range(95)])
    # Simulate the buggy 6 train / 8 val / 81 test split
    manifest = {
        "train": [f"IMG_{i}" for i in range(6)],
        "val": [f"IMG_{i}" for i in range(6, 14)],
        "test": [f"IMG_{i}" for i in range(14, 95)],
    }
    no_leakage, errors = verify_no_split_leakage(manifest, df, strict_classes=False)
    assert not no_leakage
    assert any("Extremely imbalanced split" in e for e in errors)


def test_regression_missing_classes_in_validation_or_test_detected():
    """7. Missing classes in validation or test must be flagged."""
    df = pd.DataFrame([
        {"image_id": "IMG_1", "visible_damage_type": "healthy"},
        {"image_id": "IMG_2", "visible_damage_type": "healthy"},
        {"image_id": "IMG_3", "visible_damage_type": "leaf_miner_damage"},
        {"image_id": "IMG_4", "visible_damage_type": "other_damage"},
    ])
    # Partition where val and test only have 'healthy' and missing miner and other
    manifest = {
        "train": ["IMG_3", "IMG_4"],
        "val": ["IMG_1"],
        "test": ["IMG_2"],
    }
    no_leakage, errors = verify_no_split_leakage(manifest, df, strict_classes=True)
    assert not no_leakage
    assert any("Class representation failure" in e for e in errors)
