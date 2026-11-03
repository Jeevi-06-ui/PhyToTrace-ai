"""Group-aware dataset splitting to prevent data leakage."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.split")


def create_group_key(row: pd.Series, group_hierarchy: List[str]) -> str:
    """Generate a compound group identifier from hierarchical metadata fields."""
    parts = []
    for col in group_hierarchy:
        val = str(row.get(col, "")).strip()
        if val and val != "unknown" and val != "none":
            parts.append(val)
    if not parts:
        # Fallback to row index or image_id if no group info
        return str(row.get("image_id", "ungrouped"))
    return "_".join(parts)


def make_group_splits(
    df: pd.DataFrame,
    group_col: str = "farm_id",
    fallback_group_col: str = "plant_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition DataFrame into train/val/test splits without group leakage.
    
    All images belonging to the same group_key (e.g. farm_id or plant_id) are assigned
    strictly to the SAME split partition.
    
    Args:
        df: Input DataFrame containing image metadata.
        group_col: Primary grouping column (e.g. 'farm_id').
        fallback_group_col: Secondary grouping column if primary is uniform or empty.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for test (default 0.15).
        seed: Random seed for reproducible assignment.
        
    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a list of image_ids.
    """
    if df.empty:
        return {"train": [], "val": [], "test": []}
        
    np.random.seed(seed)
    
    # Determine the most granular effective grouping key
    unique_primary = [g for g in df[group_col].unique() if g and g != "unknown" and g != "none"] if group_col in df.columns else []
    
    if len(unique_primary) >= 3:
        target_group = group_col
    elif fallback_group_col in df.columns:
        target_group = fallback_group_col
    else:
        target_group = "image_id"
        logger.warning("Fewer than 3 unique sites/groups found. Splitting at %s level with grouping caution.", target_group)

    # Group image IDs by target group
    grouped = df.groupby(target_group)["image_id"].apply(list).to_dict()
    group_names = list(grouped.keys())
    np.random.shuffle(group_names)
    
    total_samples = len(df)
    target_train = int(total_samples * train_ratio)
    target_val = int(total_samples * val_ratio)
    
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    
    train_groups: Set[str] = set()
    val_groups: Set[str] = set()
    test_groups: Set[str] = set()
    
    for g in group_names:
        ids = grouped[g]
        # Assign to train first until quota reached
        if len(train_ids) + len(ids) <= target_train or (not train_ids and not val_ids and not test_ids):
            train_ids.extend(ids)
            train_groups.add(g)
        elif len(val_ids) + len(ids) <= target_val:
            val_ids.extend(ids)
            val_groups.add(g)
        else:
            test_ids.extend(ids)
            test_groups.add(g)
            
    # Check for empty validation or test sets in small sample scenarios
    if not val_ids and len(train_groups) > 1:
        # Move one group from train to val
        moved_group = list(train_groups)[-1]
        train_groups.remove(moved_group)
        moved_ids = grouped[moved_group]
        train_ids = [i for i in train_ids if i not in moved_ids]
        val_ids.extend(moved_ids)
        val_groups.add(moved_group)

    if not test_ids and len(train_groups) > 1:
        # Move one group from train to test
        moved_group = list(train_groups)[-1]
        train_groups.remove(moved_group)
        moved_ids = grouped[moved_group]
        train_ids = [i for i in train_ids if i not in moved_ids]
        test_ids.extend(moved_ids)
        test_groups.add(moved_group)
        
    split_manifest = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "metadata": {
            "grouping_column": target_group,
            "train_groups": sorted(list(train_groups)),
            "val_groups": sorted(list(val_groups)),
            "test_groups": sorted(list(test_groups)),
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "seed": seed,
        }
    }
    
    logger.info(
        "Created splits with %s grouping: Train=%d, Val=%d, Test=%d (Total=%d)",
        target_group, len(train_ids), len(val_ids), len(test_ids), total_samples,
    )
    return split_manifest


def verify_no_split_leakage(split_manifest: Dict[str, List[str]], df: pd.DataFrame, group_cols: List[str]) -> Tuple[bool, List[str]]:
    """Strictly assert that no image ID or group key appears in more than one split partition."""
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))
    test_ids = set(split_manifest.get("test", []))
    
    errors = []
    
    # 1. Image ID disjointness
    tv_overlap = train_ids.intersection(val_ids)
    if tv_overlap:
        errors.append(f"Image ID leakage between train and val: {tv_overlap}")
    tt_overlap = train_ids.intersection(test_ids)
    if tt_overlap:
        errors.append(f"Image ID leakage between train and test: {tt_overlap}")
    vt_overlap = val_ids.intersection(test_ids)
    if vt_overlap:
        errors.append(f"Image ID leakage between val and test: {vt_overlap}")
        
    # 2. Group column disjointness
    if not df.empty and "image_id" in df.columns:
        train_df = df[df["image_id"].isin(train_ids)]
        val_df = df[df["image_id"].isin(val_ids)]
        test_df = df[df["image_id"].isin(test_ids)]
        
        for col in group_cols:
            if col not in df.columns:
                continue
            tg = set(train_df[col].dropna().unique()) - {"", "unknown", "none"}
            vg = set(val_df[col].dropna().unique()) - {"", "unknown", "none"}
            seg = set(test_df[col].dropna().unique()) - {"", "unknown", "none"}
            
            ov_tv = tg.intersection(vg)
            if ov_tv:
                errors.append(f"Group leakage for '{col}' between train and val: {ov_tv}")
            ov_tt = tg.intersection(seg)
            if ov_tt:
                errors.append(f"Group leakage for '{col}' between train and test: {ov_tt}")
            ov_vt = vg.intersection(seg)
            if ov_vt:
                errors.append(f"Group leakage for '{col}' between val and test: {ov_vt}")
                
    return len(errors) == 0, errors


def save_split_manifest(split_manifest: dict, out_path: Path) -> None:
    """Save split dictionary to JSON manifest."""
    out_path = resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    logger.info("Saved split manifest to %s", out_path)

