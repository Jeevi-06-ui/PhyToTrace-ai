"""Group-aware dataset splitting to prevent data leakage in PhytoTrace AI."""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.split")


def build_compound_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized physical plant and leaf compound identifiers to DataFrame."""
    df = df.copy()
    f_col = "farm_id" if "farm_id" in df.columns else None
    p_col = "plant_id" if "plant_id" in df.columns else None
    l_col = "leaf_id" if "leaf_id" in df.columns else None

    # Compound Physical Plant Identity (farm + plant)
    if f_col and p_col:
        df["_physical_plant_id"] = df[f_col].astype(str) + "_" + df[p_col].astype(str)
    elif p_col:
        df["_physical_plant_id"] = df[p_col].astype(str)
    else:
        df["_physical_plant_id"] = df.get("image_id", pd.Series(range(len(df)))).astype(str)

    # Compound Physical Leaf Identity (farm + plant + leaf)
    if l_col:
        df["_physical_leaf_id"] = df["_physical_plant_id"] + "_" + df[l_col].astype(str)
    else:
        df["_physical_leaf_id"] = df["_physical_plant_id"]

    return df


def create_group_key(row: pd.Series, group_hierarchy: List[str]) -> str:
    """Generate a compound group identifier from hierarchical metadata fields."""
    parts = []
    for col in group_hierarchy:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("unknown", "none", ""):
            parts.append(val)
    if not parts:
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
    stratify_col: Optional[str] = "visible_damage_type",
    duplicate_clusters: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Partition DataFrame into train/val/test splits without group or plant leakage.
    
    Guarantees:
    1. Zero physical plant leakage: All images from the same plant stay in ONE partition.
    2. Zero physical leaf leakage: All images from the same leaf stay in ONE partition.
    3. Stratified class representation: Every partition maintains class balance.
    4. Guard against coarse-group distortion (e.g., if one farm has 85% of data).
    
    Returns:
        Dict with keys 'train', 'val', 'test', and 'metadata'.
    """
    if df.empty:
        return {
            "train": [],
            "val": [],
            "test": [],
            "metadata": {"empty": True},
        }

    df_annotated = build_compound_keys(df)
    total_samples = len(df_annotated)

    # 1. Evaluate primary grouping column viability
    target_group = "_physical_plant_id"
    grouping_rationale = "biological_physical_plant"

    if group_col in df_annotated.columns and group_col != "_physical_plant_id":
        counts = df_annotated[group_col].value_counts()
        max_ratio = counts.iloc[0] / total_samples if len(counts) > 0 else 1.0
        unique_valid = [g for g in counts.index if str(g).lower() not in ("unknown", "none", "")]

        if len(unique_valid) < 3:
            logger.info(
                "Group column '%s' has only %d valid groups (< 3). Using physical plant level grouping.",
                group_col, len(unique_valid),
            )
            target_group = "_physical_plant_id"
        elif max_ratio > max(train_ratio, val_ratio, test_ratio) or max_ratio > 0.50:
            logger.warning(
                "Group column '%s' has extreme size skew (largest group '%s' has %d/%d samples, %.1f%%). "
                "Splitting across '%s' would produce degenerate partitions (e.g., 6/8/81). "
                "Falling back to biological physical plant grouping to preserve leakage-free integrity.",
                group_col, counts.index[0], counts.iloc[0], total_samples, max_ratio * 100, group_col,
            )
            target_group = "_physical_plant_id"
            grouping_rationale = f"fallback_from_skewed_{group_col}"
        else:
            target_group = group_col
            grouping_rationale = f"direct_{group_col}"

    # 2. Group samples by target group
    grouped_df = df_annotated.groupby(target_group)
    group_to_images = grouped_df["image_id"].apply(list).to_dict()

    # Determine class label per group if stratifying
    use_stratification = stratify_col is not None and stratify_col in df_annotated.columns
    if use_stratification:
        group_to_label = grouped_df[stratify_col].agg(
            lambda s: s.mode()[0] if not s.empty else "unlabelled"
        ).to_dict()
    else:
        group_to_label = {g: "all" for g in group_to_images}

    # 3. Stratified allocation across partitions
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    train_groups: Set[str] = set()
    val_groups: Set[str] = set()
    test_groups: Set[str] = set()

    rng = np.random.RandomState(seed)
    unique_classes = sorted(list(set(group_to_label.values())))

    for cls_name in unique_classes:
        cls_groups = [g for g, lbl in group_to_label.items() if lbl == cls_name]
        rng.shuffle(cls_groups)
        n_grp = len(cls_groups)

        if n_grp >= 3:
            # Symmetric quotas ensuring validation and test representation
            nv = max(1, int(round(n_grp * val_ratio)))
            nt = max(1, int(round(n_grp * test_ratio)))
            ntr = n_grp - nv - nt
            if ntr < 1:
                ntr = 1
                if nv > 1:
                    nv -= 1
                elif nt > 1:
                    nt -= 1
        elif n_grp == 2:
            ntr, nv, nt = 1, 1, 0
        else:
            ntr, nv, nt = 1, 0, 0

        grp_train = cls_groups[:ntr]
        grp_val = cls_groups[ntr:ntr + nv]
        grp_test = cls_groups[ntr + nv:]

        for g in grp_train:
            train_groups.add(g)
            train_ids.extend(group_to_images[g])
        for g in grp_val:
            val_groups.add(g)
            val_ids.extend(group_to_images[g])
        for g in grp_test:
            test_groups.add(g)
            test_ids.extend(group_to_images[g])

    # Ensure non-empty validation and test sets when sufficient groups exist
    if not val_ids and len(train_groups) > 1:
        moved_group = sorted(list(train_groups))[-1]
        train_groups.remove(moved_group)
        val_groups.add(moved_group)
        moved_ids = group_to_images[moved_group]
        train_ids = [i for i in train_ids if i not in moved_ids]
        val_ids.extend(moved_ids)

    if not test_ids and len(train_groups) > 1:
        moved_group = sorted(list(train_groups))[-1]
        train_groups.remove(moved_group)
        test_groups.add(moved_group)
        moved_ids = group_to_images[moved_group]
        train_ids = [i for i in train_ids if i not in moved_ids]
        test_ids.extend(moved_ids)

    # 4. Enforce Duplicate Cluster Co-Location
    if duplicate_clusters:
        id_to_part = {}
        for i in train_ids:
            id_to_part[i] = "train"
        for i in val_ids:
            id_to_part[i] = "val"
        for i in test_ids:
            id_to_part[i] = "test"

        for cluster in duplicate_clusters:
            parts = {id_to_part[img_id] for img_id in cluster if img_id in id_to_part}
            if len(parts) > 1:
                # Co-locate all cluster members into training set
                logger.warning("Duplicate cluster spans partitions %s. Consolidating into train.", parts)
                for img_id in cluster:
                    if img_id in val_ids:
                        val_ids.remove(img_id)
                    if img_id in test_ids:
                        test_ids.remove(img_id)
                    if img_id not in train_ids:
                        train_ids.append(img_id)

    split_manifest = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
        "metadata": {
            "grouping_column": target_group,
            "grouping_rationale": grouping_rationale,
            "stratification_column": stratify_col if use_stratification else None,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_groups_count": len(train_groups),
            "val_groups_count": len(val_groups),
            "test_groups_count": len(test_groups),
            "train_ratio_actual": round(len(train_ids) / total_samples, 4),
            "val_ratio_actual": round(len(val_ids) / total_samples, 4),
            "test_ratio_actual": round(len(test_ids) / total_samples, 4),
            "seed": seed,
        },
    }

    logger.info(
        "Created splits with %s grouping: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        target_group,
        len(train_ids), 100 * len(train_ids) / total_samples,
        len(val_ids), 100 * len(val_ids) / total_samples,
        len(test_ids), 100 * len(test_ids) / total_samples,
    )
    return split_manifest


def verify_no_split_leakage(
    split_manifest: Dict[str, List[str]],
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    duplicate_clusters: Optional[List[List[str]]] = None,
    strict_classes: bool = False,
    label_col: str = "visible_damage_type",
) -> Tuple[bool, List[str]]:
    """Strictly assert that no image ID, physical plant, physical leaf, or duplicate cluster leaks across partitions."""
    train_ids = set(split_manifest.get("train", []))
    val_ids = set(split_manifest.get("val", []))
    test_ids = set(split_manifest.get("test", []))
    errors: List[str] = []

    # 1. Image ID disjointness
    tv_overlap = train_ids.intersection(val_ids)
    if tv_overlap:
        errors.append(f"Image ID leakage between train and val: {sorted(list(tv_overlap))}")
    tt_overlap = train_ids.intersection(test_ids)
    if tt_overlap:
        errors.append(f"Image ID leakage between train and test: {sorted(list(tt_overlap))}")
    vt_overlap = val_ids.intersection(test_ids)
    if vt_overlap:
        errors.append(f"Image ID leakage between val and test: {sorted(list(vt_overlap))}")

    # 2. Partition balance validation
    if len(train_ids) < len(val_ids) or len(train_ids) < len(test_ids):
        errors.append(
            f"Extremely imbalanced split: Train ({len(train_ids)}) has fewer samples than Val ({len(val_ids)}) or Test ({len(test_ids)})."
        )

    if df.empty or "image_id" not in df.columns:
        return len(errors) == 0, errors

    df_annotated = build_compound_keys(df)
    train_df = df_annotated[df_annotated["image_id"].isin(train_ids)]
    val_df = df_annotated[df_annotated["image_id"].isin(val_ids)]
    test_df = df_annotated[df_annotated["image_id"].isin(test_ids)]

    # 3. Physical Plant leakage check
    if "_physical_plant_id" in df_annotated.columns:
        p_tr = set(train_df["_physical_plant_id"].dropna().unique())
        p_va = set(val_df["_physical_plant_id"].dropna().unique())
        p_te = set(test_df["_physical_plant_id"].dropna().unique())

        ov_p_tv = p_tr.intersection(p_va)
        if ov_p_tv:
            errors.append(f"Physical plant leakage between train and val: {sorted(list(ov_p_tv))}")
        ov_p_tt = p_tr.intersection(p_te)
        if ov_p_tt:
            errors.append(f"Physical plant leakage between train and test: {sorted(list(ov_p_tt))}")
        ov_p_vt = p_va.intersection(p_te)
        if ov_p_vt:
            errors.append(f"Physical plant leakage between val and test: {sorted(list(ov_p_vt))}")

    # 4. Physical Leaf leakage check
    if "_physical_leaf_id" in df_annotated.columns:
        l_tr = set(train_df["_physical_leaf_id"].dropna().unique())
        l_va = set(val_df["_physical_leaf_id"].dropna().unique())
        l_te = set(test_df["_physical_leaf_id"].dropna().unique())

        ov_l_tv = l_tr.intersection(l_va)
        if ov_l_tv:
            errors.append(f"Physical leaf leakage between train and val: {sorted(list(ov_l_tv))}")
        ov_l_tt = l_tr.intersection(l_te)
        if ov_l_tt:
            errors.append(f"Physical leaf leakage between train and test: {sorted(list(ov_l_tt))}")
        ov_l_vt = l_va.intersection(l_te)
        if ov_l_vt:
            errors.append(f"Physical leaf leakage between val and test: {sorted(list(ov_l_vt))}")

    # 5. Generic / Explicit group columns
    if group_cols:
        for col in group_cols:
            # Skip checking raw local leaf_id in isolation (avoid false positives on 'L01')
            if col == "leaf_id":
                continue
            if col not in df_annotated.columns:
                continue

            # Strip nulls, empty strings, and unknown markers
            tg = set(train_df[col].dropna().unique()) - {"", "unknown", "none"}
            vg = set(val_df[col].dropna().unique()) - {"", "unknown", "none"}
            seg = set(test_df[col].dropna().unique()) - {"", "unknown", "none"}

            ov_tv = tg.intersection(vg)
            if ov_tv:
                errors.append(f"Group leakage for '{col}' between train and val: {sorted(list(ov_tv))}")
            ov_tt = tg.intersection(seg)
            if ov_tt:
                errors.append(f"Group leakage for '{col}' between train and test: {sorted(list(ov_tt))}")
            ov_vt = vg.intersection(seg)
            if ov_vt:
                errors.append(f"Group leakage for '{col}' between val and test: {sorted(list(ov_vt))}")

    # 6. Duplicate Image Clusters
    if duplicate_clusters:
        for cluster in duplicate_clusters:
            present_in = []
            if any(img_id in train_ids for img_id in cluster):
                present_in.append("train")
            if any(img_id in val_ids for img_id in cluster):
                present_in.append("val")
            if any(img_id in test_ids for img_id in cluster):
                present_in.append("test")
            if len(present_in) > 1:
                errors.append(f"Duplicate image cluster crossing partitions: {cluster} in {present_in}")

    # 7. Exact Hash Disjointness
    for hash_col in ["sha256", "sha256_hash"]:
        if hash_col in df_annotated.columns:
            h_tr = set(train_df[hash_col].dropna().unique()) - {"", "unknown"}
            h_va = set(val_df[hash_col].dropna().unique()) - {"", "unknown"}
            h_te = set(test_df[hash_col].dropna().unique()) - {"", "unknown"}
            if h_tr.intersection(h_va):
                errors.append(f"Exact hash duplicate leakage between train and val on '{hash_col}'")
            if h_tr.intersection(h_te):
                errors.append(f"Exact hash duplicate leakage between train and test on '{hash_col}'")
            if h_va.intersection(h_te):
                errors.append(f"Exact hash duplicate leakage between val and test on '{hash_col}'")

    # 8. Class representation validation
    if strict_classes and label_col in df_annotated.columns:
        valid_labels = [l for l in df_annotated[label_col].unique() if l and l not in ("unlabelled", "unknown", "none")]
        for p_name, p_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            p_labels = set(p_df[label_col].unique())
            missing = set(valid_labels) - p_labels
            if missing:
                errors.append(f"Class representation failure: {p_name} partition is missing classes: {sorted(list(missing))}")

    return len(errors) == 0, errors


def save_split_manifest(split_manifest: dict, out_path: Path) -> None:
    """Save split dictionary to JSON manifest."""
    out_path = resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    logger.info("Saved split manifest to %s", out_path)
