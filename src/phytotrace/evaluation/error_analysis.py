"""Error analysis and misclassification gallery generation."""

from pathlib import Path
from typing import Dict, List
import pandas as pd

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.error_analysis")


def compile_misclassified_examples(
    image_ids: List[str],
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    metadata_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, any]]:
    """Identify misclassified samples and correlate with metadata factors (lighting, variety, farm)."""
    gallery = []
    meta_lookup = {}
    if metadata_df is not None and not metadata_df.empty and "image_id" in metadata_df.columns:
        meta_lookup = metadata_df.set_index("image_id").to_dict(orient="index")

    for img_id, true_idx, pred_idx in zip(image_ids, y_true, y_pred):
        if true_idx != pred_idx:
            sample_info = {
                "image_id": img_id,
                "ground_truth": class_names[true_idx],
                "predicted": class_names[pred_idx],
            }
            if img_id in meta_lookup:
                m = meta_lookup[img_id]
                sample_info.update({
                    "farm_id": m.get("farm_id", "unknown"),
                    "tomato_variety": m.get("tomato_variety", "unknown"),
                    "image_quality": m.get("image_quality", "unknown"),
                    "notes": m.get("notes", ""),
                })
            gallery.append(sample_info)

    logger.info("Compiled %d misclassifications for error analysis review.", len(gallery))
    return gallery

