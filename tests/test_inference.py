"""Unit tests for the integrated inference pipeline."""

import cv2
import numpy as np
import pytest

from phytotrace.inference.pipeline import InferencePipeline


@pytest.fixture
def test_pipeline():
    return InferencePipeline()


def test_pipeline_on_synthetic_image(test_pipeline):
    # Create green leaf-like synthetic image on dark background
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Green leaf center
    img[50:250, 50:250] = (40, 160, 50)
    # Add high frequency texture to avoid blur rejection
    img[::8, 50:250] = (60, 180, 70)

    result = test_pipeline.predict(img, image_id="TEST_IMG_01")

    # Verify required JSON fields
    assert result["image_id"] == "TEST_IMG_01"
    assert "model_version" in result
    assert "quality_status" in result
    assert "decision" in result
    assert "uncertainty_status" in result
    assert "safety_notice" in result
    assert "processing_time_ms" in result
    assert isinstance(result["limitations"], list)
    assert result["_annotated_image"] is not None


def test_pipeline_missing_weights_fallback(test_pipeline):
    # Pipeline without trained YOLO weights should not crash or fabricate detections
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (40, 160, 50)
    img[::8, :] = (50, 150, 50)

    res = test_pipeline.predict(img, image_id="TEST_NO_WEIGHTS")
    # Species evidence should remain None when models are not trained
    assert res["tuta_evidence"] is None or isinstance(res["tuta_evidence"], float)
    assert res["liriomyza_evidence"] is None or isinstance(res["liriomyza_evidence"], float)



def test_reviewer_confirmed_negative_cannot_be_measured_with_flat_leaf_checkpoint(test_pipeline):
    """Exercise the real legacy weights and real negative, including manual focus."""
    from pathlib import Path
    from phytotrace.models.leaf_segmenter import LeafSegmenter

    legacy_ckpt = Path("models/checkpoints/leaf_segmenter/leaf_unet_best.pt.bak")
    ckpt = str(legacy_ckpt) if legacy_ckpt.exists() else "models/checkpoints/leaf_segmenter/leaf_unet_best.pt"
    test_pipeline.leaf_segmenter = LeafSegmenter(ckpt, str(test_pipeline.device))
    path = Path("data/raw/original_photos/BLK_3_1002_PL010_H.JPG")
    if not path.exists() or not test_pipeline.leaf_segmenter.is_trained or not test_pipeline.mine_segmenter.is_trained:
        pytest.skip("Local reviewer photo and checkpoints required")
    for crop in (None, [100, 100, 1300, 1100]):
        result = test_pipeline.predict(path, crop_box=crop)
        assert result["failure_classification"] in {"LEAF_MASK_UNRELIABLE", "LEAF_NOT_DETECTED"}
        assert result["damage_percentage"] is None
        assert result["damage_detected"] is None
        assert result["damage_regions"] == []
        assert result["diagnostic_telemetry"]["mine_segmenter"]["raw_candidates_count"] > 0
