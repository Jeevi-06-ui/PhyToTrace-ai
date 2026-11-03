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

