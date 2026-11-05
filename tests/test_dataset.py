"""Unit tests for dataset schemas, loading, and synthetic fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from phytotrace.data.dataset import (
    TomatoDamageEvidenceDataset,
    TomatoLeafClassificationDataset,
    TomatoLeafSegmentationDataset,
)
from phytotrace.data.metadata import MetadataManager
from phytotrace.data.schemas import ImageMetadataRecord


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a temporary directory with synthetic test leaf images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(5):
        # Create small 256x256 RGB image
        arr = np.full((256, 256, 3), (30 + i * 20, 150, 40), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_dir / f"test_{i}.jpg")

    return img_dir


@pytest.fixture
def sample_metadata_csv(tmp_path, sample_image_dir):
    """Create a valid temporary metadata CSV."""
    records = []
    damage_types = ["none", "leaf_miner_mine", "chewing", "leaf_miner_mine", "none"]
    for i in range(5):
        records.append({
            "image_id": f"IMG_{i+1:04d}",
            "original_filename": f"test_{i}.jpg",
            "image_path": str(sample_image_dir / f"test_{i}.jpg"),
            "collector_id": "COL_TEST",
            "collection_date": "2026-10-21",
            "farm_id": f"FARM_{i%2}",
            "field_id": "F1",
            "row_id": "R1",
            "plant_id": f"P_{i}",
            "leaf_id": f"L_{i}",
            "collection_session_id": "SESS_TEST",
            "tomato_variety": "Arka_Rakshak",
            "growth_stage": "fruiting",
            "image_view_type": "whole_leaf",
            "image_quality": "sharp",
            "visible_damage_type": damage_types[i],
            "suspected_pest": "unknown",
            "confirmed_pest": "unconfirmed",
            "pest_confirmation_method": "none",
            "expert_reviewer_id": "none",
            "label_verification_status": "unverified_field_guess",
            "mixed_infestation_status": "not_evaluated",
            "annotation_status": "unannotated",
            "notes": "Synthetic fixture test sample",
            "consent_or_permission": "yes",
            "usage_license": "CC-BY-4.0",
        })
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


def test_metadata_validation(sample_metadata_csv):
    mgr = MetadataManager(sample_metadata_csv)
    is_valid, errors = mgr.validate_all_records()
    assert is_valid, f"Metadata validation failed: {errors}"
    assert len(mgr.df) == 5


def test_classification_dataset_loading(sample_metadata_csv, sample_image_dir):
    df = pd.read_csv(sample_metadata_csv)
    ds = TomatoLeafClassificationDataset(df=df, image_dir=sample_image_dir)
    assert len(ds) == 5

    x, y, img_id = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3
    assert isinstance(y, int)
    assert y in {0, 1, 2}
    assert img_id == "IMG_0001"


def test_classification_dataset_label_mapping():
    assert TomatoLeafClassificationDataset.LABEL_MAP["healthy"] == 0
    assert TomatoLeafClassificationDataset.LABEL_MAP["none"] == 0
    assert TomatoLeafClassificationDataset.LABEL_MAP["leaf_miner_mine"] == 1
    assert TomatoLeafClassificationDataset.LABEL_MAP["leaf_miner_damage"] == 1
    assert TomatoLeafClassificationDataset.LABEL_MAP["chewing"] == 2
    assert TomatoLeafClassificationDataset.LABEL_MAP["other"] == 2


def test_segmentation_dataset_loading(sample_image_dir):
    imgs = list(sample_image_dir.glob("*.jpg"))
    mask_data = np.zeros((256, 256), dtype=np.uint8)
    mask_data[32:224, 32:224] = 1
    masks = [image.with_suffix(".png") for image in imgs]
    for path in masks:
        Image.fromarray(mask_data).save(path)
    ds = TomatoLeafSegmentationDataset(image_paths=imgs, mask_paths=masks, target_size=(256, 256))
    assert len(ds) == 5
    x, mask = ds[0]
    assert x.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert set(mask.unique().tolist()) == {0.0, 1.0}


def test_evidence_dataset_unverified_masking(tmp_path, sample_image_dir):
    # Unverified label should NOT produce positive or negative loss targets
    records = [{
        "image_id": "IMG_9999",
        "original_filename": "test_0.jpg",
        "image_path": str(sample_image_dir / "test_0.jpg"),
        "confirmed_pest": "unconfirmed",
        "label_verification_status": "unverified_field_guess",
    }]
    df = pd.DataFrame(records)
    ds = TomatoDamageEvidenceDataset(df)
    _, targets, valid_mask, _ = ds[0]

    # Valid mask should be 0 (ignored by MaskedBCEWithLogitsLoss)
    assert valid_mask.sum().item() == 0.0

