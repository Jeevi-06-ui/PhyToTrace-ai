"""Unit tests for image preprocessing, duplicate checks, and quality assessment."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

from phytotrace.data.deduplicate import compute_dhash, compute_sha256, detect_duplicates
from phytotrace.data.validators import quarantine_corrupted_image, validate_single_image
from phytotrace.preprocessing.image_utils import crop_region_with_context, letterbox_image
from phytotrace.preprocessing.quality import assess_image_quality


def test_corrupt_image_validation_and_quarantine(tmp_path):
    corrupt_file = tmp_path / "broken.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    is_valid, reason, dims = validate_single_image(corrupt_file)
    assert not is_valid
    assert "cannot decode" in reason or "corrupted" in reason.lower()

    # Quarantine
    q_dir = tmp_path / "quarantine"
    dest = quarantine_corrupted_image(corrupt_file, q_dir, reason)
    assert dest.exists()
    assert not corrupt_file.exists()


def test_duplicate_detection_exact_and_near(tmp_path):
    # Image 1
    arr1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    p1 = tmp_path / "img1.png"
    Image.fromarray(arr1).save(p1)

    # Image 2 (Exact duplicate)
    p2 = tmp_path / "img2.png"
    Image.fromarray(arr1).save(p2)

    # Image 3 (Near duplicate with subtle noise)
    arr3 = arr1.copy()
    arr3[10:20, 10:20] = 105
    p3 = tmp_path / "img3.png"
    Image.fromarray(arr3).save(p3)

    exact_dups, near_clusters = detect_duplicates([p1, p2, p3], near_dup_threshold=4)
    assert len(exact_dups) == 1
    assert p1 in list(exact_dups.values())[0]
    assert p2 in list(exact_dups.values())[0]


def test_quality_assessment_sharp_vs_blurry():
    # Sharp image: high frequency checkerboard
    sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
    sharp_img[::10, :] = 255
    sharp_img[:, ::10] = 255
    res_sharp = assess_image_quality(sharp_img, blur_threshold=100.0)
    assert res_sharp["is_acceptable"]
    assert res_sharp["blur_metric"] > 100.0

    # Blurry image: uniform gray
    blurry_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    res_blurry = assess_image_quality(blurry_img, blur_threshold=100.0)
    assert not res_blurry["is_acceptable"]
    assert res_blurry["quality_status"] in {"insufficient_sharpness", "poor_lighting_and_blur"}


def test_letterbox_and_crop():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    padded, scale, (pl, pt) = letterbox_image(img, target_size=(640, 640))
    assert padded.shape == (640, 640, 3)

    crop = crop_region_with_context(img, bbox=(50, 50, 100, 100), context_margin_ratio=0.1)
    assert crop.shape[0] > 50
    assert crop.shape[1] > 50

