"""Unit and integration tests for PhytoTrace Phase 4 inspection API and database."""

import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from phytotrace.api.main import app
from phytotrace.db.database import InspectionRepository


@pytest.fixture
def client():
    return TestClient(app)


def test_predict_and_inspection_lifecycle(client, tmp_path):
    """Test full cycle: upload leaf -> AI analysis -> DB save -> image retrieval -> report export."""
    # 1. Create a synthetic test leaf image with green vegetation
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (40, 160, 50)
    # Add some light mining streaks
    img[100:110, 100:180] = (200, 200, 200)
    _, buf = cv2.imencode(".jpg", img)

    files = {"file": ("tomato_leaf_test.jpg", buf.tobytes(), "image/jpeg")}
    data = {
        "farm_id": "FARM_NORTH_GREENHOUSE",
        "plant_id": "PLANT_ROW5_09",
        "leaf_id": "LEAF_CANOPY_01",
        "notes": "Field scout observed initial serpentine tracks.",
    }

    # 2. POST /api/predict
    response = client.post("/api/predict", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert "inspection_id" in res_data
    insp_id = res_data["inspection_id"]
    assert insp_id.startswith("INSP_")
    assert res_data["farm_id"] == "FARM_NORTH_GREENHOUSE"
    assert res_data["plant_id"] == "PLANT_ROW5_09"
    assert res_data["decision"] in [
        "no_visible_mine",
        "leaf_mask_unreliable",
        "leaf_not_detected",
        "image_quality_failure",
        "mine_candidates_rejected",
        "damage_detected",
        "no_qualifying_mine_detection",
        "mine_detected_species_uncertain",
        "possible_tuta",
        "possible_liriomyza",
        "possible_mixed",
        "other_damage",
        "image_quality_insufficient",
        "unsupported_image",
    ]
    assert "image_urls" in res_data
    assert res_data["image_urls"]["annotated"] == f"/api/inspections/{insp_id}/images/annotated"

    # 3. GET /api/inspections/{id}
    detail_resp = client.get(f"/api/inspections/{insp_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == insp_id
    assert detail["plant_id"] == "PLANT_ROW5_09"
    assert detail["notes"] == "Field scout observed initial serpentine tracks."

    # 4. GET /api/inspections/{id}/images/annotated
    img_resp = client.get(f"/api/inspections/{insp_id}/images/annotated")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] in ["image/jpeg", "image/png"]
    assert len(img_resp.content) > 100

    # 5. GET /api/inspections/{id}/images/leaf_mask
    leaf_resp = client.get(f"/api/inspections/{insp_id}/images/leaf_mask")
    assert leaf_resp.status_code == 200

    # 6. GET /api/inspections/{id}/report
    report_resp = client.get(f"/api/inspections/{insp_id}/report")
    assert report_resp.status_code == 200
    report_text = report_resp.text
    assert "PhytoTrace AI Diagnostic Inspection Report" in report_text
    assert insp_id in report_text
    assert "FARM_NORTH_GREENHOUSE" in report_text


def test_inspections_filtering_and_pagination(client):
    """Test listing inspections with limit, offset, and farm filter."""
    list_resp = client.get("/api/inspections?limit=10&offset=0")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)


def test_plant_timeline_and_progression(client):
    """Test longitudinal progression tracking for a specific plant."""
    plant_id = "PLANT_TIMELINE_TEST"

    # Create two inspections for same plant to verify progression delta
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:250, 50:250] = (40, 160, 50)
    _, buf = cv2.imencode(".jpg", img)

    # First inspection
    client.post(
        "/api/predict",
        files={"file": ("leaf1.jpg", buf.tobytes(), "image/jpeg")},
        data={"plant_id": plant_id, "farm_id": "FARM_TEST"},
    )

    # Second inspection
    client.post(
        "/api/predict",
        files={"file": ("leaf2.jpg", buf.tobytes(), "image/jpeg")},
        data={"plant_id": plant_id, "farm_id": "FARM_TEST"},
    )

    # Query timeline
    timeline_resp = client.get(f"/api/plants/{plant_id}/timeline")
    assert timeline_resp.status_code == 200
    timeline = timeline_resp.json()
    assert timeline["plant_id"] == plant_id
    assert timeline["total_inspections"] >= 2
    assert "timeline" in timeline
    assert len(timeline["timeline"]) >= 2
    # Check that progression delta was calculated
    second_record = timeline["timeline"][-1]
    assert "progression_delta" in second_record


def test_farm_overview_endpoint(client):
    """Test farm overview aggregation endpoint."""
    resp = client.get("/api/farms")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_inspections" in data
    assert "total_farms" in data
    assert "total_plants" in data
    assert "average_damage_pct" in data
    assert "severity_distribution" in data


def test_sample_images_endpoint(client):
    """Test curated sample images retrieval for quick testing."""
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) >= 3
    first_sample = samples[0]
    assert "filename" in first_sample
    assert "category" in first_sample

    # Test serving sample image
    img_resp = client.get(first_sample["url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/jpeg"

