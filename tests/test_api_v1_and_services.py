"""Tests for PhytoTrace v1 Enterprise API, Services, Cloudinary, and MongoDB Atlas."""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from phytotrace.api.main import app
from phytotrace.db.mongo import PhytoTraceDatabase, get_db
from phytotrace.storage.cloudinary_storage import CloudinaryStorage
from phytotrace.storage.r2 import CloudflareR2Storage
from phytotrace.storage import get_storage


@pytest.fixture
def client():
    return TestClient(app)


def test_v1_rejects_corrupt_upload_after_yolo_load(client):
    assert client.get("/api/v1/health").status_code == 200
    for content in (b"broken", b"\xff\xd8\xff" + b"broken"):
        response = client.post("/api/v1/analyze", files={"file": ("corrupt.jpg", content, "image/jpeg")})
        assert response.status_code == 400
        assert "JPEG" in response.json()["detail"]


def test_v1_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert "models_active" in data
    assert data["models_active"]["unet_leaf_segmenter"] is True
    assert data["models_active"]["yolo_mine_detector"] is True


def test_v1_models_status(client):
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "leaf_unet" in model_ids
    assert "mine_yolov8n" in model_ids
    assert "baseline_mobilenetv3" in model_ids


def test_v1_user_profile_guest(client):
    response = client.get("/api/v1/users/me", headers={"X-Guest-Session-ID": "test_guest_session_99"})
    assert response.status_code == 200
    data = response.json()
    assert data["is_guest"] is True
    assert data["uid"] == "guest"
    assert "preferences" in data


def test_v1_weather_current(client):
    response = client.get("/api/v1/weather/current?lat=12.9716&lon=77.5946&location=Bengaluru")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    assert "agronomic_guidance" in data
    assert "irrigation_advice" in data["agronomic_guidance"]
    assert "pest_scouting_recommendation" in data["agronomic_guidance"]
    assert "alerts" in data["agronomic_guidance"]


def test_v1_weather_crop_advice(client):
    payload = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "location_name": "Bengaluru, Karnataka",
        "growth_stage": "flowering",
        "soil_type": "red_loam"
    }
    response = client.post("/api/v1/weather/crop-advice", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["growth_stage"] == "flowering"
    assert "agronomic_guidance" in data


def test_assistant_chat_endpoint(client):
    payload = {
        "message": "What are the early signs of tomato leafminer damage?",
        "session_id": "pytest_test_session",
        "language": "en"
    }
    response = client.post("/api/v1/assistant/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert len(data["message"]) > 20
    assert "source" in data
    assert "citations" in data


def test_database_sqlite_fallback():
    # Explicitly test SQLite fallback engine
    db = PhytoTraceDatabase(uri="")
    assert db.is_mongo is False

    # Test inspection save and retrieve
    record = {
        "id": "INSP_TEST_SQLITE_001",
        "farm_id": "GREENHOUSE_A",
        "plant_id": "PL001",
        "leaf_id": "LF01",
        "damage_percentage": 14.5,
        "severity_tier": "moderate",
        "damage_regions": [{"label": "Region A", "area_pixels": 120}]
    }
    saved_id = db.save_inspection(record)
    assert saved_id == "INSP_TEST_SQLITE_001"
    retrieved = db.get_inspection(saved_id)
    assert retrieved["damage_percentage"] == 14.5
    assert len(retrieved["damage_regions"]) == 1
    assert db.get_inspection(saved_id, owner_uid="some-user") is None
    assert db.list_inspections(owner_uid="some-user")["items"] == []


def test_mongodb_atlas_live_persistence_and_isolation():
    db = get_db()
    if not db.is_mongo:
        pytest.skip("MongoDB Atlas URI not configured or network unreachable.")

    # 1. Test collection readiness
    expected_collections = {
        "users", "farms", "plants", "inspections", "evidence_regions",
        "weather_snapshots", "assistant_sessions", "user_preferences"
    }
    existing_cols = set(db.db.list_collection_names())
    # Ensure they exist or are initialized
    for col in expected_collections:
        if col not in existing_cols:
            db.db[col].insert_one({"_init": True})
            db.db[col].delete_one({"_init": True})

    # 2. Test inspection write with owner isolation
    test_insp_id = "INSP_TEST_ATLAS_VERIFY_99"
    record = {
        "id": test_insp_id,
        "inspection_id": test_insp_id,
        "farm_id": "ATLAS_FARM_01",
        "plant_id": "PL_ATLAS_99",
        "leaf_id": "LF_01",
        "damage_percentage": 8.7,
        "severity_tier": "low",
        "decision": "mine_detected_species_uncertain",
        "damage_regions": [
            {"region_id": 0, "label": "Region A", "area_pixels": 95}
        ]
    }
    owner_a = "firebase_user_alpha"
    owner_b = "firebase_user_bravo"

    db.save_inspection(record, owner_uid=owner_a)

    # Owner A can read it
    insp_a = db.get_inspection(test_insp_id, owner_uid=owner_a)
    assert insp_a is not None
    assert insp_a["id"] == test_insp_id
    assert insp_a["damage_percentage"] == 8.7

    # Owner B CANNOT read Owner A's inspection (Strict Isolation)
    insp_b = db.get_inspection(test_insp_id, owner_uid=owner_b)
    assert insp_b is None

    # 3. Test user preferences
    db.save_user_preferences(owner_a, {"temperature_unit": "celsius", "notifications": True})
    prefs = db.get_user_preferences(owner_a)
    assert prefs.get("temperature_unit") == "celsius"

    # 4. Clean up test record from Atlas
    db.db.inspections.delete_one({"id": test_insp_id})
    db.db.evidence_regions.delete_many({"inspection_id": test_insp_id})
    db.db.user_preferences.delete_one({"firebase_uid": owner_a})


def test_cloudinary_storage_live():
    storage = CloudinaryStorage()
    if not storage.is_active:
        pytest.skip("Cloudinary credentials not configured.")

    import io
    from PIL import Image

    # Create test image
    img = Image.new("RGB", (64, 64), color=(60, 160, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    test_key = "test/pytest_live_test_image.png"
    url = storage.upload_bytes(data, test_key, content_type="image/png")
    assert "cloudinary.com" in url or "res.cloudinary.com" in url

    # Verify exists
    assert storage.exists(test_key) is True

    # Verify thumbnail url
    thumb_url = storage.get_thumbnail_url(test_key, width=128, height=128)
    assert "w_128" in thumb_url

    # Cleanup
    deleted = storage.delete(test_key)
    assert deleted is True


def test_cloudflare_r2_local_fallback():
    storage = CloudflareR2Storage()
    assert storage.is_r2_active is False

    test_bytes = b"\x89PNG\r\n\x1a\nfakeimagebytes"
    rel_path = storage.upload_bytes(test_bytes, "inspections/test.png", content_type="image/png")
    assert "/api/storage/inspections/test.png" == rel_path

    # Verify exists
    assert storage.exists("inspections/test.png") is True
    assert storage.exists("inspections/nonexistent_file.png") is False

    # Verify download
    downloaded = storage.download_bytes("inspections/test.png")
    assert downloaded == test_bytes

    # Verify url
    url = storage.get_url("inspections/test.png")
    assert "/inspections/test.png" in url


def test_leaf_outline_rejects_invalid_original_pixel_geometry():
    import json
    import numpy as np
    from fastapi import HTTPException
    from phytotrace.api.scan_input import read_leaf_polygon

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    valid = [[50, 40], [210, 60], [150, 110], [200, 170], [45, 170]]
    assert read_leaf_polygon(json.dumps(valid), image) == valid
    for invalid in ([[30, 30], [140, 140], [30, 140], [140, 30]],
                    [[50, 40], [320, 60], [150, 110]], [[50, 40], [51, 41], [52, 42]],
                    [[50, 40], [210, 60], [50, 40]], [[True, 40], [210, 60], [150, 110]]):
        with pytest.raises(HTTPException) as error:
            read_leaf_polygon(json.dumps(invalid), image)
        assert error.value.status_code == 422
