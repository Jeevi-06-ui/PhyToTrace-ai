"""End-to-end integration test verifying real leaf analysis, Cloudinary upload,
MongoDB Atlas persistence, and strict user ownership isolation.
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from phytotrace.api.main import app
from phytotrace.auth.firebase import UserContext, get_current_user
from phytotrace.db.mongo import get_db
from phytotrace.storage import get_storage


@pytest.mark.skipif(os.getenv("PHYTOTRACE_RUN_LIVE_TESTS") != "1", reason="Requires explicitly authorized rotated cloud credentials")
def test_real_scan_cloudinary_and_mongodb_persistence():
    client = TestClient(app)
    db = get_db()
    storage = get_storage()

    # Verify live MongoDB connection
    assert db.is_mongo is True, "MongoDB Atlas must be connected for this live integration test"
    assert storage.is_active is True, "Cloudinary must be active for this live integration test"

    # Define User Alpha (authenticated owner)
    user_alpha = UserContext(
        uid="firebase_farmer_alpha_77",
        email="alpha@example.com",
        display_name="Farmer Alpha",
        is_guest=False,
        role="farmer",
    )

    # Override get_current_user dependency for User Alpha
    app.dependency_overrides[get_current_user] = lambda: user_alpha

    # Load genuine tomato leaf photograph from greenhouse dataset
    sample_path = Path("data/raw/original_photos/BLK_3_1002_PL010_H.JPG")
    if not sample_path.exists():
        raw_images = list(Path("data/raw/original_photos").glob("*.JPG"))
        assert len(raw_images) > 0, "No leaf test images available."
        sample_path = raw_images[0]

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    # Post to real /api/v1/analyze endpoint
    response = client.post(
        "/api/v1/analyze",
        files={"image": ("test_tomato_leaf.jpg", img_bytes, "image/jpeg")},
        data={"farm_id": "GREENHOUSE_SOUTH", "plant_id": "TOMATO_A07", "leaf_id": "LEAF_04"},
    )

    assert response.status_code == 200, f"Analysis failed: {response.text}"
    scan_data = response.json()
    insp_id = scan_data["id"]

    try:
        # 1. Verify real AI model execution outputs
        assert scan_data["quality_status"] in ["acceptable", "warning"]
        assert scan_data["failure_classification"]
        if scan_data["leaf_area_pixels"] is None:
            assert scan_data["damage_percentage"] is None
        assert "damage_percentage" in scan_data
        assert "severity_tier" in scan_data
        assert "damage_regions" in scan_data

        # 2. Verify Cloudinary asset URLs
        image_urls = scan_data["image_urls"]
        assert "cloudinary.com" in image_urls["original"] or "res.cloudinary.com" in image_urls["original"]
        assert "cloudinary.com" in image_urls["annotated"] or "res.cloudinary.com" in image_urls["annotated"]
        assert "cloudinary.com" in scan_data["thumbnail_url"] or "res.cloudinary.com" in scan_data["thumbnail_url"]

        # 3. Verify real persistence in MongoDB Atlas
        mongo_doc = db.db.inspections.find_one({"id": insp_id}, {"_id": 0})
        assert mongo_doc is not None, f"Inspection {insp_id} was not saved to MongoDB Atlas!"
        assert mongo_doc["owner_uid"] == user_alpha.uid
        assert mongo_doc["damage_percentage"] == scan_data["damage_percentage"]

        # 4. Verify user can query their own inspection via API
        get_res = client.get(f"/api/v1/analyses/{insp_id}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == insp_id

        # 5. Verify User Bravo (different user) CANNOT access User Alpha's inspection
        user_bravo = UserContext(
            uid="firebase_farmer_bravo_88",
            email="bravo@example.com",
            display_name="Farmer Bravo",
            is_guest=False,
            role="farmer",
        )
        app.dependency_overrides[get_current_user] = lambda: user_bravo

        forbidden_res = client.get(f"/api/v1/analyses/{insp_id}")
        assert forbidden_res.status_code == 404, "Data isolation failed: User Bravo accessed User Alpha's record!"

        # Verify User Bravo's inspection list does not contain User Alpha's inspection
        list_res = client.get("/api/v1/inspections")
        assert list_res.status_code == 200
        bravo_items = list_res.json().get("items", [])
        assert all(item["id"] != insp_id for item in bravo_items), "User Bravo saw User Alpha's record in list!"

    finally:
        keys = (db.db.inspections.find_one({"id": insp_id}) or {}).get("asset_keys", {})
        # Clean up dependency overrides
        app.dependency_overrides.clear()

        # Clean up MongoDB test inspection and regions
        db.db.inspections.delete_one({"id": insp_id})
        db.db.evidence_regions.delete_many({"inspection_id": insp_id})

        for key in keys.values():
            assert storage.delete_private(key)
