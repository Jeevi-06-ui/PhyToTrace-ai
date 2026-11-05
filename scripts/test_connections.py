"""Test Cloudinary and MongoDB Atlas live connectivity and cleanup."""

import os
from pathlib import Path

# Load .env if present
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def test_mongo():
    from pymongo import MongoClient
    uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME", "phytotrace")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    ping = client.admin.command("ping")
    assert ping.get("ok") == 1, "Ping failed"
    db = client[db_name]
    # Test write and delete in a test collection
    res = db.connectivity_check.insert_one({"test": "atlas_live", "status": "ok"})
    doc = db.connectivity_check.find_one({"_id": res.inserted_id})
    assert doc is not None, "Failed to read back document from MongoDB Atlas"
    db.connectivity_check.delete_one({"_id": res.inserted_id})
    print("[PASS] MongoDB Atlas: Read/Write/Delete verified successfully.")


def test_cloudinary():
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api

    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    folder = os.environ.get("CLOUDINARY_FOLDER", "phytotrace")

    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )

    import io
    from PIL import Image

    # Create a genuine 32x32 RGB PNG image
    img = Image.new("RGB", (32, 32), color=(40, 180, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    res = cloudinary.uploader.upload(
        png_bytes,
        folder=f"{folder}/test",
        public_id="test_connectivity_asset",
        overwrite=True,
        resource_type="image",
    )
    assert "public_id" in res and "secure_url" in res, "Cloudinary upload failed"
    pub_id = res["public_id"]
    secure_url = res["secure_url"]

    # Verify retrieval info
    details = cloudinary.api.resource(pub_id)
    assert details.get("public_id") == pub_id, "Could not retrieve asset details"

    # Cleanup
    del_res = cloudinary.uploader.destroy(pub_id, resource_type="image")
    assert del_res.get("result") in ["ok", "not found"], f"Deletion failed: {del_res}"
    print("[PASS] Cloudinary: Upload/Retrieve/Delete verified successfully.")


if __name__ == "__main__":
    test_mongo()
    test_cloudinary()
