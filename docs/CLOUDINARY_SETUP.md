# PhytoTrace AI — Cloudinary Asset Storage Setup Guide

This document describes how PhytoTrace AI uses **Cloudinary** as the primary storage provider for leaf photographs, AI-generated segmentation overlays, binary masks, and thumbnails.

---

## 1. Overview & Advantages

Cloudinary serves as the primary asset delivery network for PhytoTrace:
- **Server-Side Signed Uploads:** Images and overlays are securely uploaded directly from the FastAPI backend using private API keys.
- **Dynamic On-the-Fly Transformations:** Generates responsive web thumbnails (e.g. `w_300,h_300,c_fill`) without consuming additional local storage.
- **Secure URLs:** Serves assets via encrypted HTTPS CDN endpoints.
- **Direct Asset Deletion:** Allows automatic removal of test scans and compliance with farmer privacy requests.

---

## 2. Environment Configuration

Add the following environment variables to your backend `.env`:

```bash
# Primary Asset Storage Provider
STORAGE_PROVIDER=cloudinary

# Cloudinary Credentials
CLOUDINARY_CLOUD_NAME=dlktbzcut
CLOUDINARY_API_KEY=877976886375243
CLOUDINARY_API_SECRET=your_api_secret_here
CLOUDINARY_FOLDER=phytotrace
```

---

## 3. Storage Hierarchy & Asset Types

When a leaf photograph is inspected via `/api/v1/analyze`, the following assets are published to Cloudinary under `{CLOUDINARY_FOLDER}/inspections/{insp_id}/`:

| Asset Role | Cloudinary Public ID Pattern | Format | Description |
| :--- | :--- | :--- | :--- |
| **Original Photograph** | `{folder}/inspections/{insp_id}/original` | JPEG | Native unprocessed field camera photograph |
| **AI Annotated Overlay** | `{folder}/inspections/{insp_id}/annotated` | JPEG | Composite image with U-Net boundary and YOLOv8 mine lesion boxes |
| **Leaf Boundary Mask** | `{folder}/inspections/{insp_id}/leaf_mask` | PNG | Binary mask (0/255) of segmented leaf lamina |
| **Mine Lesion Mask** | `{folder}/inspections/{insp_id}/mine_mask` | PNG | Binary mask (0/255) of localized serpentine mines |
| **Inspection Thumbnail** | `{folder}/inspections/{insp_id}/annotated` (w_300,h_300) | WebP/JPEG | CDN-transformed thumbnail for farmer dashboard and timelines |

---

## 4. Programmatic API

The storage adapter is accessed via `src/phytotrace/storage`:

```python
from phytotrace.storage import get_storage

storage = get_storage()

# 1. Upload bytes
url = storage.upload_bytes(jpeg_bytes, "inspections/INSP_001/annotated.jpg", content_type="image/jpeg")

# 2. Get dynamic thumbnail URL
thumb_url = storage.get_thumbnail_url("inspections/INSP_001/annotated", width=300, height=300)

# 3. Check existence
exists = storage.exists("inspections/INSP_001/annotated")

# 4. Delete asset
storage.delete("inspections/INSP_001/annotated")
```

---

## 5. Verification Command

Verify live Cloudinary connectivity and upload/delete functionality:

```bash
.venv/bin/python scripts/test_connections.py
```

Expected output:
```
[PASS] MongoDB Atlas: Read/Write/Delete verified successfully.
[PASS] Cloudinary: Upload/Retrieve/Delete verified successfully.
```

