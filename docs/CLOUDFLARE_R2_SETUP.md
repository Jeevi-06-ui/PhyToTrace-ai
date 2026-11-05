# PhytoTrace AI — Cloudflare R2 Asset Storage Setup Guide

This document describes how to configure Cloudflare R2 for storing high-resolution leaf photographs, U-Net binary masks, and YOLOv8 visual evidence overlays with zero egress fees.

---

## 1. Storage Architecture Overview

PhytoTrace AI implements `CloudflareR2Storage` (`phytotrace.storage.r2`), which offers an S3-compatible cloud storage adapter paired with an automatic local filesystem fallback:

```
                       ┌──────────────────────────────────────────────┐
                       │          Leaf Photograph Upload              │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │       CloudflareR2Storage       │
                             │     (phytotrace.storage.r2)     │
                             └───────┬─────────────────┬───────┘
                                     │                 │
              ┌──────────────────────┘                 └──────────────────────┐
              ▼                                                               ▼
   [ Cloudflare R2 Bucket ]                                      [ Local Disk Storage ]
   - Zero egress fees worldwide                                  - Path: data/storage/
   - S3-compatible API                                           - Automatic local development
   - Custom domain / Pre-signed URLs                              - Offline fallover
```

---

## 2. Cloudflare R2 Provisioning Steps

### Step 2.1: Create an R2 Bucket
1. Log in to the [Cloudflare Dashboard](https://dash.cloudflare.com/).
2. In the left sidebar, click **R2**.
3. Click **Create bucket**.
4. Set **Bucket name**: `phytotrace-assets`.
5. Location: **Automatic** (or select Asia / India region for minimal latency).
6. Click **Create bucket**.

### Step 2.2: Generate S3-Compatible API Tokens
1. On the R2 Overview page, click **Manage R2 API Tokens** in the right sidebar.
2. Click **Create API token**.
3. Name: `PhytoTrace Production Token`.
4. Permissions: **Object Read & Write**.
5. Specify bucket: Select `phytotrace-assets`.
6. TTL: As desired (default forever or rotate annually).
7. Click **Create API Token**.
8. Cloudflare will display:
   - **Access Key ID**
   - **Secret Access Key**
   - **Endpoint**: `https://<account-id>.r2.cloudflarestorage.com`
9. Copy these values into your `.env` file.

### Step 2.3: Configure CORS (Cross-Origin Resource Sharing)
To allow web browsers to load overlays directly:
1. In your `phytotrace-assets` bucket, go to **Settings** -> **CORS Policy**.
2. Add the following JSON rule:
```json
[
  {
    "AllowedOrigins": [
      "http://localhost:3000",
      "https://phytotrace.ai",
      "https://*.vercel.app"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

---

## 3. Environment Configuration

Add the credentials to your root `.env`:

```bash
# Cloudflare R2 Configuration
R2_ACCOUNT_ID="your_cloudflare_account_id_hex"
R2_ACCESS_KEY_ID="your_r2_access_key_id"
R2_SECRET_ACCESS_KEY="your_r2_secret_access_key"
R2_BUCKET_NAME="phytotrace-assets"
R2_PUBLIC_DOMAIN="assets.phytotrace.ai" # Optional custom domain, or leave blank for pre-signed URLs
```

---

## 4. Local Filesystem Fallback

If R2 credentials are omitted or invalid:
- All assets are automatically written to `data/storage/`.
- URLs are generated as `/api/storage/<relative_path>`.
- The FastAPI application serves these assets locally via static file routing.

---

## 5. Verification Checklist

Run the unit tests:

```bash
.venv/bin/pytest tests/test_api_v1_and_services.py -k "test_cloudflare_r2"
```

Expected output:
```
tests/test_api_v1_and_services.py::test_cloudflare_r2_local_fallback PASSED
```

