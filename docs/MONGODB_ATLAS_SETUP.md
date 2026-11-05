# PhytoTrace AI — MongoDB Atlas Setup & Database Architecture Guide

This document describes the database design, MongoDB Atlas cluster provisioning, index definitions, and the resilient SQLite fallback mechanism implemented in PhytoTrace AI.

---

## 1. Dual-Engine Database Architecture

PhytoTrace AI implements an active dual-adapter database pattern (`phytotrace.db.mongo.PhytoTraceDatabase`):

```
                       ┌──────────────────────────────────────────────┐
                       │            PhytoTrace API v1                 │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │       PhytoTraceDatabase        │
                             │   (phytotrace.db.mongo)         │
                             └───────┬─────────────────┬───────┘
                                     │                 │
              ┌──────────────────────┘                 └──────────────────────┐
              ▼                                                               ▼
   [ MongoDB Atlas (Cloud) ]                                       [ SQLite 3 (Local) ]
   - Multi-tenant cloud persistence                                - Zero-dependency local file
   - Collections: inspections,                                     - Location: data/phytotrace.db
     evidence_regions, weather,                                    - High-integrity transactions
     assistant_sessions, users                                     - Automated offline fallback
```

### Core Invariants:
1. **Always Resilient:** If `MONGODB_URI` is empty, incorrect, or blocked by network/firewall policies, PhytoTrace silently logs a warning and routes all read and write queries to local SQLite without crashing.
2. **Deterministic Schemas:** Documents stored in MongoDB Atlas mirror relational columns in SQLite, allowing bidirectional synchronization or auditing.

---

## 2. MongoDB Atlas Cluster Provisioning

### Step 2.1: Create Free Tier (M0) Cluster
1. Log in to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Click **Create** -> Choose **M0 (Shared Cluster)** (Free).
3. Select Cloud Provider (AWS, Google Cloud, or Azure) and Region (e.g., `Mumbai (ap-south-1)` for lowest latency in India).
4. Cluster Name: `phytotrace-cluster`.
5. Click **Create Deployment**.

### Step 2.2: Configure Database User & Security
1. Under **Database Access**:
   - Click **Add New Database User**.
   - Authentication Method: **Password**.
   - Username: `phytotrace_admin`.
   - Autogenerate a secure password or specify one.
   - Database User Privileges: `Read and write to any database` (or restrict to `phytotrace` database).
   - Click **Add User**.
2. Under **Network Access**:
   - Click **Add IP Address**.
   - For local development: Click **Add Current IP Address** or `0.0.0.0/0` (Allow Access from Anywhere) with a temporary expiry.
   - For Cloud Run / Vercel production: Add your server's static egress IP or VPC Peering block.
   - Click **Confirm**.

### Step 2.3: Retrieve Connection String
1. On the Clusters Overview page, click **Connect**.
2. Choose **Drivers** (Python 3.12+).
3. Copy the SRV connection string:
   ```
   mongodb+srv://phytotrace_admin:<password>@phytotrace-cluster.abcde.mongodb.net/?retryWrites=true&w=majority
   ```
4. Replace `<password>` with your actual database user password.

---

## 3. Environment Variable Configuration

In your root `.env`:

```bash
# MongoDB Atlas Connection URI
MONGODB_URI="mongodb+srv://phytotrace_admin:YourSecretPassword@phytotrace-cluster.abcde.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DB_NAME="phytotrace"
```

---

## 4. Collection Schemas & Indexing

The application automatically verifies and builds the required indexes on initial startup in `_ensure_indexes()`:

### Collection 1: `users`
Stores user profile, agricultural affiliation, and language preferences.
- **Indexes:**
  - `firebase_uid` (Unique)

### Collection 2: `farms`
Stores greenhouse and field plot identifiers.
- **Indexes:**
  - Compound: `[("owner_uid", 1), ("farm_id", 1)]`

### Collection 3: `plants`
Tracks individual monitored tomato plants and longitudinal progression.
- **Indexes:**
  - Compound: `[("farm_id", 1), ("plant_id", 1)]`

### Collection 4: `inspections`
Stores core inspection records, U-Net leaf area, YOLOv8 mine detection results, and severity tiers.
- **Fields:**
  - `id`: Inspection unique identifier (e.g., `INSP_20261105_120000_12345_PL001`)
  - `owner_uid`: Firebase UID or guest session ID
  - `farm_id`, `plant_id`, `leaf_id`
  - `quality_status`: `acceptable`, `blurry`, `glare_detected`
  - `leaf_area_pixels`: Integer pixel count of U-Net segmented lamina
  - `damage_detected`: Boolean
  - `damage_area_pixels`: Integer pixel count of YOLOv8 mine lesions inside leaf
  - `damage_percentage`: $100 \times \text{Damage} / \text{Leaf}$
  - `severity_tier`: `none`, `trace`, `low`, `moderate`, `severe`
  - `decision`: `healthy`, `mine_detected_species_uncertain`, etc.
  - `explanation`: Contextual diagnostic sentence
  - `created_at`: ISO 8601 UTC timestamp
- **Indexes:**
  - Compound: `[("owner_uid", 1), ("created_at", -1)]`
  - Compound: `[("plant_id", 1), ("created_at", -1)]`

### Collection 5: `evidence_regions`
Stores individual localized damage lesions (Regions A, B, C...) sorted deterministically.
- **Fields:**
  - `inspection_id`: Parent inspection reference
  - `region_id`: Deterministic integer index (0, 1, 2...)
  - `label`: Human-readable letter (`Region A`, `Region B`...)
  - `bbox`: `[ymin, xmin, ymax, xmax]`
  - `area_pixels`: Lesion area in pixels
  - `centroid`: `[cx, cy]`
- **Indexes:**
  - Compound: `[("inspection_id", 1), ("label", 1)]`

### Collection 6: `weather_snapshots`
Records local microclimate telemetry associated with scouting sessions.
- **Indexes:**
  - Compound: `[("location", 1), ("timestamp", -1)]`

### Collection 7: `assistant_sessions`
Stores chat history with PhytoTrace Agri Assistant.
- **Indexes:**
  - `session_id` (Unique)

---

## 5. Verification & Testing

Run the automated database test suite:

```bash
.venv/bin/pytest tests/test_api_v1_and_services.py -k "test_database"
```

Expected output:
```
tests/test_api_v1_and_services.py::test_database_sqlite_fallback PASSED
```

