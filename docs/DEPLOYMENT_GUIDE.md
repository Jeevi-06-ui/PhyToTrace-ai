# PhytoTrace AI — Production Deployment Guide

This guide covers complete deployment procedures for PhytoTrace AI across **Google Cloud Run (FastAPI Backend)**, **Vercel (Next.js Frontend)**, and **Docker Compose (Local/On-Premise Full-Stack)**.

---

## 1. Production Architecture Overview

```
                                  [ Internet / DNS ]
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
                     ▼                                         ▼
         ┌────────────────────────┐               ┌────────────────────────┐
         │     Next.js Frontend   │               │     FastAPI Backend    │
         │   Hosted on Vercel     │               │ Hosted on Cloud Run    │
         │   (https://phytotrace) │               │ (https://api.phytotrace)│
         └───────────┬────────────┘               └───────────┬────────────┘
                     │                                        │
                     │ API Requests                           │ Checkpoint Inference
                     └───────────────────►────────────────────┤
                                                              │
                     ┌───────────────────┬────────────────────┤
                     ▼                   ▼                    ▼
             [ MongoDB Atlas ]   [ Cloudflare R2 ]     [ Gemini 2.5 API ]
             Metadata & History  Images & Overlays     Agri Assistant
```

---

## 2. Deploying FastAPI Backend to Google Cloud Run

Google Cloud Run provides serverless container execution with auto-scaling to zero and cost-effective CPU/memory allocation.

### Step 2.1: Build Container with Google Cloud Build
From the project root:

```bash
# Set your GCP Project ID
export GCP_PROJECT="phytotrace-prod"

# Submit build using root Dockerfile
gcloud builds submit \
  --project $GCP_PROJECT \
  --tag gcr.io/$GCP_PROJECT/phytotrace-api:latest
```

### Step 2.2: Deploy Service on Cloud Run
```bash
gcloud run deploy phytotrace-api \
  --project $GCP_PROJECT \
  --image gcr.io/$GCP_PROJECT/phytotrace-api:latest \
  --platform managed \
  --region asia-south1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars "\
PHYTOTRACE_ENV=production,\
PHYTOTRACE_DEVICE=cpu,\
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true,\
MONGODB_DB_NAME=phytotrace,\
R2_ACCOUNT_ID=your_id,\
R2_ACCESS_KEY_ID=your_key,\
R2_SECRET_ACCESS_KEY=your_secret,\
R2_BUCKET_NAME=phytotrace-assets,\
GEMINI_API_KEY=your_gemini_key"
```

### Step 2.3: Verify Backend Deployment
```bash
curl https://phytotrace-api-xyz-as.a.run.app/api/v1/health
```
Expected output:
```json
{"status":"healthy","version":"PhytoTrace-v1.0.0-MVP","device":"cpu","models_active":{"unet_leaf_segmenter":true,"yolo_mine_detector":true,"mobilenet_baseline":true}}
```

---

## 3. Deploying Next.js Frontend to Vercel

### Step 3.1: Connect GitHub Repository
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New** -> **Project**.
3. Select the `PhyToTrace` repository.

### Step 3.2: Configure Project Settings
- **Framework Preset:** Next.js
- **Root Directory:** Click Edit and select `frontend`.
- **Build Command:** `npm run build`
- **Output Directory:** `.next`

### Step 3.3: Set Environment Variables in Vercel
Add the following in the Vercel dashboard:
- `NEXT_PUBLIC_API_URL`: `https://phytotrace-api-xyz-as.a.run.app` (Your Cloud Run backend URL)
- `NEXT_PUBLIC_FIREBASE_API_KEY`: Your Firebase Web API key
- `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`: `phytotrace-ai.firebaseapp.com`
- `NEXT_PUBLIC_FIREBASE_PROJECT_ID`: `phytotrace-ai`
- `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`: `phytotrace-ai.appspot.com`
- `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`: `123456789012`
- `NEXT_PUBLIC_FIREBASE_APP_ID`: `1:123456789012:web:abcdef1234567890`

Click **Deploy**.

---

## 4. Local / On-Premise Docker Compose Deployment

To run the complete stack locally with Docker:

```bash
# Build and start all services in detached mode
docker-compose up -d --build

# Inspect logs
docker-compose logs -f
```

Services exposed:
- **Frontend:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`

---

## 5. Security & Production Checklist

- [x] Preflight image quality validator active (rejects blur variance < 20.0).
- [x] CORS restricted to authorized frontend origins.
- [x] Secrets stored exclusively in environment variables, never hardcoded.
- [x] Real PyTorch weights packaged inside backend container image.
- [x] Automatic fallbacks for database (SQLite) and asset storage (local disk) active.

