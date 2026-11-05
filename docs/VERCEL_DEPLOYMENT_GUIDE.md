# PhytoTrace — Vercel Deployment & Cloud Production Guide

This guide details how to deploy the **PhytoTrace Next.js Frontend** to Vercel so that farmers, agronomists, and researchers worldwide can access the platform, while connecting securely to the FastAPI inference engine.

---

## 1. Architecture Overview

```
                                  ┌──────────────────────────────┐
                                  │      Client Web Browser      │
                                  │   (Farmers & Agronomists)    │
                                  └──────────────┬───────────────┘
                                                 │
                                 HTTPS Requests  │
                                                 ▼
               ┌─────────────────────────────────────────────────────────────────┐
               │              Vercel Edge Global Network                         │
               │  • Hosted Next.js 16 Web Application                            │
               │  • Multilingual Localization (English, Kannada, Hindi)         │
               │  • Web Speech Audio Narration Engine (Web Speech API)           │
               │  • Firebase Client Auth SDK                                     │
               └──────────────┬───────────────────────────────────┬──────────────┘
                              │                                   │
              Client Auth     │                                   │ Next.js Rewrites:
              & Persistence   │                                   │ /api/:path* ->
                              ▼                                   ▼
               ┌─────────────────────────────┐   ┌───────────────────────────────┐
               │  Firebase & MongoDB Atlas   │   │     FastAPI Inference Host    │
               │  • Phone/Google Auth        │   │ (Railway / Render / AWS / GCP)│
               │  • Inspection Telemetry     │   │  • Dual-Stage Vision Pipeline │
               │  • Saved Farm Profiles      │   │  • Google Weather Intelligence│
               └─────────────────────────────┘   └───────────────────────────────┘
```

---

## 2. Quick Deploy via Vercel Dashboard (Recommended)

1. **Push your repository to GitHub** (master or main branch).
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Select your GitHub repository: `PhyToTrace`.
4. In the Project Configuration screen:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click *Edit* and select `frontend` (or leave default root as `vercel.json` will automatically build `frontend`).
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
5. **Add Environment Variables** in Vercel:
   | Variable | Value / Description | Example |
   | :--- | :--- | :--- |
   | `PHYTOTRACE_API_ORIGIN` | Your deployed FastAPI backend URL | `https://api.phytotrace.example.com` |
   | `NEXT_PUBLIC_FIREBASE_API_KEY` | Firebase Web API Key | `AIzaSy...` |
   | `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN` | Firebase Project Auth Domain | `phytotrace-prod.firebaseapp.com` |
   | `NEXT_PUBLIC_FIREBASE_PROJECT_ID` | Firebase Project ID | `phytotrace-prod` |
   | `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` | Firebase Storage Bucket | `phytotrace-prod.appspot.com` |
   | `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` | Cloud Messaging Sender ID | `1234567890` |
   | `NEXT_PUBLIC_FIREBASE_APP_ID` | Firebase App ID | `1:1234567890:web:abcd...` |
6. Click **Deploy**. Vercel will build and assign a global production domain (e.g., `https://phytotrace.vercel.app`).

---

## 3. Alternative: Deploy via Vercel CLI

From your local terminal:

```bash
# Navigate to the frontend directory
cd /Users/jeevithp/Desktop/PhyToTrace/frontend

# Install or run Vercel CLI
npx vercel

# Follow the interactive CLI prompts:
# ? Set up and deploy “~/Desktop/PhyToTrace/frontend”? [Y/n] Y
# ? Which scope do you want to deploy to? <Your Account>
# ? Link to existing project? [y/N] N
# ? What’s your project’s name? phytotrace
# ? In which directory is your code located? ./

# For production deployment:
npx vercel --prod
```

---

## 4. How the FastAPI Backend Connects

In [`frontend/next.config.ts`](file:///Users/jeevithp/Desktop/PhyToTrace/frontend/next.config.ts), Next.js includes server rewrites:

```typescript
async rewrites() {
  const backend = process.env.PHYTOTRACE_API_ORIGIN || "http://127.0.0.1:8000";
  return [
    {
      source: "/api/:path*",
      destination: `${backend}/api/:path*`,
    },
  ];
}
```

- When running locally, `PHYTOTRACE_API_ORIGIN` defaults to `http://127.0.0.1:8000`.
- In Vercel production, setting `PHYTOTRACE_API_ORIGIN` to your cloud backend (e.g. `https://api.phytotrace.example.com` on Railway or Render) ensures:
  1. No CORS preflight errors in modern browsers.
  2. All `/api/v1/scan`, `/api/v1/weather/*`, and `/api/health` requests seamlessly proxy.
  3. API keys (Google Weather, MongoDB connection string) stay private in the backend.

---

## 5. Security & Isolation Guarantee

- **Zero Disruption to Local Environment**: Deploying the frontend to Vercel does not touch or alter your local Apple Silicon MPS acceleration, PyTorch weights, or SQLite database.
- **Key Safety**: The Google Weather API key and MongoDB credentials are never bundled into the client build.
- **Web Speech Audio**: The rural voice narration feature runs entirely client-side using browser speech synthesis, requiring zero third-party audio hosting or extra cloud billing.

