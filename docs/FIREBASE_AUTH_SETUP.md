# PhytoTrace AI — Firebase Authentication Setup Guide

This document provides complete instructions for configuring Firebase Authentication for PhytoTrace AI, including Google Sign-In, Firebase Admin SDK verification on the FastAPI backend, and seamless Guest Mode operation.

---

## 1. Architecture Overview

PhytoTrace AI implements a **dual-mode authentication architecture**:

```
                       ┌──────────────────────────────────────────────┐
                       │               Frontend User                  │
                       └──────────────────────┬───────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
          [ Google Sign-In ]                                   [ Guest Farmer Mode ]
       Firebase Auth Client SDK                              Local UUID in localStorage
     ID Token: Bearer <JWT>                                Header: X-Guest-Session-ID
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                             ┌─────────────────────────────────┐
                             │    FastAPI Auth Middleware     │
                             │   (phytotrace.auth.firebase)    │
                             └────────────────┬────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ Firebase Admin SDK ]                           [ Guest Context Generator ]
     Verify JWT with Google certs                        Generate transient guest uid:
      Extract: uid, email, name                           "guest_<uuid4>" (no barrier)
```

### Core Design Principles:
1. **Zero Login Barrier for Scanning:** A farmer in the field must be able to upload a photo and receive real U-Net + YOLOv8 segmentation results immediately without signing in or registering.
2. **Seamless Upgrade Path:** Guest inspections are associated with the guest session ID (`X-Guest-Session-ID`). When the user later signs in with Google, inspections can be linked to their permanent account.
3. **Stateless Backend Verification:** The FastAPI backend verifies Firebase JWTs cryptographically using the Google public keys via `firebase-admin`. No session state is held in memory.

---

## 2. Firebase Console Configuration

### Step 2.1: Create Firebase Project
1. Navigate to the [Firebase Console](https://console.firebase.google.com/).
2. Click **Add Project** and enter project name: `phytotrace-ai`.
3. Disable Google Analytics (optional, or configure as desired) and click **Create Project**.

### Step 2.2: Enable Authentication Providers
1. In the left navigation, click **Build** -> **Authentication**.
2. Click **Get Started**.
3. Under the **Sign-in method** tab:
   - Click **Google**.
   - Toggle **Enable**.
   - Set the **Project public-facing name**: `PhytoTrace AI`.
   - Select your project support email.
   - Click **Save**.
4. (Optional) Under **Authorized domains**, ensure `localhost`, `127.0.0.1`, and your production domain (e.g., `phytotrace.ai` or Vercel URL) are listed.

### Step 2.3: Register Web App
1. In the Project Overview page, click the **Web icon (`</>`)** to add an app.
2. App nickname: `PhytoTrace Web`.
3. Check or leave unchecked "Firebase Hosting".
4. Click **Register app**.
5. Copy the `firebaseConfig` keys into your frontend environment.

---

## 3. Environment Configuration

### Frontend Configuration (`frontend/.env.local`)
Create or edit `frontend/.env.local`:

```bash
NEXT_PUBLIC_FIREBASE_API_KEY="AIzaSyYourActualApiKeyHere"
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN="phytotrace-ai.firebaseapp.com"
NEXT_PUBLIC_FIREBASE_PROJECT_ID="phytotrace-ai"
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET="phytotrace-ai.appspot.com"
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID="123456789012"
NEXT_PUBLIC_FIREBASE_APP_ID="1:123456789012:web:abcdef1234567890"
NEXT_PUBLIC_API_URL="http://localhost:8000"
```

### Backend Configuration (`.env`)
1. In Firebase Console, go to **Project Settings** (gear icon) -> **Service Accounts**.
2. Click **Generate New Private Key** -> Confirm **Generate Key**.
3. Save the downloaded JSON file securely to:
   ```
   configs/firebase-service-account.json
   ```
4. Set the path in your root `.env`:
   ```bash
   FIREBASE_SERVICE_ACCOUNT_JSON=configs/firebase-service-account.json
   ```

> [!CAUTION]
> Never commit `configs/firebase-service-account.json` to Git. This file is already ignored by `.gitignore`.

---

## 4. Backend Implementation Details

The backend authentication logic is encapsulated in `src/phytotrace/auth/firebase.py`.

### Dependency Injection in FastAPI Endpoints:
```python
from fastapi import APIRouter, Depends
from phytotrace.auth.firebase import UserContext, get_current_user

router = APIRouter()

@router.get("/api/v1/inspections")
def list_inspections(user: UserContext = Depends(get_current_user)):
    # user.uid is either authenticated Google UID or "guest_<session_id>"
    # user.is_guest indicates whether login has occurred
    return {"user_id": user.uid, "is_guest": user.is_guest}
```

### Verification Flow:
1. If the request provides `Authorization: Bearer <token>`, the token is verified using `firebase_admin.auth.verify_id_token()`.
2. If the token is valid, `UserContext` is returned with `is_guest=False`, `email`, and `display_name`.
3. If no Authorization header is provided, the middleware checks for `X-Guest-Session-ID`.
4. If present, it creates a `UserContext` with `uid=f"guest_{session_id}"` and `is_guest=True`.
5. If neither is provided, an ephemeral guest session is automatically assigned.

---

## 5. Verification Checklist

| Test Item | Command / Procedure | Expected Result |
| :--- | :--- | :--- |
| **Backend Auth Unit Tests** | `.venv/bin/pytest tests/test_api_v1_and_services.py` | `test_v1_user_profile_guest PASSED` |
| **Guest Profile API** | `curl -H "X-Guest-Session-ID: test123" http://localhost:8000/api/v1/users/me` | Returns JSON with `"is_guest": true` |
| **Protected Analysis API** | `curl -X POST http://localhost:8000/api/v1/analyze` (with image) | Processes inference under guest session |
| **Frontend Google Sign-In** | Click "Sign In with Google" on header | Firebase popup opens and persists user session |

