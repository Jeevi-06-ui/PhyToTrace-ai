"""PhytoTrace AI — Firebase Authentication & User Identity Management.

Provides FastAPI dependencies for verifying Firebase ID tokens,
handling guest sessions, and enforcing record ownership.
Supports both Firebase Admin SDK (with service account) and
public Google x509 certificate verification via PyJWT (zero-config project mode).
"""

import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional
from cryptography.x509 import load_pem_x509_certificate
from fastapi import Depends, Header, HTTPException
import jwt
from pydantic import BaseModel

from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.auth")

_firebase_admin_initialized = False
_CERT_CACHE: Dict[str, Any] = {"certs": {}, "expires_at": 0}
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


def _get_project_id() -> str:
    return os.getenv("FIREBASE_PROJECT_ID", "phytotrace")


def _init_firebase_admin() -> bool:
    global _firebase_admin_initialized
    if _firebase_admin_initialized:
        return True

    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

    try:
        import firebase_admin
        from firebase_admin import credentials

        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            _firebase_admin_initialized = True
            logger.info("Firebase Admin SDK initialized with service account certificate: %s", cred_path)
            return True
    except Exception as e:
        logger.warning("Could not initialize Firebase Admin SDK with service account: %s", e)

    return False


def _get_google_public_key(kid: str):
    """Retrieve Google's public x509 cert for verifying Firebase RS256 ID tokens."""
    now = time.time()
    if not _CERT_CACHE["certs"] or now > _CERT_CACHE["expires_at"]:
        try:
            req = urllib.request.Request(GOOGLE_CERTS_URL, headers={"User-Agent": "PhytoTrace-Auth/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cache_control = resp.headers.get("Cache-Control", "")
                max_age = 19000
                for part in cache_control.split(","):
                    if "max-age=" in part:
                        try:
                            max_age = int(part.split("=")[1].strip())
                        except Exception:
                            pass
                _CERT_CACHE["certs"] = data
                _CERT_CACHE["expires_at"] = now + max_age
        except Exception as e:
            logger.error("Failed to fetch Google public certs: %s", e)
            if not _CERT_CACHE["certs"]:
                raise HTTPException(status_code=503, detail="Unable to fetch Google public certificates.") from e

    certs = _CERT_CACHE["certs"]
    if kid not in certs:
        raise HTTPException(status_code=401, detail=f"Firebase ID token key ID '{kid}' not found in Google certs.")

    cert_pem = certs[kid]
    cert = load_pem_x509_certificate(cert_pem.encode("utf-8"))
    return cert.public_key()


def verify_firebase_token(token: str) -> Dict[str, Any]:
    """Verify a Firebase ID token via Firebase Admin (if configured) or Google public certs."""
    # Strategy 1: Firebase Admin SDK with Service Account
    if _init_firebase_admin():
        try:
            from firebase_admin import auth
            return auth.verify_id_token(token)
        except Exception as exc:
            logger.warning("Firebase Admin token verification failed: %s", exc)
            raise HTTPException(status_code=401, detail="Invalid or expired Firebase authentication token.") from exc

    # Strategy 2: Direct RS256 Verification against Google public certificates
    project_id = _get_project_id()
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        alg = unverified_header.get("alg")
        if alg != "RS256" or not kid:
            raise HTTPException(status_code=401, detail="Invalid token header (RS256 algorithm and kid required).")

        public_key = _get_google_public_key(kid)
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=project_id,
            issuer=f"https://securetoken.google.com/{project_id}",
            options={"require": ["exp", "iat", "aud", "iss", "sub"]},
        )
        # uid in Firebase token is the subject 'sub'
        decoded["uid"] = decoded.get("user_id") or decoded.get("sub")
        return decoded
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Firebase authentication token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid Firebase token: %s", exc)
        raise HTTPException(status_code=401, detail=f"Invalid Firebase authentication token: {exc}") from exc


class UserContext(BaseModel):
    uid: str
    email: Optional[str] = None
    display_name: str = "Guest Agronomist"
    is_guest: bool = True
    role: str = "farmer"


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> UserContext:
    """Dependency that decodes and validates Firebase ID tokens or returns a guest context."""
    if not authorization:
        return UserContext(uid="guest")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Firebase bearer token required.")

    decoded = verify_firebase_token(token)
    return UserContext(
        uid=decoded["uid"],
        email=decoded.get("email"),
        display_name=decoded.get("name") or decoded.get("email") or "Farmer",
        is_guest=False,
        role="authenticated_user",
    )


async def require_authenticated_user(user: UserContext = Depends(get_current_user)) -> UserContext:
    if user.is_guest:
        raise HTTPException(status_code=401, detail="Sign in to access saved records.")
    return user
