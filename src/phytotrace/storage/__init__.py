"""PhytoTrace AI — Unified Asset Storage Interface.

Supports Cloudinary (default primary), Cloudflare R2, and local fallback.
"""

import os
from typing import Any, Union
from phytotrace.storage.cloudinary_storage import CloudinaryStorage, get_cloudinary_storage
from phytotrace.storage.r2 import CloudflareR2Storage, get_storage as get_r2_storage

_primary_storage = None


def get_storage():
    """Retrieve the active primary storage adapter (Cloudinary or Cloudflare R2)."""
    global _primary_storage
    if _primary_storage is not None:
        return _primary_storage

    provider = os.getenv("STORAGE_PROVIDER", "").lower()
    
    # If explicitly set to r2, or if R2 configured and Cloudinary not
    if provider == "r2" or (os.getenv("R2_ACCOUNT_ID") and not os.getenv("CLOUDINARY_CLOUD_NAME")):
        _primary_storage = get_r2_storage()
    else:
        # Default to Cloudinary
        _primary_storage = get_cloudinary_storage()

    return _primary_storage

