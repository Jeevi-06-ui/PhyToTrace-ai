"""PhytoTrace AI — Cloudinary Asset Storage Adapter.

Manages original leaf photographs, AI segmentation masks, overlays, and thumbnails
using Cloudinary with secure server-side signed uploads, private asset access, and deletion.
"""

import io
import os
from time import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.request import urlopen

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.storage.cloudinary")


class CloudinaryStorage:
    """Enterprise Cloudinary storage adapter with signed uploads, URL delivery, and cleanup."""

    def __init__(
        self,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        folder: Optional[str] = None,
    ):
        self.cloud_name = cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME", "")
        self.api_key = api_key or os.getenv("CLOUDINARY_API_KEY", "")
        self.api_secret = api_secret or os.getenv("CLOUDINARY_API_SECRET", "")
        self.folder = (folder or os.getenv("CLOUDINARY_FOLDER", "phytotrace")).strip("/")

        self.is_active = False
        self._local_fallback_dir = resolve_path("data/storage")

        if os.getenv("PHYTOTRACE_ROTATED_CLOUD_CREDENTIALS", "1") == "1" and self.cloud_name and self.api_key and self.api_secret:
            try:
                import cloudinary
                cloudinary.config(
                    cloud_name=self.cloud_name,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    secure=True,
                )
                self.is_active = True
                logger.info("Cloudinary storage provider initialized (cloud: %s, folder: %s).", self.cloud_name, self.folder)
            except Exception as e:
                logger.error("Failed to initialize Cloudinary: %s", e)
                self.is_active = False
        else:
            logger.info("Cloudinary credentials not provided. Operating in local fallback mode.")

    def _format_public_id(self, object_key: str) -> str:
        """Strip leading slashes and file extensions for Cloudinary public_id conventions."""
        clean = object_key.lstrip("/")
        # If folder prefix is already present, avoid duplicating
        if not clean.startswith(self.folder):
            clean = f"{self.folder}/{clean}"
        # Cloudinary prefers public_id without standard extensions for image delivery
        stem, _ = os.path.splitext(clean)
        return stem

    def upload_private(self, data: bytes, object_key: str) -> None:
        import cloudinary.uploader

        cloudinary.uploader.upload(
            data, public_id=self._format_public_id(object_key), overwrite=False,
            resource_type="image", type="authenticated",
        )

    def private_url(self, object_key: str) -> str:
        from cloudinary.utils import private_download_url

        return private_download_url(
            self._format_public_id(object_key), Path(object_key).suffix.lstrip("."),
            type="authenticated", resource_type="image", expires_at=int(time()) + 600,
        )

    def delete_private(self, object_key: str) -> bool:
        import cloudinary.uploader

        try:
            result = cloudinary.uploader.destroy(
                self._format_public_id(object_key), resource_type="image", type="authenticated",
            )
            return result.get("result") in {"ok", "not found"}
        except Exception:
            logger.warning("Could not delete an authenticated Cloudinary asset")
            return False

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "image/jpeg",
        sign_url: bool = False,
    ) -> str:
        """Upload raw image bytes to Cloudinary or local fallback."""
        if not self.is_active:
            if os.getenv("PHYTOTRACE_ENV") == "production":
                raise RuntimeError("Cloudinary is unconfigured or unavailable in production environment.")
            # Local fallback
            dest = self._local_fallback_dir / object_key.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return f"/api/storage/{object_key.lstrip('/')}"

        import cloudinary.uploader
        import cloudinary.utils

        public_id = self._format_public_id(object_key)
        try:
            res = cloudinary.uploader.upload(
                data,
                public_id=public_id,
                overwrite=True,
                resource_type="image",
            )
            if sign_url:
                url, _ = cloudinary.utils.cloudinary_url(
                    public_id,
                    sign_url=True,
                    secure=True,
                )
                return url
            return res.get("secure_url", res.get("url", ""))
        except Exception as e:
            logger.error("Cloudinary upload_bytes error for '%s': %s", public_id, e)
            if os.getenv("PHYTOTRACE_ENV") == "production":
                raise
            # Fallback to local
            dest = self._local_fallback_dir / object_key.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return f"/api/storage/{object_key.lstrip('/')}"

    def upload_file(
        self,
        file_path: Union[str, Path],
        object_key: str,
        content_type: str = "image/jpeg",
        sign_url: bool = False,
    ) -> str:
        """Upload a file on disk to Cloudinary."""
        p = resolve_path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist.")
        with open(p, "rb") as f:
            data = f.read()
        return self.upload_bytes(data, object_key, content_type=content_type, sign_url=sign_url)

    def get_url(
        self,
        object_key: str,
        signed: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
        crop: Optional[str] = None,
    ) -> str:
        """Generate delivery URL, optionally with CDN transformation or signature."""
        if not self.is_active:
            return f"/api/storage/{object_key.lstrip('/')}"

        import cloudinary.utils
        public_id = self._format_public_id(object_key)
        options: Dict[str, Any] = {"secure": True, "sign_url": signed}
        if width:
            options["width"] = width
        if height:
            options["height"] = height
        if crop:
            options["crop"] = crop

        url, _ = cloudinary.utils.cloudinary_url(public_id, **options)
        return url

    def get_thumbnail_url(self, object_key: str, width: int = 300, height: int = 300) -> str:
        """Generate an optimized CDN thumbnail URL."""
        return self.get_url(object_key, signed=False, width=width, height=height, crop="fill")

    def exists(self, object_key: str) -> bool:
        """Check if asset exists in Cloudinary or local storage."""
        if not self.is_active:
            return (self._local_fallback_dir / object_key.lstrip("/")).is_file()

        import cloudinary.api
        public_id = self._format_public_id(object_key)
        try:
            cloudinary.api.resource(public_id, resource_type="image")
            return True
        except Exception:
            return False

    def download_bytes(self, object_key: str) -> bytes:
        """Download asset bytes from Cloudinary or local fallback."""
        if not self.is_active:
            p = self._local_fallback_dir / object_key.lstrip("/")
            if not p.is_file():
                raise FileNotFoundError(f"Asset '{object_key}' not found locally.")
            with open(p, "rb") as f:
                return f.read()

        url = self.get_url(object_key)
        with urlopen(url) as resp:
            return resp.read()

    def delete(self, object_key: str) -> bool:
        """Delete an asset from Cloudinary."""
        if not self.is_active:
            p = self._local_fallback_dir / object_key.lstrip("/")
            if p.is_file():
                p.unlink()
                return True
            return False

        import cloudinary.uploader
        public_id = self._format_public_id(object_key)
        try:
            res = cloudinary.uploader.destroy(public_id, resource_type="image")
            return res.get("result") in ["ok", "not found"]
        except Exception as e:
            logger.error("Cloudinary delete error for '%s': %s", public_id, e)
            return False


# Global singleton
_cloudinary_instance: Optional[CloudinaryStorage] = None


def get_cloudinary_storage() -> CloudinaryStorage:
    global _cloudinary_instance
    if _cloudinary_instance is None:
        _cloudinary_instance = CloudinaryStorage()
    return _cloudinary_instance

