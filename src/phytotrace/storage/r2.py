"""PhytoTrace AI — Cloudflare R2 & Local Filesystem Resilient Asset Storage.

Stores original leaf photographs, U-Net masks, and YOLOv8 evidence overlays.
Uses Cloudflare R2 (S3-compatible) when configured, or local filesystem fallback.
"""

import os
from pathlib import Path
from typing import Optional, Union
from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.storage.r2")

LOCAL_STORAGE_DIR = "data/storage"


class CloudflareR2Storage:
    """Manages secure image asset storage on Cloudflare R2 with local fallback."""

    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "phytotrace-assets")
        self.public_domain = os.getenv("R2_PUBLIC_DOMAIN", "")

        self.s3_client = None
        self.is_r2_active = False

        self.local_dir = resolve_path(LOCAL_STORAGE_DIR)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        if self.account_id and self.access_key_id and self.secret_access_key:
            try:
                import boto3
                endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
                self.s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name="auto",
                )
                self.is_r2_active = True
                logger.info("Cloudflare R2 client initialized for bucket '%s'.", self.bucket_name)
            except Exception as e:
                logger.warning("Could not initialize Cloudflare R2 client (%s). Using local storage.", e)
                self.is_r2_active = False
        else:
            logger.info("Cloudflare R2 credentials not present. Using local storage at '%s'.", self.local_dir)

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload image bytes to R2 or save locally."""
        # Clean key
        clean_key = object_key.lstrip("/")

        if self.is_r2_active and self.s3_client is not None:
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=clean_key,
                    Body=data,
                    ContentType=content_type,
                )
                if self.public_domain:
                    return f"https://{self.public_domain}/{clean_key}"
                return self.generate_presigned_url(clean_key)
            except Exception as e:
                logger.error("R2 upload error for '%s': %s. Falling back to local.", clean_key, e)

        # Local storage fallback
        dest_path = self.local_dir / clean_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)

        return f"/api/storage/{clean_key}"

    def upload_file(
        self,
        file_path: Union[str, Path],
        object_key: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload a local file to R2 or copy to local storage."""
        p = resolve_path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist.")

        with open(p, "rb") as f:
            data = f.read()
        return self.upload_bytes(data, object_key, content_type)

    def generate_presigned_url(self, object_key: str, expiration_seconds: int = 3600) -> str:
        """Generate a time-limited signed URL for private asset access."""
        clean_key = object_key.lstrip("/")
        if self.is_r2_active and self.s3_client is not None:
            try:
                url = self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket_name, "Key": clean_key},
                    ExpiresIn=expiration_seconds,
                )
                return url
            except Exception as e:
                logger.error("Error generating pre-signed URL for '%s': %s", clean_key, e)

        return f"/api/storage/{clean_key}"

    def get_local_path(self, object_key: str) -> Optional[Path]:
        """Retrieve local filesystem path if available."""
        clean_key = object_key.lstrip("/")
        p = self.local_dir / clean_key
        if p.is_file():
            return p
        return None

    def exists(self, object_key: str) -> bool:
        """Check if asset exists in storage."""
        clean_key = object_key.lstrip("/")
        if self.is_r2_active and self.s3_client is not None:
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=clean_key)
                return True
            except Exception:
                return False
        return (self.local_dir / clean_key).is_file()

    def download_bytes(self, object_key: str) -> bytes:
        """Download asset bytes from storage."""
        clean_key = object_key.lstrip("/")
        if self.is_r2_active and self.s3_client is not None:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=clean_key)
            return obj["Body"].read()
        p = self.local_dir / clean_key
        if not p.is_file():
            raise FileNotFoundError(f"Asset '{clean_key}' not found.")
        with open(p, "rb") as f:
            return f.read()

    def get_url(self, object_key: str) -> str:
        """Get accessible URL for the asset."""
        clean_key = object_key.lstrip("/")
        if self.is_r2_active and self.public_domain:
            return f"https://{self.public_domain}/{clean_key}"
        return self.generate_presigned_url(clean_key)


# Global singleton
_storage_instance: Optional[CloudflareR2Storage] = None


def get_storage() -> CloudflareR2Storage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = CloudflareR2Storage()
    return _storage_instance
