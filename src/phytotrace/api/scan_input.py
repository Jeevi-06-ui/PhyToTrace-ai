"""Shared image and original-pixel crop validation for the two scan endpoints."""

from io import BytesIO
import json

import cv2
from fastapi import HTTPException, UploadFile
import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin, PngImagePlugin, WebPImagePlugin

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000


async def read_scan_upload(upload: UploadFile, crop_box: str | None):
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "File size exceeds maximum 25 MB limit.")
    try:
        # Use only the explicitly supported decoders. Ultralytics patches PIL.Image.open
        # to attempt optional HEIF dependency installation on unidentified data.
        if contents.startswith(b"\xff\xd8\xff"):
            decoder = JpegImagePlugin.JpegImageFile
        elif contents.startswith(b"\x89PNG\r\n\x1a\n"):
            decoder = PngImagePlugin.PngImageFile
        elif contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
            decoder = WebPImagePlugin.WebPImageFile
        else:
            raise ValueError("Unsupported image signature")
        with decoder(BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Unsupported image format or dimensions")
            oriented = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        bgr = cv2.cvtColor(oriented, cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, ValueError, OSError, SyntaxError) as exc:
        raise HTTPException(400, "Upload a valid JPEG, PNG or WebP image under 24 megapixels.") from exc

    if crop_box is None:
        return bgr, None
    try:
        parsed = json.loads(crop_box) if crop_box.lstrip().startswith("[") else [int(v) for v in crop_box.split(",")]
        if not isinstance(parsed, list) or len(parsed) != 4 or any(type(v) is not int for v in parsed):
            raise ValueError
        x1, y1, x2, y2 = parsed
        h, w = bgr.shape[:2]
        if not (0 <= x1 <= x2 - 32 and x2 <= w and 0 <= y1 <= y2 - 32 and y2 <= h):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Select a rectangle of at least 32 by 32 original pixels inside the photo.") from exc
    return bgr, parsed


def read_leaf_polygon(value: str | None, image: np.ndarray):
    if value is None:
        return None
    h, w = image.shape[:2]
    try:
        points = json.loads(value)
        if not isinstance(points, list) or not 3 <= len(points) <= 100:
            raise ValueError
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) is not int for v in point)
                    or not (0 <= point[0] < w and 0 <= point[1] < h)):
                raise ValueError
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError

        def side(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        edges = list(zip(points, points[1:] + points[:1]))
        for i, (a, b) in enumerate(edges):
            for j in range(i + 2, len(edges)):
                if i == 0 and j == len(edges) - 1:
                    continue
                c, d = edges[j]
                boxes_overlap = all(max(min(a[k], b[k]), min(c[k], d[k])) <=
                                    min(max(a[k], b[k]), max(c[k], d[k])) for k in (0, 1))
                if boxes_overlap and side(a, b, c) * side(a, b, d) <= 0 and side(c, d, a) * side(c, d, b) <= 0:
                    raise ValueError
        contour = np.asarray(points, dtype=np.int32)
        _, _, width, height = cv2.boundingRect(contour)
        if min(width, height) < 32 or cv2.contourArea(contour) < 1:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Draw a non-crossing outline with at least three distinct points inside the image.") from exc
    return points
