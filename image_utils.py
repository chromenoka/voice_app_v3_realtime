"""Validation and normalization for one session-scoped image attachment."""

from __future__ import annotations

import base64
import io

from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_and_normalize_image(data_url: str) -> str:
    """Return a normalized JPEG data URL, rejecting malformed or oversized files."""
    try:
        header, encoded = data_url.split(",", 1)
        mime_type = header.removeprefix("data:").removesuffix(";base64").lower()
        if mime_type not in ALLOWED_IMAGE_MIME_TYPES or not header.endswith(";base64"):
            raise ValueError("unsupported image type")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 5 MB")
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            normalized = image.convert("RGB")
            normalized.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
            output = io.BytesIO()
            normalized.save(output, format="JPEG", quality=85, optimize=True)
    except (ValueError, UnidentifiedImageError, OSError, base64.binascii.Error) as exc:
        raise ValueError("invalid image upload") from exc

    encoded_normalized = base64.b64encode(output.getvalue()).decode("ascii")
