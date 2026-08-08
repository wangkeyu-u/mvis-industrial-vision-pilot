"""In-memory image security validation; user filenames never become paths."""

from __future__ import annotations

import base64
import binascii
import io
import math
import struct
import warnings

from fastapi import UploadFile
from PIL import Image

from src.core.config import ServiceSettings
from src.core.errors import ErrorCode, ServiceError
from src.core.schemas import ImageMetadata

MIME_TO_FORMAT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


def detect_signature(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


async def validate_upload(
    upload: UploadFile, settings: ServiceSettings
) -> tuple[bytes, ImageMetadata]:
    content_type = upload.content_type or ""
    declared_format = MIME_TO_FORMAT.get(content_type.lower())
    if declared_format is None:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "image MIME type must be image/jpeg, image/png, or image/webp",
        )

    data = bytearray()
    try:
        while chunk := await upload.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > settings.max_image_bytes:
                raise ServiceError(
                    ErrorCode.INVALID_IMAGE,
                    f"image exceeds the {settings.max_image_bytes}-byte limit",
                )
    finally:
        await upload.close()

    return validate_image_bytes(bytes(data), content_type, settings)


def decode_image_data_url(value: str, settings: ServiceSettings) -> tuple[bytes, str]:
    prefix, separator, encoded = value.partition(",")
    if separator != "," or not prefix.endswith(";base64"):
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "JSON image must be a base64-encoded data URL",
        )
    content_type = prefix.removeprefix("data:").removesuffix(";base64").lower()
    if content_type not in MIME_TO_FORMAT:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "data URL MIME type must be image/jpeg, image/png, or image/webp",
        )
    max_encoded = 4 * math.ceil(settings.max_image_bytes / 3)
    if len(encoded) > max_encoded:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            f"image exceeds the {settings.max_image_bytes}-byte limit",
        )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE, "image base64 payload is invalid"
        ) from exc
    return payload, content_type


def validate_image_bytes(
    payload: bytes, content_type: str, settings: ServiceSettings
) -> tuple[bytes, ImageMetadata]:
    declared_format = MIME_TO_FORMAT.get(content_type.lower())
    if declared_format is None:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "image MIME type must be image/jpeg, image/png, or image/webp",
        )
    if not payload:
        raise ServiceError(ErrorCode.INVALID_IMAGE, "image file is empty")
    if len(payload) > settings.max_image_bytes:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            f"image exceeds the {settings.max_image_bytes}-byte limit",
        )

    signature_format = detect_signature(payload)
    if signature_format is None or signature_format != declared_format:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "image signature does not match its declared MIME type",
        )
    if not _has_exact_container_termination(payload, signature_format):
        raise ServiceError(
            ErrorCode.INVALID_IMAGE,
            "image container has trailing or inconsistent data",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                detected_format = (image.format or "").lower()
                width, height = image.size
                mode = image.mode
                frames = getattr(image, "n_frames", 1)
                image.verify()
            if detected_format == "jpg":
                detected_format = "jpeg"
            if detected_format != signature_format:
                raise ValueError("decoder format differs from signature")
            if frames != 1:
                raise ValueError("animated or multi-frame images are not supported")
            if (
                width > settings.max_image_dimension
                or height > settings.max_image_dimension
            ):
                raise ServiceError(
                    ErrorCode.INVALID_IMAGE,
                    f"image dimensions exceed {settings.max_image_dimension}px",
                )
            if width * height > settings.max_image_pixels:
                raise ServiceError(
                    ErrorCode.INVALID_IMAGE,
                    f"image exceeds the {settings.max_image_pixels}-pixel limit",
                )
            # verify() checks container integrity but does not decode pixels. Reopen and
            # force decoding so corrupt compressed data cannot reach a model adapter.
            with Image.open(io.BytesIO(payload)) as decoded:
                decoded.load()
    except ServiceError:
        raise
    except (
        Image.UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE, "image content cannot be safely decoded"
        ) from exc
    except Image.DecompressionBombWarning as exc:
        raise ServiceError(
            ErrorCode.INVALID_IMAGE, "image exceeds safe decoder limits"
        ) from exc

    return payload, ImageMetadata(
        width=width,
        height=height,
        format=signature_format,
        mode=mode,
        byte_size=len(payload),
    )


def _has_exact_container_termination(payload: bytes, image_format: str) -> bool:
    """Reject common image/polyglot payloads with bytes after the image container."""

    if image_format == "jpeg":
        return payload.endswith(b"\xff\xd9")
    if image_format == "png":
        return payload.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82")
    if image_format == "webp":
        if len(payload) < 12:
            return False
        declared_size = struct.unpack("<I", payload[4:8])[0] + 8
        return declared_size == len(payload)
    return False
