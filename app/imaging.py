"""Image normalization: orientation fix, downscale, re-encode."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps


class ImageProcessingError(ValueError):
    pass


@dataclass
class ProcessedImage:
    name: str
    width: int
    height: int
    size_bytes: int
    base64_data: str
    mime_type: str


def process_image(
    raw: bytes,
    *,
    name: str,
    max_side: int,
    jpeg_quality: int,
) -> ProcessedImage:
    """Decode raw image bytes, fix orientation, downscale, and re-encode.

    Keeps PNG for images with an alpha channel, otherwise encodes as JPEG
    to keep payloads small for the Ollama API call.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001 - surface as a client error
        raise ImageProcessingError(f"could not decode image '{name}': {exc}") from exc

    img = ImageOps.exif_transpose(img)

    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

    if max_side and max_side > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    if has_alpha:
        img = img.convert("RGBA")
        img.save(buf, format="PNG", optimize=True)
        mime_type = "image/png"
    else:
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=jpeg_quality)
        mime_type = "image/jpeg"

    encoded = buf.getvalue()
    width, height = img.size
    return ProcessedImage(
        name=name,
        width=width,
        height=height,
        size_bytes=len(encoded),
        base64_data=base64.b64encode(encoded).decode("ascii"),
        mime_type=mime_type,
    )
