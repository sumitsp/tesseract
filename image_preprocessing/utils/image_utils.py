"""Shared image conversions and geometric transforms.

Sign convention
---------------
``rotation_angle`` / ``tilt_angle`` are the clockwise offset of content from
upright, in degrees. Correction rotates the image counter-clockwise by that
amount. OpenCV ``getRotationMatrix2D`` is counter-clockwise-positive, so the
OpenCV angle equals the stored angle.

Exact 90 / 180 / 270 corrections use lossless ``cv2.rotate`` (no interpolation).
Arbitrary angles use an expanded canvas (rotate-bound) so corners are never
cropped.
"""

from __future__ import annotations

import io
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

INTER_MASK = cv2.INTER_NEAREST
INTER_FINAL = cv2.INTER_CUBIC
INTER_DOWNSAMPLE = cv2.INTER_AREA


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    rgb = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def encode_png_bytes(image: np.ndarray) -> bytes:
    pil = bgr_to_pil(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def resize_max_dimension(
    image: np.ndarray,
    max_dim: int,
    interpolation: int = INTER_DOWNSAMPLE,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    out = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=interpolation,
    )
    return out, scale


def border_value(image: np.ndarray) -> int | Sequence[int]:
    if image.ndim == 2:
        return 0 if image.dtype != np.uint8 else _likely_background(image)
    bg = _likely_background(to_gray(image))
    return (int(bg), int(bg), int(bg))


def _likely_background(gray: np.ndarray) -> int:
    # Document pages are usually light. Use the median of a thin border sample.
    h, w = gray.shape
    band = max(1, min(8, h // 50, w // 50))
    samples = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    )
    return int(np.median(samples))


def _angle_mod_360(angle: float) -> float:
    return float(angle % 360.0)


def is_cardinal_angle(angle: float, *, tol: float = 0.05) -> int | None:
    """Return 0/90/180/270 if ``angle`` is a cardinal CCW rotation, else None."""
    a = _angle_mod_360(angle)
    for cand in (0, 90, 180, 270):
        if min(abs(a - cand), 360.0 - abs(a - cand)) <= tol:
            return cand
    return None


def rotate_lossless_ccw(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees = int(degrees) % 360
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Not a lossless cardinal rotation: {degrees}")


def rotate_bound(
    image: np.ndarray,
    angle_ccw_deg: float,
    *,
    interpolation: int = INTER_FINAL,
    border: int | Sequence[int] | None = None,
) -> np.ndarray:
    """Rotate counter-clockwise, expanding the canvas so no content is cropped."""
    if image is None or abs(float(angle_ccw_deg)) < 1e-6:
        return image
    cardinal = is_cardinal_angle(angle_ccw_deg)
    if cardinal is not None:
        return rotate_lossless_ccw(image, cardinal)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_ccw_deg), 1.0)
    cos = abs(float(matrix[0, 0]))
    sin = abs(float(matrix[0, 1]))
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    if border is None:
        border = border_value(image)
    return cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def downsample_to_dpi(
    image: np.ndarray,
    input_dpi: float,
    target_dpi: float,
) -> np.ndarray:
    """Downsample so the effective DPI is approximately ``target_dpi``.

    Never upscales. If input_dpi <= target_dpi the image is returned as-is.
    """
    if input_dpi is None or input_dpi <= 0 or target_dpi <= 0:
        return image
    if input_dpi <= target_dpi * 1.02:
        return image
    scale = target_dpi / float(input_dpi)
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image
    return cv2.resize(image, (new_w, new_h), interpolation=INTER_DOWNSAMPLE)


def save_png(path, image: np.ndarray, dpi: float | None = None) -> None:
    pil = bgr_to_pil(image)
    kwargs: dict = {}
    if dpi is not None and dpi > 0:
        kwargs["dpi"] = (float(dpi), float(dpi))
    pil.save(str(path), format="PNG", **kwargs)
