"""Apply orientation corrections at full resolution (OpenCV only)."""
from __future__ import annotations

import cv2
import numpy as np

from .result import OrientationResult


def _border_value(image: np.ndarray) -> int | tuple[int, int, int]:
    if image.ndim == 2:
        return 255
    return (255, 255, 255)


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def rotate_coarse_cw(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg == 0:
        return image
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported coarse rotation: {rotation_deg}")


def rotate_fine(
    image: np.ndarray,
    tilt_clockwise_deg: float,
    *,
    expand: bool = True,
) -> np.ndarray:
    """
    Undo clockwise residual skew ``tilt_clockwise_deg``.

    OpenCV ``getRotationMatrix2D`` uses counter-clockwise-positive angles,
    so we pass ``+tilt_clockwise_deg`` to rotate the image CCW and straighten
    clockwise-leaning content.
    """
    if abs(tilt_clockwise_deg) < 1e-4:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    # CCW by +tilt undoes clockwise lean of +tilt.
    matrix = cv2.getRotationMatrix2D(center, float(tilt_clockwise_deg), 1.0)
    if expand:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        size = (new_w, new_h)
    else:
        size = (w, h)
    return cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(image),
    )


def correct_image(
    image: np.ndarray,
    result: OrientationResult | dict,
    *,
    expand: bool = True,
    apply_tilt: bool = True,
    max_tilt_abs: float | None = None,
) -> np.ndarray:
    """
    Correction order:
      1) coarse rotation
      2) mirror
      3) fine tilt

    Mirror is applied after rotation because mirror features are defined on
    upright (LTR) page geometry. Rotation and horizontal flip are not
    interchangeable for 90/270.
    """
    if isinstance(result, dict):
        mirror = bool(result["mirror"])
        rotation = int(result["rotation"])
        tilt = float(result["tilt"])
    else:
        mirror = result.mirror
        rotation = result.rotation
        tilt = result.tilt

    out = rotate_coarse_cw(image, rotation)
    if mirror:
        out = flip_horizontal(out)

    if apply_tilt:
        if max_tilt_abs is not None and abs(tilt) > max_tilt_abs:
            pass
        else:
            out = rotate_fine(out, tilt, expand=expand)
    return out
