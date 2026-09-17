"""Coarse page orientation from Tesseract OSD (model-repo contract)."""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Same as model-repo stages/lib/imaging/osd.py — do not raise this via config.
MIN_CONFIDENCE = 1.0

# OSD can fail on very large rasters; retry on a downscaled copy with the same answer.
OSD_MAX_DIMENSION = 2400


def _prepare_rgb(image: Any) -> np.ndarray | None:
    if image is None or not hasattr(image, "shape"):
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # pytesseract treats ndarray as RGB; OpenCV loads BGR.
        rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _downscale(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return rgb
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _run_osd(rgb: np.ndarray) -> Optional[dict[str, Any]]:
    try:
        import pytesseract
        from pytesseract import Output

        from image_preprocessing.utils.tesseract_config import configure_tesseract

        configure_tesseract()
    except ImportError as exc:
        logger.warning("OSD unavailable (pytesseract): %s", exc)
        return None

    try:
        osd = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
    except Exception as exc:
        logger.debug("OSD did not resolve: %s", str(exc).splitlines()[0])
        return None

    try:
        rotation = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None

    if rotation not in (0, 90, 180, 270):
        logger.debug("OSD returned a non-quadrant rotation: %s", rotation)
        return None
    if confidence < MIN_CONFIDENCE:
        logger.debug(
            "OSD confidence %.2f below %.2f; leaving page as-is",
            confidence,
            MIN_CONFIDENCE,
        )
        return None

    return {
        "rotation": rotation,
        "confidence": confidence,
        "script": str(osd.get("script") or ""),
    }


def detect_rotation(
    image: Any,
    *,
    min_confidence: float | None = None,
) -> Optional[dict[str, Any]]:
    """Clockwise degrees to rotate ``image`` upright, or None if undecidable."""
    if min_confidence is not None and min_confidence > MIN_CONFIDENCE:
        logger.debug(
            "Ignoring min_confidence=%.2f; model-repo uses fixed %.2f",
            min_confidence,
            MIN_CONFIDENCE,
        )

    rgb = _prepare_rgb(image)
    if rgb is None:
        return None

    result = _run_osd(rgb)
    if result is not None:
        return result

    small = _downscale(rgb, OSD_MAX_DIMENSION)
    if small.shape != rgb.shape:
        result = _run_osd(small)
        if result is not None:
            logger.debug("OSD succeeded on downscaled copy (%sx%s)", small.shape[1], small.shape[0])
        return result
    return None
