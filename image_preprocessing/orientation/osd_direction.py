"""Stage 4B — resolve the 180° ambiguity with Tesseract OSD only.

Text-line geometry cannot tell 120° from 300°. After the geometric detector
has aligned lines to (approximately) horizontal, Tesseract OSD is asked a
single question: is the page upright, or upside down?

OSD is never used to estimate the arbitrary angle itself. Full OCR is not
run. If OSD is unavailable, low-confidence, or disagrees with a 90°/270°
reading, the pipeline abstains rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.utils.image_utils import (
    INTER_DOWNSAMPLE,
    bgr_to_pil,
    ensure_bgr,
    resize_max_dimension,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class OsdDirectionResult:
    rotation_angle: float | None  # final clockwise content offset in [0, 360)
    osd_rotate: int | None
    osd_orientation: int | None
    osd_confidence: float | None
    confidence: float
    status: str  # RESOLVED | UNCERTAIN
    warning: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _run_osd(image: np.ndarray) -> dict[str, Any] | None:
    try:
        import pytesseract
    except Exception as exc:
        LOGGER.warning("pytesseract is not available for OSD: %s", exc)
        return None
    work, _ = resize_max_dimension(ensure_bgr(image), 1000, interpolation=INTER_DOWNSAMPLE)
    pil = bgr_to_pil(work)
    try:
        osd = pytesseract.image_to_osd(pil, output_type=pytesseract.Output.DICT)
        return osd
    except Exception as exc:
        LOGGER.warning("Tesseract OSD failed: %s", exc)
        return None


def resolve_180(
    aligned_image: np.ndarray,
    geometric_angle_180: float,
    config: PipelineConfig,
) -> OsdDirectionResult:
    """``aligned_image`` has already been rotated CCW by ``geometric_angle_180``.

    Lines should be approximately horizontal. OSD ``rotate`` is the clockwise
    correction Tesseract wants on that aligned image (0/90/180/270).
    """
    osd = _run_osd(aligned_image)
    if not osd:
        return OsdDirectionResult(
            rotation_angle=None,
            osd_rotate=None,
            osd_orientation=None,
            osd_confidence=None,
            confidence=0.0,
            status="UNCERTAIN",
            warning="Rotation could not be determined confidently",
            diagnostics={"reason": "osd_unavailable"},
        )

    rotate = int(osd.get("rotate", 0)) % 360
    orientation = int(osd.get("orientation", 0))
    conf_raw = osd.get("orientation_conf", osd.get("orientation_confidence", 0.0))
    try:
        osd_conf = float(conf_raw)
    except (TypeError, ValueError):
        osd_conf = 0.0

    diag = {
        "osd": {k: osd[k] for k in osd},
        "geometric_angle_180": geometric_angle_180,
        "osd_rotate": rotate,
        "osd_orientation": orientation,
        "osd_confidence": osd_conf,
    }

    if osd_conf < config.osd_min_orientation_confidence:
        return OsdDirectionResult(
            rotation_angle=None,
            osd_rotate=rotate,
            osd_orientation=orientation,
            osd_confidence=osd_conf,
            confidence=round(float(np.clip(osd_conf / 10.0, 0.0, 0.49)), 4),
            status="UNCERTAIN",
            warning="Rotation could not be determined confidently",
            diagnostics={**diag, "reason": "osd_low_confidence"},
        )

    if rotate in (90, 270):
        # Geometry claimed the lines are horizontal; OSD says the page is
        # sideways. The two stages disagree — do not force a correction.
        return OsdDirectionResult(
            rotation_angle=None,
            osd_rotate=rotate,
            osd_orientation=orientation,
            osd_confidence=osd_conf,
            confidence=0.2,
            status="UNCERTAIN",
            warning="Rotation could not be determined confidently",
            diagnostics={**diag, "reason": "osd_axis_disagrees_with_geometry"},
        )

    if rotate == 180:
        final = (float(geometric_angle_180) + 180.0) % 360.0
    else:
        final = float(geometric_angle_180) % 360.0

    # Map OSD confidence (often ~0–20) into 0–1 without claiming certainty
    # OSD cannot actually provide.
    mapped = float(np.clip(0.45 + 0.55 * min(osd_conf / 8.0, 1.0), 0.0, 1.0))
    return OsdDirectionResult(
        rotation_angle=round(final, 2),
        osd_rotate=rotate,
        osd_orientation=orientation,
        osd_confidence=osd_conf,
        confidence=round(mapped, 4),
        status="RESOLVED",
        warning=None,
        diagnostics=diag,
    )
