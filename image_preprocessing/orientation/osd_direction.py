"""Stage 4B — resolve the 180° ambiguity.

Text-line geometry cannot tell 120° from 300°. After the geometric detector
has aligned lines to (approximately) horizontal, this stage asks: is the
page upright, or upside down?

Primary signal: Tesseract OSD (orientation only — never used to estimate the
arbitrary angle itself). Full OCR is not run.

Fallback (when Tesseract is missing / OSD fails): classical per-line
ascender/descender-style votes from glyph components. This is conservative;
if the fallback is also ambiguous the pipeline abstains rather than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.text_geometry import (
    extract_ink,
    extract_text_components,
    group_text_lines,
)
from image_preprocessing.utils.image_utils import (
    INTER_DOWNSAMPLE,
    bgr_to_pil,
    ensure_bgr,
    resize_max_dimension,
    to_gray,
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


def _line_updown_votes(image: np.ndarray) -> tuple[int, int, dict[str, Any]]:
    """Classical 180° vote after lines are already near-horizontal.

    Latin text tends to keep small marks (periods, commas) near the baseline
    (bottom of each line box). Returns (votes_upright, votes_upside_down, diag).
    """
    gray = to_gray(image)
    analysis, _ = resize_max_dimension(gray, 1200, interpolation=INTER_DOWNSAMPLE)
    _clahe, _ink, ink = extract_ink(analysis)
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=3)
    upright = 0
    upside = 0
    for line in lines:
        if len(line.components) < 4:
            continue
        heights = np.array([c.h for c in line.components], dtype=np.float64)
        med_h = float(np.median(heights))
        if med_h <= 0:
            continue
        small = [c for c in line.components if c.h <= med_h * 0.6]
        if len(small) < 2:
            continue
        y_top = min(c.y for c in line.components)
        y_bot = max(c.y + c.h for c in line.components)
        mid = (y_top + y_bot) / 2.0
        lower = sum(1 for c in small if c.cy > mid)
        upper = len(small) - lower
        if lower == upper:
            continue
        if lower > upper:
            upright += 1
        else:
            upside += 1
    return upright, upside, {"n_lines": len(lines), "n_comps": len(comps)}


def _resolve_with_geometry_fallback(
    geometric_angle_180: float,
    aligned_image: np.ndarray,
    *,
    reason: str,
) -> OsdDirectionResult:
    upright, upside, vote_diag = _line_updown_votes(aligned_image)
    total = upright + upside
    diag = {
        "reason": reason,
        "fallback": "line_updown_votes",
        "upright_votes": upright,
        "upside_votes": upside,
        **vote_diag,
        "geometric_angle_180": geometric_angle_180,
    }
    # Need a clear majority on enough independent lines.
    if total < 3:
        return OsdDirectionResult(
            rotation_angle=None,
            osd_rotate=None,
            osd_orientation=None,
            osd_confidence=None,
            confidence=0.2,
            status="UNCERTAIN",
            warning="Rotation could not be determined confidently",
            diagnostics={**diag, "fallback_result": "insufficient_votes"},
        )
    margin = abs(upright - upside) / float(total)
    if margin < 0.30:
        return OsdDirectionResult(
            rotation_angle=None,
            osd_rotate=None,
            osd_orientation=None,
            osd_confidence=None,
            confidence=round(float(0.35 * margin), 4),
            status="UNCERTAIN",
            warning="Rotation could not be determined confidently",
            diagnostics={**diag, "fallback_result": "ambiguous_votes"},
        )

    add_180 = upside > upright
    final = (float(geometric_angle_180) + (180.0 if add_180 else 0.0)) % 360.0
    conf = float(np.clip(0.50 + 0.45 * margin, 0.0, 0.92))
    warning = None
    if reason == "osd_unavailable":
        warning = "Tesseract OSD unavailable; used classical 180-degree fallback"
    return OsdDirectionResult(
        rotation_angle=round(final, 2),
        osd_rotate=180 if add_180 else 0,
        osd_orientation=None,
        osd_confidence=None,
        confidence=round(conf, 4),
        status="RESOLVED",
        warning=warning,
        diagnostics={**diag, "fallback_result": "resolved", "add_180": add_180},
    )


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
        return _resolve_with_geometry_fallback(
            geometric_angle_180, aligned_image, reason="osd_unavailable"
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
        # Weak OSD → try classical fallback before giving up.
        fallback = _resolve_with_geometry_fallback(
            geometric_angle_180, aligned_image, reason="osd_low_confidence"
        )
        fallback.diagnostics = {**diag, **fallback.diagnostics}
        fallback.osd_rotate = rotate
        fallback.osd_orientation = orientation
        fallback.osd_confidence = osd_conf
        return fallback

    if rotate in (90, 270):
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
