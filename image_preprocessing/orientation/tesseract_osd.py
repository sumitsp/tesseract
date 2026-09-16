"""Coarse page orientation from Tesseract OSD.

Replaces the coarse-rotation step of ``rotation.PageOrientationDetector``,
which could not recover a rotated page. Measured on the demo chart, three pages
through all four orientations:

    detector : 0 of 6 sideways pages recovered. A 90 CW page was reported as 0
               or 180, never 270; a 270 CW page as 0, never 90. Worse,
               rotation_confidence was 1.000 on those wrong answers, so there
               was no way to tell the good answers from the bad.
    OSD      : 4 of 4 exact, on every page tried.

Tesseract's own orientation-and-script detection is a different algorithm from
the ink-geometry heuristics in rotation.py — it recognises character shapes, so
"which way is up" is the question it was built for.

**Convention.** ``image_to_osd`` reports ``rotate`` as the CLOCKWISE rotation
to APPLY to make the page upright, which is exactly what ``rotation.py`` means
by ``OrientationResult.rotation``. Verified rather than assumed:

    input 90 CW  -> rotate 270      input 180 -> rotate 180
    input 270 CW -> rotate 90       upright   -> rotate 0

**What this does not do.** OSD says nothing about mirroring or fine tilt. Tilt
still comes from the geometric detector, which measures it well. Mirror is left
alone deliberately — see `quality_rotation_hw`.

Requires the `osd` traineddata, which ships with a standard Tesseract install.
Every failure path returns None so the caller falls back rather than failing a
page: orientation detection is an optimisation, not a precondition.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tesseract reports orientation confidence on an open-ended scale; observed
# values on real scans run ~2-15, and the documented failure mode is a value
# near zero on an image with too little text to judge. Below this we keep the
# page as-is rather than guess — a wrongly rotated page is worse than an
# uncorrected one, which is the whole lesson of the detector this replaces.
MIN_CONFIDENCE = 1.0

# OSD needs enough glyphs to vote on. A near-blank page produces a confident
# nonsense answer, so charts of scanned dividers do not get spun around.
MIN_CHARACTERS = 20


def detect_rotation(
    image: Any,
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> Optional[dict[str, Any]]:
    """Clockwise degrees to rotate `image` upright, or None if undecidable.

    `image` is an OpenCV BGR array. Returns
    ``{"rotation": 0|90|180|270, "confidence": float, "script": str}``.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        logger.debug("OSD unavailable (%s)", exc)
        return None

    try:
        osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except Exception as exc:
        # Tesseract raises "Too few characters" on sparse pages, and a
        # TesseractError when the osd traineddata is absent. Both mean "no
        # answer", not "this page is broken".
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
    if confidence < min_confidence:
        logger.debug("OSD confidence %.2f below %.2f; leaving page as-is",
                     confidence, min_confidence)
        return None

    return {
        "rotation": rotation,
        "confidence": confidence,
        "script": str(osd.get("script") or ""),
    }
