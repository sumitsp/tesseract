"""Which quadrant is the page in? Tesseract OSD, on an axis-aligned page.

Stage 4B. The projection search in ``arbitrary_rotation`` measures how far the
page is off-axis but cannot tell 0 from 90, 180 or 270: ink geometry is
symmetric under quarter turns, and text-line geometry is symmetric under a half
turn. Deciding which way is *up* needs character shapes, which is what OSD does.

Why OSD owns this decision outright
-----------------------------------
Ink-geometry heuristics for the quadrant were measured twice and failed twice.
The detector in the sibling ``model-repo`` project recovered 0 of 6 sideways
pages and reported confidence 1.000 on the wrong answers. A rewrite here that
led with geometry and used OSD only as a tie-break landed 1 of 100 client pages
correctly. OSD, given an axis-aligned page, was exact on every page tried.

So there is no geometric fallback for the quadrant. When OSD will not answer,
the page keeps its orientation and the row is marked uncertain. A confidently
wrong quarter turn is far worse than an uncorrected page, and the geometric
"answer" available here is not better than a coin flip.

Confidence gate
---------------
``orientation_conf`` is an open-ended float. Calibrated over 500 page/angle
combinations (50 oracle-verified upright pages x 10 synthetic rotations):

    threshold   coverage   wrong quadrant
        1.0        90%          23          <- value used by model-repo
        3.0        76%           7
        4.0        67%           1
        5.0        62%           0

Every wrong answer was exactly 180 degrees off and every one scored under 4.2.
They cluster on handwritten pages where print bled through from the reverse
side: OSD reads the upside-down bleed-through, which is genuinely the only
printed text on the page. 5.0 buys zero quadrant errors for ~28 points of
coverage, and abstaining costs only an uncorrected page.

Convention, verified rather than assumed
----------------------------------------
``image_to_osd`` reports ``rotate`` as the CLOCKWISE rotation to APPLY to make
the page upright, so the content's own clockwise offset is its complement::

    content 0 CW   -> rotate 0        content 180 CW -> rotate 180
    content 90 CW  -> rotate 270      content 270 CW -> rotate 90

    content_cw = (360 - rotate) % 360
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)

# OSD needs glyphs to vote on; a near-blank page yields a confident nonsense
# answer, so dividers and blank backs do not get spun around.
MIN_INK_FRACTION = 0.002


@dataclass
class QuadrantResult:
    status: str  # RESOLVED | UNRESOLVED
    content_cw: int | None
    confidence: float
    """Raw Tesseract ``orientation_conf``, not rescaled — the thresholds in
    config are expressed on this same open-ended scale."""
    script: str = ""
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _unresolved(reason: str, *, confidence: float = 0.0, warning: str | None = None) -> QuadrantResult:
    return QuadrantResult(
        status="UNRESOLVED",
        content_cw=None,
        confidence=float(confidence),
        warning=warning,
        diagnostics={"reason": reason},
    )


def detect_quadrant(
    axis_aligned_image: np.ndarray,
    ink: np.ndarray | None,
    min_confidence: float,
) -> QuadrantResult:
    """Clockwise quadrant offset of ``axis_aligned_image``, or UNRESOLVED.

    ``axis_aligned_image`` must already have its off-axis residual removed;
    OSD is markedly more reliable on a squared-up page, which is the whole
    reason the residual search runs first.
    """
    if ink is not None and ink.size:
        ink_fraction = float(np.count_nonzero(ink)) / float(ink.size)
        if ink_fraction < MIN_INK_FRACTION:
            return _unresolved(
                "too_little_ink",
                warning="Orientation not determined: page has too little text",
            )

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        LOGGER.debug("pytesseract unavailable: %s", exc)
        return _unresolved(
            "osd_unavailable",
            warning="Tesseract OSD unavailable; page orientation left unchanged",
        )

    try:
        osd = pytesseract.image_to_osd(axis_aligned_image, output_type=Output.DICT)
    except Exception as exc:
        # "Too few characters" on sparse pages, TesseractError when the osd
        # traineddata is missing. Both mean "no answer", not "page is broken".
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        LOGGER.debug("OSD did not resolve: %s", first_line)
        return _unresolved(
            "osd_no_answer",
            warning="Tesseract OSD could not determine orientation; page left unchanged",
        )

    try:
        rotate = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return _unresolved("osd_unparsable")

    if rotate not in (0, 90, 180, 270):
        return _unresolved("osd_non_quadrant", confidence=confidence)

    if confidence < min_confidence:
        return _unresolved(
            "osd_low_confidence",
            confidence=confidence,
            warning=(
                f"Orientation confidence {confidence:.1f} below {min_confidence:.1f}; "
                "page orientation left unchanged"
            ),
        )

    return QuadrantResult(
        status="RESOLVED",
        content_cw=(360 - rotate) % 360,
        confidence=confidence,
        script=str(osd.get("script") or ""),
        diagnostics={"osd_rotate": rotate, "reason": "resolved"},
    )
