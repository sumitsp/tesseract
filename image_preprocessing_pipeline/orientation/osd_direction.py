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

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# OSD needs glyphs to vote on; a near-blank page yields a confident nonsense
# answer, so dividers and blank backs do not get spun around.
MIN_INK_FRACTION = 0.002

# How much more confident the flipped reading must be than the as-is reading
# before ``detect_quadrant_and_mirror`` concludes the page is mirrored.
# Mirrored Latin glyphs are not valid character shapes, so a genuinely
# mirrored page measures at roughly 0.2-4.0 raw ``orientation_conf`` as-is but
# 16-20 once flipped back to normal reading direction -- a >4x gap, verified
# across all four quadrants combined with mirroring. 2.0 leaves a wide safety
# margin against mirroring a page that merely scored unevenly between the two
# reading directions, while still reliably catching a real mirror.
MIRROR_FLIP_WIN_RATIO = 2.0


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


@dataclass
class QuadrantMirrorResult:
    status: str  # RESOLVED | UNRESOLVED
    content_cw: int | None
    """Clockwise quadrant to apply BEFORE mirroring (pipeline order is
    rotate -> mirror). Already accounts for ``mirror`` -- see derivation in
    ``detect_quadrant_and_mirror``."""
    mirror: bool | None
    """None only when ``status`` is UNRESOLVED. Otherwise a definite verdict:
    OSD's confidence gap between the two reading directions is decisive
    enough (see ``MIRROR_FLIP_WIN_RATIO``) that there is no third option."""
    confidence: float
    script: str = ""
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


def detect_quadrant_and_mirror(
    axis_aligned_image: np.ndarray,
    ink: np.ndarray | None,
    min_confidence: float,
) -> QuadrantMirrorResult:
    """Resolve quadrant AND mirror together, from Tesseract OSD alone.

    Why together, not mirror-after-quadrant
    ----------------------------------------
    An earlier version of this pipeline only ran mirror detection once the
    quadrant was already resolved and treated as an upright page (see
    ``mirror_detector.py``). That created a deadlock for every mirrored page:
    mirrored Latin glyphs are not valid character shapes, so OSD's confidence
    on a mirrored page collapses (measured empirically: 0.2-4.0 raw
    ``orientation_conf`` as-is, against 16-20 on the same page flipped back to
    normal reading direction). The quadrant stage therefore reported
    UNCERTAIN on every mirrored page regardless of ``min_confidence``, which
    left the downstream mirror stage never running at all -- it required a
    confirmed quadrant that a mirrored page could never produce. Measured
    against a synthetic corpus spanning all four quadrants combined with
    mirroring, that deadlock dropped 39 of 40 mirrored pages entirely.

    Running OSD on both the image and its horizontal flip and keeping
    whichever answer is actually confident resolves quadrant and mirror in
    the same step, turning the exact failure mode that broke the old design
    into the detection signal for the new one: a real mirror is exactly the
    condition where flipping first turns a low-confidence OSD read into a
    high-confidence one.

    Sign derivation for the flipped branch
    ---------------------------------------
    ``detect_quadrant`` on the flipped image reports ``Q = content_cw`` of
    ``flip_h(image)``, i.e. rotating ``flip_h(image)`` by ``Q`` gives an
    upright page. Reflections invert the rotation they're conjugated by
    (``flip_h ∘ rotate(θ) = rotate(-θ) ∘ flip_h``), so that is equivalent to
    ``flip_h(rotate(image, -Q))`` being upright. The pipeline's correction
    order is rotate-then-mirror, so the quadrant to apply *before* flipping
    is ``R = (-Q) % 360``. Verified with pixel-exact round trips for all four
    quadrants combined with mirroring.
    """
    normal = detect_quadrant(axis_aligned_image, ink, min_confidence)
    flipped_image = cv2.flip(axis_aligned_image, 1)
    flipped = detect_quadrant(flipped_image, ink, min_confidence)

    diagnostics = {
        "normal_status": normal.status,
        "normal_confidence": normal.confidence,
        "normal_content_cw": normal.content_cw,
        "flipped_status": flipped.status,
        "flipped_confidence": flipped.confidence,
        "flipped_content_cw": flipped.content_cw,
    }

    normal_ok = normal.status == "RESOLVED" and normal.content_cw is not None
    flipped_ok = flipped.status == "RESOLVED" and flipped.content_cw is not None

    if flipped_ok and flipped.confidence >= MIRROR_FLIP_WIN_RATIO * max(normal.confidence, 1e-9):
        quadrant = (-int(flipped.content_cw)) % 360
        return QuadrantMirrorResult(
            status="RESOLVED",
            content_cw=quadrant,
            mirror=True,
            confidence=flipped.confidence,
            script=flipped.script,
            diagnostics=diagnostics,
        )

    if normal_ok:
        return QuadrantMirrorResult(
            status="RESOLVED",
            content_cw=int(normal.content_cw),
            mirror=False,
            confidence=normal.confidence,
            script=normal.script,
            diagnostics=diagnostics,
        )

    return QuadrantMirrorResult(
        status="UNRESOLVED",
        content_cw=None,
        mirror=None,
        confidence=max(normal.confidence, flipped.confidence),
        warning=normal.warning or flipped.warning,
        diagnostics=diagnostics,
    )
