"""Stage 4 — arbitrary page rotation, in two independent sub-stages.

    4A  residual off-axis angle, in [-45, 45)   -- projection profile, classical
    4B  which quadrant of the remaining 4, AND
        whether the page is mirrored            -- Tesseract OSD (osd_direction)

    rotation_angle = residual + quadrant        (clockwise offset of content)

Why 4B also resolves mirror
----------------------------
Mirrored Latin glyphs are not valid character shapes, so OSD's confidence on
a mirrored page collapses regardless of which quadrant is tried. An earlier
design ran mirror detection as its own stage 5, gated on the quadrant already
being resolved -- which deadlocked on every mirrored page, since OSD could
never resolve the quadrant of a mirrored page in the first place. 4B now
runs OSD on the image and on its horizontal flip and keeps whichever answer
is actually confident, which resolves quadrant and mirror together and
removes that deadlock. See ``osd_direction.detect_quadrant_and_mirror``.

Why this order, and not the other way round
-------------------------------------------
The obvious arrangement is to search the whole 0-360 space geometrically and use
OSD only to settle 180. That was tried and it is what produced 1 correct page in
100: searching 360 degrees of ink geometry has two failure modes that compound.
Rival structures (table rules, the page outline, a stamp) win the sweep outright,
and the score is nearly flat across quadrants, so the reported angle is confident
and wrong.

Splitting the problem removes both. The residual search only has to answer a
well-conditioned question — "how far off-axis is this page" — over a 90 degree
span where the text-line peak is sharp and unambiguous. Measured over 240
page/angle combinations it lands within 1 degree on 236 of them, median error
0.02 degrees. The quadrant, the part geometry is bad at, goes to OSD, which is
good at it precisely because it reads glyph shapes.

Small residuals are handed to the tilt stage
--------------------------------------------
A residual inside ``max_skew_angle`` is not reported as rotation. It is left in
place for stage 6, which measures the same quantity on a page that is by then
upright, using the sharper single-axis score, and which validates its own result.
Rotation therefore reports quadrant turns and genuinely large arbitrary angles
(17, 63, 120, 237 degrees) and never competes with the deskewer over half a degree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.angle_search import (
    AngleEstimate,
    axis_score,
    search,
    text_ink,
    wrap_pm90,
)
from image_preprocessing.orientation.osd_direction import (
    QuadrantMirrorResult,
    detect_quadrant_and_mirror,
)
from image_preprocessing.utils.image_utils import ensure_bgr, rotate_bound

LOGGER = logging.getLogger(__name__)


@dataclass
class RotationResult:
    status: str  # DETECTED | NOT_NEEDED | UNCERTAIN
    angle_cw_deg: float | None
    """Clockwise offset of content from upright, in [0, 360). Correct by
    rotating the image counter-clockwise by this amount."""
    confidence: float
    residual_deg: float | None
    """Off-axis residual as measured, in [-45, 45), whether or not it is used."""
    quadrant_deg: int | None
    mirror_from_osd: bool | None = None
    """Mirror verdict from the same OSD pass that resolved ``quadrant_deg``
    (see ``osd_direction.detect_quadrant_and_mirror``). None only when the
    quadrant itself was not resolved."""
    residual_component_deg: float = 0.0
    """The part of ``residual_deg`` folded into ``angle_cw_deg``. Zero when the
    residual was small enough to leave to the tilt stage, or not trustworthy.
    Kept explicit so the correction can apply the quadrant losslessly and warp
    only once, without re-deriving which part went where."""
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _residual_confidence(est: AngleEstimate, config: PipelineConfig) -> float:
    """Map sweep shape onto 0-1.

    Two things make a residual trustworthy: the winning angle stands well clear
    of a typical angle on this page (``peak_margin``), and no rival structure at
    a different angle comes close to matching it (``runner_up_ratio``).
    Calibration: sparse pages that produced the only large residual errors had
    peak_margin near 0.23, while pages that were measured correctly sat at
    0.40-0.90.
    """
    if est.angle_cw_deg is None:
        return 0.0
    margin = est.peak_margin
    # 0.20 -> 0, 0.55 -> 1
    margin_term = float(np.clip((margin - 0.20) / 0.35, 0.0, 1.0))
    # runner_up 0.995 -> 0, 0.95 -> 1: only near-ties are penalised
    rivalry_term = float(np.clip((0.995 - est.runner_up_ratio) / 0.045, 0.0, 1.0))
    evidence_term = float(np.clip(est.n_points / 5000.0, 0.0, 1.0))
    conf = 0.55 * margin_term + 0.30 * rivalry_term + 0.15 * evidence_term

    # An answer sitting on the +/-45 boundary is inherently ambiguous: +45 and
    # -45 are the same line geometry, and this is exactly where weak-evidence
    # pages pile up.
    if abs(abs(est.angle_cw_deg) - 45.0) < 0.75:
        conf = min(conf, 0.35)
    return float(np.clip(conf, 0.0, 1.0))


def detect_rotation(
    image: np.ndarray,
    config: PipelineConfig,
    *,
    debug: dict | None = None,
) -> RotationResult:
    """Measure the page's clockwise rotation from upright.

    ``image`` is the full-quality working page; all measurement happens on a
    downscaled analysis copy.
    """
    ink = text_ink(image, config.analysis_max_dimension)
    if debug is not None:
        debug["ink"] = ink

    est = search(
        ink,
        lo=-45.0,
        hi=45.0,
        coarse_step=config.rotation_coarse_step_deg,
        scorer=axis_score,
    )
    residual_conf = _residual_confidence(est, config)
    diagnostics: dict = {
        "residual_raw": est.angle_cw_deg,
        "residual_confidence": round(residual_conf, 4),
        "peak_margin": round(est.peak_margin, 4),
        "runner_up_ratio": round(est.runner_up_ratio, 4),
        "ink_points": est.n_points,
        **est.diagnostics,
    }

    if est.angle_cw_deg is None:
        return RotationResult(
            status="UNCERTAIN",
            angle_cw_deg=None,
            confidence=0.0,
            residual_deg=None,
            quadrant_deg=None,
            warning="Rotation could not be determined confidently",
            diagnostics=diagnostics,
        )

    residual = wrap_pm90(float(est.angle_cw_deg))

    # Only trust a large residual when the sweep was decisive; an untrustworthy
    # large residual is dropped to zero rather than applied, and OSD still gets
    # a chance on the page as it stands.
    trust_residual = residual_conf >= config.rotation_residual_confidence_threshold
    applied_residual = residual if trust_residual else 0.0
    diagnostics["residual_trusted"] = bool(trust_residual)

    # Square the page up before asking OSD which way is up. OSD is materially
    # more reliable on an axis-aligned page, and the residual is the only thing
    # standing between an arbitrary angle and axis alignment.
    if abs(applied_residual) > 0.05:
        aligned = rotate_bound(ensure_bgr(image), applied_residual)
        aligned_ink = text_ink(aligned, config.analysis_max_dimension)
    else:
        aligned = ensure_bgr(image)
        aligned_ink = ink
    if debug is not None:
        debug["aligned"] = aligned

    quad: QuadrantMirrorResult = detect_quadrant_and_mirror(
        aligned, aligned_ink, config.osd_min_orientation_confidence
    )
    diagnostics["osd_status"] = quad.status
    diagnostics["osd_confidence"] = quad.confidence
    diagnostics["osd_script"] = quad.script
    diagnostics["osd_mirror"] = quad.mirror
    diagnostics.update({f"osd_{k}": v for k, v in quad.diagnostics.items()})

    if quad.status != "RESOLVED" or quad.content_cw is None:
        # Without a quadrant there is no total angle to report. The page keeps
        # its orientation; the deskewer downstream may still fix a small tilt.
        return RotationResult(
            status="UNCERTAIN",
            angle_cw_deg=None,
            confidence=round(residual_conf, 4),
            residual_deg=round(residual, 3),
            quadrant_deg=None,
            mirror_from_osd=None,
            warning=quad.warning or "Rotation could not be determined confidently",
            diagnostics=diagnostics,
        )

    quadrant = int(quad.content_cw)

    # Residuals within the deskew band belong to stage 6, not here.
    report_residual = applied_residual if abs(applied_residual) > config.max_skew_angle else 0.0
    total = (report_residual + quadrant) % 360.0
    diagnostics["residual_deferred_to_tilt"] = bool(
        abs(applied_residual) <= config.max_skew_angle and abs(applied_residual) > 0.05
    )

    # OSD confidence is open-ended; express it as 0-1 for the report by
    # saturating at 3x the gate, then take the weaker of the two sub-stages.
    osd_term = float(np.clip(quad.confidence / (3.0 * config.osd_min_orientation_confidence), 0.0, 1.0))
    confidence = float(min(max(residual_conf, 0.5) if report_residual == 0.0 else residual_conf, osd_term))

    if abs(total) < 0.05 or abs(total - 360.0) < 0.05:
        return RotationResult(
            status="NOT_NEEDED",
            angle_cw_deg=0.0,
            confidence=round(confidence, 4),
            residual_deg=round(residual, 3),
            quadrant_deg=quadrant,
            mirror_from_osd=quad.mirror,
            residual_component_deg=0.0,
            diagnostics=diagnostics,
        )

    return RotationResult(
        status="DETECTED",
        angle_cw_deg=round(float(total), 3),
        confidence=round(confidence, 4),
        residual_deg=round(residual, 3),
        quadrant_deg=quadrant,
        mirror_from_osd=quad.mirror,
        residual_component_deg=round(float(report_residual), 3),
        diagnostics=diagnostics,
    )


def overlay_ink(gray_or_bgr: np.ndarray, ink: np.ndarray, label: str) -> np.ndarray:
    """Debug view: ink mask tinted over the page, with the stage's verdict."""
    base = ensure_bgr(gray_or_bgr)
    if ink.shape[:2] != base.shape[:2]:
        ink = cv2.resize(ink, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = base.copy()
    vis[ink > 0] = (0, 0, 255)
    cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
    return vis
