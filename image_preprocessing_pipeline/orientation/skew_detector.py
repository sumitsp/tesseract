"""Stage 6 — fine tilt / skew, after rotation and mirror.

The page is upright by now, so this is the classical deskew problem: find the
small angle that flattens text lines. Two things make it better conditioned than
the stage-4 residual search, and both are the reason a residual inside
``max_skew_angle`` is deferred to here rather than corrected as rotation:

  * the quadrant is settled, so the score can use the horizontal projection
    alone (``line_score``) instead of the quadrant-blind ``max(y, x)``. Half the
    score's freedom to be fooled disappears with it.
  * the search spans only +/-``max_skew_angle``, so a rival structure 30 degrees
    away cannot win.

Page borders are deliberately not used. Tables, stamps, signatures and scanner
edges all produce long straight lines that are not text baselines; the ink mask
from ``angle_search.text_ink`` has already dropped every page-scale component,
so what is measured here is glyph ink only.

A tilt is reported only when the sweep is decisive; otherwise the page is left
alone and the row records UNCERTAIN. Text-line grouping supplies an independent
cross-check on the projection answer, and disagreement lowers confidence rather
than overriding it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.angle_search import (
    AngleEstimate,
    line_score,
    search,
    text_ink,
)
from image_preprocessing.orientation.text_geometry import (
    extract_text_components,
    group_text_lines,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SkewResult:
    status: str  # DETECTED | NOT_NEEDED | UNCERTAIN
    tilt_cw_deg: float | None
    confidence: float
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _text_line_tilt(ink: np.ndarray, limit: float) -> tuple[float | None, int]:
    """Median slope of grouped text lines — an independent second opinion.

    Returns (angle_cw, n_lines_used). Robust median with MAD rejection so a few
    mis-grouped lines cannot move the answer.
    """
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=4)
    angles = np.array([ln.angle_deg for ln in lines], dtype=np.float64)
    angles = angles[np.abs(angles) <= limit]
    if len(angles) < 3:
        return None, int(len(angles))
    median = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median))) + 1e-6
    inliers = angles[np.abs(angles - median) <= 2.5 * mad]
    if len(inliers) < 3:
        return None, int(len(inliers))
    return float(np.median(inliers)), int(len(inliers))


def _confidence(est: AngleEstimate, line_angle: float | None, n_lines: int) -> tuple[float, dict]:
    margin_term = float(np.clip((est.peak_margin - 0.20) / 0.35, 0.0, 1.0))
    rivalry_term = float(np.clip((0.995 - est.runner_up_ratio) / 0.045, 0.0, 1.0))
    evidence_term = float(np.clip(est.n_points / 5000.0, 0.0, 1.0))
    conf = 0.50 * margin_term + 0.25 * rivalry_term + 0.15 * evidence_term

    agreement = None
    if line_angle is not None and est.angle_cw_deg is not None:
        agreement = abs(line_angle - est.angle_cw_deg)
        # Independent confirmation is worth a real boost; open disagreement
        # between two different measurements of the same quantity is a reason
        # to abstain, not to pick one.
        if agreement <= 1.0:
            conf += 0.10
        elif agreement > 3.0:
            conf = min(conf, 0.45)
    return float(np.clip(conf, 0.0, 1.0)), {
        "line_tilt": line_angle,
        "line_count": n_lines,
        "line_agreement_deg": agreement,
    }


def detect_skew(image: np.ndarray, config: PipelineConfig) -> SkewResult:
    """Residual clockwise tilt of an already-upright page."""
    ink = text_ink(image, config.analysis_max_dimension)
    limit = float(config.max_skew_angle)

    est = search(
        ink,
        lo=-limit,
        hi=limit + 1e-9,
        coarse_step=config.skew_coarse_step_deg,
        fine_steps=(0.1, config.skew_fine_step_deg),
        scorer=line_score,
    )
    line_angle, n_lines = _text_line_tilt(ink, limit)
    confidence, cross = _confidence(est, line_angle, n_lines)

    diagnostics = {
        "tilt_raw": est.angle_cw_deg,
        "peak_margin": round(est.peak_margin, 4),
        "runner_up_ratio": round(est.runner_up_ratio, 4),
        "ink_points": est.n_points,
        **cross,
        **est.diagnostics,
    }

    if est.angle_cw_deg is None:
        return SkewResult(
            status="UNCERTAIN",
            tilt_cw_deg=None,
            confidence=0.0,
            warning="Tilt could not be determined confidently",
            diagnostics=diagnostics,
        )

    if confidence < config.skew_confidence_threshold:
        return SkewResult(
            status="UNCERTAIN",
            tilt_cw_deg=None,
            confidence=round(confidence, 4),
            warning="Tilt could not be determined confidently",
            diagnostics=diagnostics,
        )

    tilt = float(est.angle_cw_deg)
    if abs(tilt) < config.skew_min_abs_to_apply:
        return SkewResult(
            status="NOT_NEEDED",
            tilt_cw_deg=0.0,
            confidence=round(confidence, 4),
            diagnostics=diagnostics,
        )

    return SkewResult(
        status="DETECTED",
        tilt_cw_deg=round(tilt, 3),
        confidence=round(confidence, 4),
        diagnostics=diagnostics,
    )
