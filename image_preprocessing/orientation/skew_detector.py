"""Stage 6 — fine tilt / skew AFTER rotation and mirror.

The page is assumed approximately upright. This is not arbitrary rotation.
Search is limited to [-MAX_SKEW_ANGLE, +MAX_SKEW_ANGLE] (default ±10°).

Primary evidence: text-line fits, horizontal projection-profile search,
connected-component alignment. Hough lines are supporting validation only
and page borders are suppressed.

Positive tilt = clockwise lean of content (y-down image coordinates).
Correction rotates the image CCW by that amount (OpenCV +tilt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.text_geometry import (
    extract_ink,
    extract_text_components,
    group_text_lines,
    projection_score_at_angle,
)
from image_preprocessing.utils.image_utils import resize_max_dimension, to_gray


@dataclass
class SkewResult:
    tilt_angle: float | None
    confidence: float | None
    status: str  # DETECTED | UNCERTAIN | SKIPPED
    warning: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _component_line_tilt(ink: np.ndarray) -> tuple[float | None, float]:
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=4)
    if len(lines) < 3:
        return None, 0.0
    angles = np.array([ln.angle_deg for ln in lines], dtype=np.float64)
    angles = angles[np.abs(angles) <= 15]
    if len(angles) < 3:
        return None, 0.0
    med = float(np.median(angles))
    mad = float(np.median(np.abs(angles - med))) + 1e-6
    inliers = angles[np.abs(angles - med) <= 2.5 * mad]
    if len(inliers) < 3:
        return None, 0.0
    conf = float(np.clip(0.25 + 0.05 * len(inliers), 0.0, 1.0))
    return float(np.median(inliers)), conf


def _hough_support(gray: np.ndarray) -> tuple[float | None, float]:
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    m = max(3, int(0.03 * min(h, w)))
    edges[:m, :] = 0
    edges[-m:, :] = 0
    edges[:, :m] = 0
    edges[:, -m:] = 0
    min_len = max(30, w // 6)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=80, minLineLength=min_len, maxLineGap=10
    )
    if lines is None:
        return None, 0.0
    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        y_mid = 0.5 * (y1 + y2)
        if length > 0.85 * w and (y_mid < 0.08 * h or y_mid > 0.92 * h):
            continue
        raw = float(np.degrees(np.arctan2(dy, dx)))
        if abs(raw) > 20:
            continue
        angles.append(raw)
        weights.append(length)
    if len(angles) < 5:
        return None, 0.0
    ang = np.array(angles, dtype=np.float64)
    wts = np.array(weights, dtype=np.float64)
    order = np.argsort(ang)
    ang_s, w_s = ang[order], wts[order]
    cdf = np.cumsum(w_s)
    median = float(ang_s[np.searchsorted(cdf, 0.5 * cdf[-1])])
    frac = float(np.mean(np.abs(ang - median) <= 2.0))
    if frac < 0.45:
        return None, 0.0
    conf = float(np.clip(0.2 + 0.6 * frac + 0.01 * min(len(angles), 40), 0.0, 1.0))
    return median, conf


def _projection_search(ink: np.ndarray, seed: float, config: PipelineConfig) -> tuple[float, float]:
    max_a = float(config.max_skew_angle)
    lo = max(-max_a, seed - max_a)
    hi = min(max_a, seed + max_a)
    best_a, best_s = seed, projection_score_at_angle(ink, seed)
    a = lo
    while a <= hi + 1e-9:
        s = projection_score_at_angle(ink, float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
        a += config.skew_coarse_step_deg
    fine_lo = max(-max_a, best_a - config.skew_coarse_step_deg)
    fine_hi = min(max_a, best_a + config.skew_coarse_step_deg)
    a = fine_lo
    while a <= fine_hi + 1e-9:
        s = projection_score_at_angle(ink, float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
        a += config.skew_fine_step_deg
    neighbor = [
        projection_score_at_angle(ink, best_a - 0.5),
        projection_score_at_angle(ink, best_a + 0.5),
    ]
    conf = float(np.clip((best_s - float(np.mean(neighbor))) / (best_s + 1e-9) * 2.0, 0.0, 1.0))
    return float(best_a), conf


def detect_skew(upright_image: np.ndarray, config: PipelineConfig) -> SkewResult:
    gray = to_gray(upright_image)
    analysis, _ = resize_max_dimension(gray, config.analysis_max_dimension)
    _clahe, _ink, ink = extract_ink(analysis)

    line_a, line_c = _component_line_tilt(ink)
    hough_a, hough_c = _hough_support(analysis)
    diag: dict[str, Any] = {
        "line_tilt": line_a,
        "line_conf": line_c,
        "hough_tilt": hough_a,
        "hough_conf": hough_c,
        "note": "Hough is supporting validation only; text-line evidence is primary.",
    }

    if line_a is None and hough_a is None:
        return SkewResult(
            tilt_angle=None,
            confidence=0.0,
            status="UNCERTAIN",
            warning="Tilt could not be determined confidently",
            diagnostics=diag,
        )

    seed = float(line_a if line_a is not None else hough_a)
    refined, refine_c = _projection_search(ink, seed, config)
    diag["seed"] = seed
    diag["refined"] = refined
    diag["refine_conf"] = refine_c

    # If Hough and text-lines exist, they must not strongly disagree.
    if line_a is not None and hough_a is not None and abs(line_a - hough_a) > 3.5:
        return SkewResult(
            tilt_angle=None,
            confidence=round(min(0.4, 0.5 * (line_c + hough_c)), 4),
            status="UNCERTAIN",
            warning="Tilt could not be determined confidently",
            diagnostics={**diag, "reason": "line_hough_disagreement"},
        )

    conf = float(
        np.clip(
            0.45 * (line_c if line_a is not None else 0.3)
            + 0.20 * (hough_c if hough_a is not None else 0.2)
            + 0.35 * refine_c,
            0.0,
            1.0,
        )
    )
    if abs(refined) > config.max_skew_angle:
        return SkewResult(
            tilt_angle=None,
            confidence=round(conf, 4),
            status="UNCERTAIN",
            warning="Tilt could not be determined confidently",
            diagnostics={**diag, "reason": "outside_search_range"},
        )
    if conf < config.skew_confidence_threshold:
        return SkewResult(
            tilt_angle=None,
            confidence=round(conf, 4),
            status="UNCERTAIN",
            warning="Tilt could not be determined confidently",
            diagnostics=diag,
        )
    if abs(refined) < config.skew_min_abs_to_apply:
        return SkewResult(
            tilt_angle=0.0,
            confidence=round(max(conf, 0.7), 4),
            status="DETECTED",
            warning=None,
            diagnostics={**diag, "reason": "below_apply_threshold_treated_as_zero"},
        )
    return SkewResult(
        tilt_angle=round(float(refined), 2),
        confidence=round(conf, 4),
        status="DETECTED",
        warning=None,
        diagnostics=diag,
    )
