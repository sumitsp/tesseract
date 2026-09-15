"""Stage 4A — arbitrary text-orientation estimation (classical CV only).

This module does NOT resolve 0° vs 180°. Text-line geometry is identical
under a 180° flip. Stage 4B (``osd_direction.py``) uses Tesseract OSD
exclusively for that binary choice.

Search objective
----------------
Find the angle θ in [0, 180) such that rotating the page counter-clockwise
by θ produces the strongest horizontal text-line alignment.

Signals (combined; none is used alone):
  * connected-component long-axis histogram
  * text-line fit angles after grouping
  * morphological horizontal-opening ratio after trial rotations
  * horizontal projection-profile peakiness after trial rotations
  * Hough line angles as supporting evidence only

Sign convention
---------------
The returned ``angle_deg`` is the clockwise offset of the content from
upright, modulo 180°. Correction rotates the image CCW by that amount
(OpenCV positive angle). Example: content at 120° clockwise → 120.4.

If evidence is weak the detector abstains: angle=None, status=UNCERTAIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.text_geometry import (
    GeometryBundle,
    circular_distance_180,
    circular_mean_180,
    extract_text_components,
    group_text_lines,
    projection_score_at_angle,
    wrap_180,
)
from image_preprocessing.utils.image_utils import resize_max_dimension, rotate_bound


@dataclass
class GeometricRotationResult:
    angle_deg: float | None  # [0, 180) clockwise content offset, or None
    confidence: float
    status: str  # DETECTED | UNCERTAIN
    scores: dict[float, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


def _component_orientation_vote(bundle: GeometryBundle) -> tuple[float | None, float]:
    comps = bundle.components
    if len(comps) < 8:
        return None, 0.0
    # Long, thin components (word fragments, stemmed glyphs) are more
    # informative than nearly-square blobs.
    angles = []
    weights = []
    for c in comps:
        aspect = max(c.w, c.h) / float(min(c.w, c.h) + 1e-9)
        if aspect < 1.35:
            continue
        angles.append(c.angle_deg)
        weights.append(float(c.area) * min(aspect, 6.0))
    if len(angles) < 6:
        return None, 0.0
    hist = _angle_histogram(np.array(angles, dtype=np.float64), np.array(weights, dtype=np.float64))
    peak = int(np.argmax(hist))
    # Histogram bins are 2° over [0, 180).
    angle = peak * 2.0 + 1.0
    peak_mass = float(hist[peak] + hist[(peak - 1) % 90] + hist[(peak + 1) % 90])
    total = float(hist.sum()) + 1e-9
    conf = float(np.clip(peak_mass / total, 0.0, 1.0))
    return wrap_180(angle), conf


def _line_orientation_vote(bundle: GeometryBundle) -> tuple[float | None, float]:
    if len(bundle.lines) < 2:
        return None, 0.0
    angles = np.array([wrap_180(ln.angle_deg) for ln in bundle.lines], dtype=np.float64)
    spans = np.array([max(1.0, ln.x_max - ln.x_min) for ln in bundle.lines], dtype=np.float64)
    hist = _angle_histogram(angles, spans)
    peak = int(np.argmax(hist))
    angle = peak * 2.0 + 1.0
    peak_mass = float(hist[peak] + hist[(peak - 1) % 90] + hist[(peak + 1) % 90])
    conf = float(np.clip(peak_mass / (hist.sum() + 1e-9), 0.0, 1.0))
    if len(bundle.lines) < 3:
        conf *= 0.6
    return wrap_180(angle), conf


def _hough_orientation_vote(gray: np.ndarray, ink: np.ndarray) -> tuple[float | None, float]:
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    m = max(3, int(0.03 * min(h, w)))
    edges[:m, :] = 0
    edges[-m:, :] = 0
    edges[:, :m] = 0
    edges[:, -m:] = 0
    min_len = max(24, w // 8)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=70, minLineLength=min_len, maxLineGap=12
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
        # Ignore near-full-width page borders.
        if length > 0.85 * w and (y_mid < 0.08 * h or y_mid > 0.92 * h):
            continue
        raw = float(np.degrees(np.arctan2(dy, dx)))  # y-down ⇒ clockwise-positive
        angles.append(wrap_180(raw))
        weights.append(length)
    if len(angles) < 6:
        return None, 0.0
    hist = _angle_histogram(np.array(angles), np.array(weights))
    peak = int(np.argmax(hist))
    angle = peak * 2.0 + 1.0
    peak_mass = float(hist[peak] + hist[(peak - 1) % 90] + hist[(peak + 1) % 90])
    conf = float(np.clip(peak_mass / (hist.sum() + 1e-9), 0.0, 1.0))
    return wrap_180(angle), conf


def _angle_histogram(angles: np.ndarray, weights: np.ndarray, bin_deg: float = 2.0) -> np.ndarray:
    bins = int(round(180.0 / bin_deg))
    hist = np.zeros(bins, dtype=np.float64)
    wrapped = np.mod(np.asarray(angles, dtype=np.float64), 180.0)
    idx = np.floor(wrapped / bin_deg).astype(int) % bins
    for i, w in zip(idx, weights):
        hist[int(i)] += float(w)
    # Light circular smoothing.
    hist = 0.25 * np.roll(hist, -1) + 0.5 * hist + 0.25 * np.roll(hist, 1)
    return hist


def _projection_search(
    ink: np.ndarray,
    *,
    coarse_step: float,
    fine_step: float,
    seed_angles: list[float],
) -> tuple[float, float, dict[float, float]]:
    """Search [0, 180) for the CCW rotation that maximises line alignment."""
    coarse_ink, _ = resize_max_dimension(ink, 1100, interpolation=cv2.INTER_NEAREST)
    scores: dict[float, float] = {}
    best_a, best_s = 0.0, -1.0
    angle = 0.0
    while angle < 180.0 - 1e-9:
        s = projection_score_at_angle(coarse_ink, angle)
        scores[round(angle, 2)] = s
        if s > best_s:
            best_s, best_a = s, angle
        angle += coarse_step

    for seed in seed_angles:
        seed = wrap_180(seed)
        for delta in (-coarse_step, 0.0, coarse_step):
            a = wrap_180(seed + delta)
            if min(circular_distance_180(a, k) for k in scores) < 0.51 * coarse_step:
                continue
            s = projection_score_at_angle(coarse_ink, a)
            scores[round(a, 2)] = s
            if s > best_s:
                best_s, best_a = s, a

    # Keep the top few coarse peaks (including wrap-around near 0/180).
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    peaks = [ranked[0][0]]
    for a, _s in ranked[1:]:
        if all(circular_distance_180(a, p) > 8.0 for p in peaks):
            peaks.append(a)
        if len(peaks) >= 3:
            break

    fine_best_a, fine_best_s = best_a, best_s
    for peak in peaks:
        lo = peak - 3.0
        hi = peak + 3.0
        a = lo
        while a <= hi + 1e-9:
            aa = wrap_180(a)
            s = projection_score_at_angle(ink, aa)
            scores[round(aa, 2)] = s
            if s > fine_best_s:
                fine_best_s, fine_best_a = s, aa
            a += fine_step
    return float(fine_best_a), float(fine_best_s), scores


def _peak_margin(scores: dict[float, float], best_a: float) -> float:
    if not scores:
        return 0.0
    best = scores.get(round(best_a, 2), max(scores.values()))
    orthogonal = wrap_180(best_a + 90.0)
    # Median of scores near the orthogonal direction, plus global second peak.
    others = [v for k, v in scores.items() if circular_distance_180(k, best_a) > 8.0]
    if not others:
        return 0.0
    second = max(others)
    ortho_vals = [v for k, v in scores.items() if circular_distance_180(k, orthogonal) <= 8.0]
    ortho = float(np.median(ortho_vals)) if ortho_vals else second
    denom = max(best, 1e-9)
    return float(np.clip((best - max(second, ortho)) / denom, 0.0, 1.0))


def _line_structure_at_angle(ink: np.ndarray, angle_ccw_deg: float) -> tuple[int, int, float]:
    """Count glyph components and text lines after rotating CCW by ``angle``."""
    rotated = rotate_bound(ink, angle_ccw_deg, interpolation=cv2.INTER_NEAREST, border=0)
    comps = extract_text_components(rotated)
    lines = group_text_lines(comps, min_comps=3)
    score = projection_score_at_angle(ink, angle_ccw_deg)
    return len(comps), len(lines), float(score)


def detect_arbitrary_rotation(
    bundle: GeometryBundle,
    config: PipelineConfig,
) -> GeometricRotationResult:
    comps = bundle.components
    lines = bundle.lines
    diag: dict[str, Any] = {
        "n_components": len(comps),
        "n_lines": len(lines),
    }

    if len(comps) < config.rotation_min_text_components and len(lines) < config.rotation_min_text_lines:
        return GeometricRotationResult(
            angle_deg=None,
            confidence=0.0,
            status="UNCERTAIN",
            diagnostics=diag,
            warning="Rotation could not be determined confidently",
        )

    # Use real ink pixels, not axis-aligned component boxes. AABB masks
    # destroy the orientation signal on arbitrarily rotated pages.
    work = bundle.ink_clean

    cc_angle, cc_conf = _component_orientation_vote(bundle)
    line_angle, line_conf = _line_orientation_vote(bundle)
    hough_angle, hough_conf = _hough_orientation_vote(bundle.clahe, bundle.ink_clean)
    diag.update(
        {
            "component_angle": cc_angle,
            "component_conf": cc_conf,
            "line_angle": line_angle,
            "line_conf": line_conf,
            "hough_angle": hough_angle,
            "hough_conf": hough_conf,
        }
    )

    seeds = [a for a in (cc_angle, line_angle, hough_angle) if a is not None]
    proj_angle, proj_score, scores = _projection_search(
        work,
        coarse_step=config.rotation_coarse_step_deg,
        fine_step=config.rotation_fine_step_deg,
        seed_angles=seeds,
    )
    margin = _peak_margin(scores, proj_angle)

    # Horizontal line grouping on the *unrotated* page is not a valid vote
    # for arbitrary angles (it assumes neighbours share a y-band). Confirm
    # the projection peak by grouping lines AFTER rotating to that angle,
    # versus the orthogonal direction.
    n_at, lines_at, _ = _line_structure_at_angle(work, proj_angle)
    n_ortho, lines_ortho, _ = _line_structure_at_angle(work, wrap_180(proj_angle + 90.0))

    hough_agrees = (
        hough_angle is not None
        and circular_distance_180(hough_angle, proj_angle) <= max(8.0, config.rotation_signal_agreement_deg)
    )
    if hough_agrees:
        # Same peak: report the circular mean, but keep structure scores from
        # the projection peak so a 0.5° blend cannot overturn a solid 0° page.
        proj_angle = round(circular_mean_180([proj_angle, float(hough_angle)]), 2)

    diag["lines_at_projection"] = lines_at
    diag["lines_at_orthogonal"] = lines_ortho
    diag["comps_at_projection"] = n_at
    diag["comps_at_orthogonal"] = n_ortho
    diag["projection_angle"] = proj_angle
    diag["projection_score"] = proj_score
    diag["projection_margin"] = margin
    diag["hough_agrees_with_projection"] = hough_agrees

    line_ratio = lines_at / float(lines_at + lines_ortho + 1e-9)
    evidence = min(1.0, n_at / 80.0 + lines_at / 20.0)
    conf = float(
        np.clip(
            0.30 * margin
            + 0.25 * line_ratio
            + 0.20 * evidence
            + 0.25 * (1.0 if hough_agrees else 0.30),
            0.0,
            1.0,
        )
    )

    structure_ok = lines_at >= config.rotation_min_text_lines and (
        lines_at > lines_ortho + 2
        or (
            hough_agrees
            and margin >= config.rotation_min_peak_margin
            and lines_at >= lines_ortho
        )
    )
    weak_evidence = (
        n_at < config.rotation_min_text_components
        or not structure_ok
        or margin < config.rotation_min_peak_margin
    )
    if weak_evidence or conf < config.rotation_confidence_threshold:
        return GeometricRotationResult(
            angle_deg=None,
            confidence=round(min(conf, 0.49), 4),
            status="UNCERTAIN",
            scores=scores,
            diagnostics=diag,
            warning="Rotation could not be determined confidently",
        )

    # Residual angles inside the skew search window are not arbitrary
    # rotations. Snap them to 0° here so Stage 6 can deskew them and so a
    # 175° projection peak (the 180° partner of -5°) is not applied as a
    # near-upside-down correction of an already-upright page.
    snapped = float(proj_angle)
    dist_to_zero = min(snapped, 180.0 - snapped)
    if dist_to_zero <= float(config.max_skew_angle):
        diag["snapped_to_zero_from"] = snapped
        snapped = 0.0

    return GeometricRotationResult(
        angle_deg=round(float(snapped), 2),
        confidence=round(conf, 4),
        status="DETECTED",
        scores=scores,
        diagnostics=diag,
        warning=None,
    )


def overlay_components(gray: np.ndarray, bundle: GeometryBundle, angle: float | None) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in bundle.components:
        cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 180, 0), 1)
    for ln in bundle.lines:
        cv2.line(vis, (ln.x_min, int(ln.y_mean)), (ln.x_max, int(ln.y_mean)), (0, 220, 255), 1)
    if angle is not None:
        h, w = gray.shape
        rad = np.deg2rad(angle)
        # Draw the detected content axis.
        cx, cy = w / 2.0, h / 2.0
        dx, dy = np.cos(rad) * w * 0.4, np.sin(rad) * w * 0.4
        cv2.line(
            vis,
            (int(cx - dx), int(cy - dy)),
            (int(cx + dx), int(cy + dy)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            vis,
            f"geom {angle:.2f} deg",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return vis
