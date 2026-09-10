"""Fine tilt/skew detection after coarse orientation — no OCR."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .components import extract_text_components, group_text_lines


def _clamp_near_horizontal(angle: float) -> float | None:
    """Map any line angle into near-horizontal residual skew, or reject."""
    # Normalize to (-90, 90]
    a = ((angle + 90) % 180) - 90
    if abs(a) <= 20:
        return float(a)
    if abs(a) >= 160:
        # nearly 180 — fold
        a2 = a - 180 if a > 0 else a + 180
        if abs(a2) <= 20:
            return float(a2)
    return None


def hough_tilt(ink: np.ndarray, gray: np.ndarray) -> tuple[float | None, float]:
    """
    Weighted median of near-horizontal Hough segments.
    Returns (angle_clockwise_deg or None, confidence).
    """
    edges = cv2.Canny(gray, 50, 150)
    # Suppress outer border influence.
    h, w = edges.shape
    m = max(3, int(0.03 * min(h, w)))
    edges[:m, :] = 0
    edges[-m:, :] = 0
    edges[:, :m] = 0
    edges[:, -m:] = 0

    min_len = max(30, w // 6)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=10,
    )
    if lines is None:
        return None, 0.0

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        # Reject near-full-width border lines sitting at top/bottom.
        y_mid = 0.5 * (y1 + y2)
        if length > 0.85 * w and (y_mid < 0.08 * h or y_mid > 0.92 * h):
            continue
        raw = math.degrees(math.atan2(dy, dx))
        a = _clamp_near_horizontal(raw)
        if a is None:
            continue
        angles.append(a)
        weights.append(length)

    if len(angles) < 5:
        return None, 0.0

    ang = np.array(angles, dtype=np.float64)
    wts = np.array(weights, dtype=np.float64)
    order = np.argsort(ang)
    ang_s, w_s = ang[order], wts[order]
    cdf = np.cumsum(w_s)
    median = float(ang_s[np.searchsorted(cdf, 0.5 * cdf[-1])])

    # Agreement.
    close = np.abs(ang - median) <= 2.0
    frac = float(close.mean())
    if frac < 0.45:
        return None, 0.0
    conf = float(np.clip(0.2 + 0.6 * frac + 0.01 * min(len(angles), 40), 0, 1))
    return median, conf


def component_line_tilt(ink: np.ndarray) -> tuple[float | None, float]:
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=4)
    if len(lines) < 3:
        return None, 0.0
    angles = np.array([ln.angle_deg for ln in lines], dtype=np.float64)
    # Keep near-horizontal estimates only.
    angles = angles[np.abs(angles) <= 15]
    if len(angles) < 3:
        return None, 0.0
    med = float(np.median(angles))
    mad = float(np.median(np.abs(angles - med))) + 1e-6
    inliers = angles[np.abs(angles - med) <= 2.5 * mad]
    if len(inliers) < 3:
        return None, 0.0
    final = float(np.median(inliers))
    conf = float(np.clip(0.25 + 0.05 * len(inliers), 0, 1))
    return final, conf


def projection_refine(
    ink: np.ndarray,
    seed_deg: float,
    *,
    search: float = 1.0,
    coarse_step: float = 0.25,
    fine_step: float = 0.05,
) -> tuple[float, float]:
    """
    Search local angles around seed; maximize horizontal text-line projection score
    on the text/component mask (not raw black pixels).
    """
    comps = extract_text_components(ink)
    if len(comps) >= 10:
        mask = np.zeros_like(ink)
        for c in comps:
            mask[c.y : c.y + c.h, c.x : c.x + c.w] = ink[c.y : c.y + c.h, c.x : c.x + c.w]
        work = mask
    else:
        work = ink

    def score_at(angle: float) -> float:
        # Rotate mask by -angle (undo clockwise lean) using OpenCV CCW-positive = +angle.
        h, w = work.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(work, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        row = rot.sum(axis=1).astype(np.float64)
        if row.sum() < 1:
            return 0.0
        row /= row.sum()
        # Peak concentration: variance + top-k mass.
        var = float(np.var(row))
        k = max(5, len(row) // 50)
        peak = float(np.sort(row)[-k:].sum())
        return var * (0.4 + 0.6 * peak)

    best_a, best_s = seed_deg, score_at(seed_deg)
    for a in np.arange(seed_deg - search, seed_deg + search + 1e-9, coarse_step):
        s = score_at(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
    # Fine search.
    for a in np.arange(best_a - coarse_step, best_a + coarse_step + 1e-9, fine_step):
        s = score_at(float(a))
        if s > best_s:
            best_s, best_a = s, float(a)

    # Confidence from sharpness vs neighbors.
    neigh = [score_at(best_a - 0.5), score_at(best_a + 0.5), best_s]
    conf = float(np.clip((best_s - np.mean(neigh[:2])) / (best_s + 1e-9) * 2.0, 0, 1))
    return float(best_a), conf


def regional_tilts(ink: np.ndarray, gray: np.ndarray) -> list[float]:
    """3x3 regional Hough/component estimates; skip empty regions."""
    h, w = ink.shape
    ys = [0, h // 3, 2 * h // 3, h]
    xs = [0, w // 3, 2 * w // 3, w]
    angles: list[float] = []
    for i in range(3):
        for j in range(3):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            patch_ink = ink[y0:y1, x0:x1]
            patch_gray = gray[y0:y1, x0:x1]
            if cv2.countNonZero(patch_ink) < 200:
                continue
            a1, c1 = hough_tilt(patch_ink, patch_gray)
            a2, c2 = component_line_tilt(patch_ink)
            cands = []
            if a1 is not None and c1 >= 0.25:
                cands.append(a1)
            if a2 is not None and c2 >= 0.25:
                cands.append(a2)
            if cands:
                angles.append(float(np.median(cands)))
    return angles


def robust_aggregate(angles: list[float]) -> tuple[float | None, float]:
    if not angles:
        return None, 0.0
    arr = np.array(angles, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-6
    inliers = arr[np.abs(arr - med) <= 2.5 * mad]
    if len(inliers) == 0:
        return None, 0.0
    final = float(np.median(inliers))
    agree = float(len(inliers) / len(arr))
    conf = float(np.clip(0.2 + 0.7 * agree + 0.02 * len(inliers), 0, 1))
    # Strong disagreement → lower confidence.
    if len(arr) >= 4 and float(np.std(arr)) > 2.0:
        conf = min(conf, 0.4)
    return final, conf


def detect_tilt(
    ink: np.ndarray,
    gray: np.ndarray,
) -> tuple[float, float, bool, dict[str, Any]]:
    """
    Returns (tilt_deg_clockwise, confidence, needs_review_flag, diagnostics).

    Positive tilt = clockwise residual skew of content.
    """
    hough_a, hough_c = hough_tilt(ink, gray)
    comp_a, comp_c = component_line_tilt(ink)
    region_as = regional_tilts(ink, gray)
    region_a, region_c = robust_aggregate(region_as)

    seeds: list[tuple[float, float]] = []
    if hough_a is not None:
        seeds.append((hough_a, hough_c))
    if comp_a is not None:
        seeds.append((comp_a, comp_c))
    if region_a is not None:
        seeds.append((region_a, region_c))

    if not seeds:
        return 0.0, 0.15, True, {
            "tilt_hough": hough_a,
            "tilt_components": comp_a,
            "tilt_region_angles": region_as,
            "tilt_reason": "insufficient_evidence",
        }

    # Weighted seed.
    seed = float(np.average([a for a, _ in seeds], weights=[c for _, c in seeds]))
    refined, refine_c = projection_refine(ink, seed)

    # Final aggregate of refined + regional.
    pool = [refined] + region_as
    final, agg_c = robust_aggregate(pool)
    if final is None:
        final = refined

    conf = float(np.clip(0.4 * agg_c + 0.3 * refine_c + 0.3 * max(c for _, c in seeds), 0, 1))
    disagree = len(region_as) >= 4 and float(np.std(region_as)) > 2.5
    needs = disagree or conf < 0.35 or (hough_a is None and comp_a is None)

    # Tiny angles → treat as 0.
    if abs(final) < 0.15:
        final = 0.0

    diag = {
        "tilt_hough": hough_a,
        "tilt_hough_conf": hough_c,
        "tilt_components": comp_a,
        "tilt_component_conf": comp_c,
        "tilt_region_angles": region_as,
        "tilt_seed": seed,
        "tilt_refined": refined,
        "tilt_final": final,
        "tilt_disagreement": disagree,
    }
    return float(final), float(conf), bool(needs), diag
