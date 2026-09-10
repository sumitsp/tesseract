"""Coarse rotation detection (0/90/180/270) via multi-signal voting — no OCR."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .components import extract_text_components, group_text_lines


def _rotate_gray(gray: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return gray
    if deg == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(deg)


def _projection_score(ink: np.ndarray) -> float:
    row = ink.sum(axis=1).astype(np.float64)
    col = ink.sum(axis=0).astype(np.float64)
    if row.sum() < 1:
        return 0.0
    row_n = row / (row.sum() + 1e-9)
    col_n = col / (col.sum() + 1e-9)
    row_var = float(np.var(row_n))
    col_var = float(np.var(col_n))
    # Peakiness: fraction of mass in top-k peaks.
    k = max(5, len(row) // 40)
    peaks = np.sort(row)[-k:].sum() / (row.sum() + 1e-9)
    structure = row_var / (col_var + 1e-12)
    return float(structure * (0.5 + 0.5 * peaks))


def _morphology_score(ink: np.ndarray) -> float:
    h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v)
    hs = float(cv2.countNonZero(horiz))
    vs = float(cv2.countNonZero(vert))
    return hs / (vs + 1.0)


def _component_score(ink: np.ndarray) -> tuple[float, int, int]:
    comps = extract_text_components(ink)
    n = len(comps)
    if n < 8:
        return 0.0, n, 0
    widths = np.array([c.w for c in comps], dtype=np.float64)
    heights = np.array([c.h for c in comps], dtype=np.float64)
    # Upright glyphs tend to be taller than wide on average for Latin/forms.
    aspect = float(np.median(heights) / (np.median(widths) + 1e-9))
    lines = group_text_lines(comps, min_comps=3)
    n_lines = len(lines)
    if n_lines >= 2:
        spans = [ln.x_max - ln.x_min for ln in lines]
        line_quality = float(np.median(spans)) / (ink.shape[1] + 1e-9)
        angles = np.array([abs(ln.angle_deg) for ln in lines], dtype=np.float64)
        flat = float(np.mean(angles < 8.0))
    else:
        line_quality = 0.0
        flat = 0.0
    score = (0.35 * min(aspect, 2.0) + 0.40 * min(n_lines / 20.0, 1.0)
             + 0.15 * min(line_quality * 2, 1.0) + 0.10 * flat)
    return float(score), n, n_lines


def _layout_upright_bias(ink: np.ndarray) -> float:
    """Mild 0-vs-180 signal: more structural mass in upper half is common."""
    h = ink.shape[0]
    top = float(ink[: h // 2].sum())
    bottom = float(ink[h // 2 :].sum())
    return (top - bottom) / (top + bottom + 1e-9)


def _punctuation_bias(ink: np.ndarray) -> float:
    """
    Tiny components often denser toward line baselines / lower page regions
    when upright; weak signal, used only for 0/180 tie-break.
    """
    comps = extract_text_components(ink, min_area=8, max_area_frac=0.002, min_h=3, max_h_frac=0.04)
    if len(comps) < 12:
        return 0.0
    ys = np.array([c.cy for c in comps], dtype=np.float64)
    h = ink.shape[0]
    lower = float(np.mean(ys > 0.55 * h))
    upper = float(np.mean(ys < 0.45 * h))
    return lower - upper


def score_orientation(ink: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    """Return per-orientation raw scores for {0,90,180,270}."""
    scores: dict[str, float] = {}
    details: dict[str, Any] = {}
    for deg in (0, 90, 180, 270):
        ink_r = _rotate_gray(ink, deg)
        # Projection + morphology on ink; components on cleaned ink.
        proj = _projection_score(ink_r)
        morph = _morphology_score(ink_r)
        comp, n_comp, n_lines = _component_score(ink_r)
        upright = _layout_upright_bias(ink_r)
        punct = _punctuation_bias(ink_r)

        # Base structural score (shared by 0 and 180 when page is flipped).
        base = 0.40 * proj + 0.25 * morph + 0.35 * comp

        # 0 vs 180: add weak layout biases only for those candidates.
        if deg in (0, 180):
            # Map bias so higher favors this orientation being "upright".
            # For deg=0, positive upright/punct favors keeping 0.
            # For deg=180, we evaluate after rotating 180, so same features
            # should look "upright" if 180 was the needed correction.
            bias = 0.08 * upright + 0.05 * punct
            total = base * (1.0 + bias)
        else:
            total = base

        scores[str(deg)] = float(max(total, 0.0))
        details[str(deg)] = {
            "proj": proj,
            "morph": morph,
            "comp": comp,
            "n_comp": n_comp,
            "n_lines": n_lines,
            "upright_bias": upright,
            "punct_bias": punct,
            "base": base,
            "total": float(max(total, 0.0)),
        }
    return scores | {"_details": details}  # type: ignore[return-value]


def detect_rotation(
    ink: np.ndarray,
    gray: np.ndarray,
    *,
    close_ratio: float = 1.08,
) -> tuple[int, float, bool, dict[str, Any]]:
    """
    Returns (rotation_deg, confidence, ambiguous_0_180, diagnostics).

    ``rotation`` is the clockwise correction to apply to the input image.
    """
    packed = score_orientation(ink, gray)
    details = packed.pop("_details")  # type: ignore[misc]
    scores = {int(k): float(v) for k, v in packed.items()}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_deg, best = ranked[0]
    second_deg, second = ranked[1]
    evidence = float(details[str(best_deg)]["n_comp"]) + 2.0 * float(details[str(best_deg)]["n_lines"])

    sep = (best - second) / (best + 1e-9)
    conf = float(np.clip(0.35 + 2.0 * sep + 0.01 * min(evidence, 40), 0.0, 1.0))

    ambiguous = False
    # Special handling: 0 vs 180 near-tie.
    s0, s180 = scores[0], scores[180]
    if max(s0, s180) > 0 and min(s0, s180) / (max(s0, s180) + 1e-9) > 0.92:
        if best_deg in (0, 180):
            ambiguous = True
            conf = min(conf, 0.45)
            # Prefer 0 when essentially tied (do not invent 180).
            if abs(s0 - s180) / (max(s0, s180) + 1e-9) < 0.04:
                best_deg = 0

    if evidence < 12:
        conf = min(conf, 0.35)
        ambiguous = True

    if best > 0 and best < second * close_ratio and best_deg != 0:
        # Weak winner — fall back to 0 rather than a shaky 90/270.
        if best_deg in (90, 270) and scores[0] > best * 0.85:
            best_deg = 0
            conf = min(conf, 0.5)
            ambiguous = True

    diag = {
        "rotation_scores": {str(k): v for k, v in scores.items()},
        "rotation_details": details,
        "rotation_best": best_deg,
        "rotation_second": second_deg,
        "rotation_evidence": evidence,
        "rotation_ambiguous_0_180": ambiguous,
    }
    return int(best_deg), float(conf), bool(ambiguous), diag
