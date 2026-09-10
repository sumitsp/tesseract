"""
Page orientation detection + Azure batch correction — ONE FILE.

OpenCV + NumPy only. No OCR.

API:
    detector = PageOrientationDetector()
    result = detector.detect(image)      # dict
    corrected = detector.correct(image, result)

Sign convention:
    rotation = clockwise correction (0/90/180/270)
    tilt     = residual skew degrees; positive = clockwise
    mirror   = True means apply horizontal flip after rotation

Correction order: rotation -> mirror -> tilt
Tilt is applied only when |tilt| <= max_tilt_to_apply (default 5).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Azure imports are deferred to connect_azure() so local detect/correct works
# without azure packages installed.
# ---- result.py ----
@dataclass
class OrientationResult:
    """
    Orientation estimate for a document page.

    Sign convention
    ---------------
    ``tilt`` is residual fine skew in degrees after coarse rotation + mirror.
    **Positive tilt = clockwise** (text lines lean down to the right in
    image coordinates where y increases downward).

    Correction applies an OpenCV counter-clockwise rotation of ``+tilt``
    degrees (which undoes a clockwise lean of the same magnitude).
    """

    rotation: int  # 0 | 90 | 180 | 270  (clockwise correction to apply)
    tilt: float
    mirror: bool

    rotation_confidence: float
    tilt_confidence: float
    mirror_confidence: float
    overall_confidence: float
    needs_review: bool

    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_diagnostics: bool = False) -> dict[str, Any]:
        data = {
            "rotation": int(self.rotation),
            "tilt": float(self.tilt),
            "mirror": bool(self.mirror),
            "rotation_confidence": float(self.rotation_confidence),
            "tilt_confidence": float(self.tilt_confidence),
            "mirror_confidence": float(self.mirror_confidence),
            "overall_confidence": float(self.overall_confidence),
            "needs_review": bool(self.needs_review),
        }
        if include_diagnostics:
            data["diagnostics"] = self.diagnostics
        return data

    def as_public_dict(self) -> dict[str, Any]:
        return self.to_dict(include_diagnostics=False)

# ---- preprocess.py ----
@dataclass
class PreprocessBundle:
    """Analysis-resolution images/masks used by detectors."""

    gray: np.ndarray
    clahe: np.ndarray
    ink: np.ndarray           # text-like ink, white=ink on black
    ink_clean: np.ndarray     # border/noise cleaned
    scale: float              # analysis / original linear scale
    analysis_size: tuple[int, int]  # (w, h)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_for_analysis(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray, 1.0
    scale = max_dim / float(longest)
    out = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by large-kernel background estimate to flatten uneven lighting."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 20) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg = np.maximum(bg, 1)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _ink_from_adaptive(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )


def _ink_from_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Decide polarity: ink should be minority (typically).
    inv = 255 - thr
    if cv2.countNonZero(inv) < cv2.countNonZero(thr):
        return inv
    return thr if cv2.countNonZero(thr) < cv2.countNonZero(inv) else inv


def combine_ink_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union with light opening to suppress speckles without killing thin strokes."""
    combined = cv2.bitwise_or(a, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)


def remove_page_borders(ink: np.ndarray, margin_frac: float = 0.02) -> np.ndarray:
    """Zero out a thin outer border band (page edges / scanner frames)."""
    h, w = ink.shape
    m_y = max(2, int(h * margin_frac))
    m_x = max(2, int(w * margin_frac))
    out = ink.copy()
    out[:m_y, :] = 0
    out[-m_y:, :] = 0
    out[:, :m_x] = 0
    out[:, -m_x:] = 0
    return out


def remove_tiny_noise(ink: np.ndarray, min_area: int = 12) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = np.zeros_like(ink)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def preprocess(image: np.ndarray, analysis_max_dimension: int = 1800) -> PreprocessBundle:
    gray_full = to_gray(image)
    gray, scale = resize_for_analysis(gray_full, analysis_max_dimension)
    norm = normalize_illumination(gray)
    clahe = apply_clahe(norm)
    ink_a = _ink_from_adaptive(clahe)
    ink_b = _ink_from_otsu(clahe)
    ink = combine_ink_masks(ink_a, ink_b)
    ink_borderless = remove_page_borders(ink)
    ink_clean = remove_tiny_noise(ink_borderless, min_area=10)
    h, w = gray.shape
    return PreprocessBundle(
        gray=gray,
        clahe=clahe,
        ink=ink,
        ink_clean=ink_clean,
        scale=scale,
        analysis_size=(w, h),
    )

# ---- components.py ----
@dataclass
class TextComponent:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float


@dataclass
class TextLine:
    components: list[TextComponent]
    y_mean: float
    x_min: int
    x_max: int
    angle_deg: float  # local line slant, clockwise-positive in image coords


def extract_text_components(
    ink: np.ndarray,
    *,
    min_area: int = 20,
    max_area_frac: float = 0.08,
    min_h: int = 6,
    max_h_frac: float = 0.12,
    min_aspect: float = 0.12,
    max_aspect: float = 12.0,
) -> list[TextComponent]:
    """Filter CCs that look like glyphs / glyph clusters (not tables/borders)."""
    h_img, w_img = ink.shape
    max_area = int(h_img * w_img * max_area_frac)
    max_h = int(h_img * max_h_frac)

    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps: list[TextComponent] = []
    for i in range(1, num):
        x, y, w, h, area = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        if area < min_area or area > max_area:
            continue
        if h < min_h or h > max_h:
            continue
        if w < 2:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        # Reject long thin border-like bars.
        if w > 0.55 * w_img and h < 0.025 * h_img:
            continue
        if h > 0.55 * h_img and w < 0.025 * w_img:
            continue
        fill = area / float(w * h + 1e-9)
        if fill < 0.08 or fill > 0.95:
            continue
        comps.append(
            TextComponent(
                x=x, y=y, w=w, h=h, area=area,
                cx=float(centroids[i, 0]), cy=float(centroids[i, 1]),
            )
        )
    return comps


def group_text_lines(
    comps: list[TextComponent],
    *,
    y_tol_frac: float = 0.55,
    min_comps: int = 3,
) -> list[TextLine]:
    """Greedy horizontal grouping of nearby components into line candidates."""
    if not comps:
        return []
    ordered = sorted(comps, key=lambda c: (c.cy, c.cx))
    used = [False] * len(ordered)
    lines: list[TextLine] = []

    for i, seed in enumerate(ordered):
        if used[i]:
            continue
        y_tol = max(3.0, seed.h * y_tol_frac)
        group = [seed]
        used[i] = True
        # Expand by y proximity and left-to-right chaining.
        changed = True
        while changed:
            changed = False
            y_mean = float(np.mean([c.cy for c in group]))
            x_right = max(c.x + c.w for c in group)
            median_h = float(np.median([c.h for c in group]))
            for j, cand in enumerate(ordered):
                if used[j]:
                    continue
                if abs(cand.cy - y_mean) > max(y_tol, median_h * 0.7):
                    continue
                # Allow modest gaps; reject far jumps.
                gap = cand.x - x_right
                if gap > median_h * 8:
                    continue
                if cand.x + cand.w < min(c.x for c in group) - median_h * 8:
                    continue
                group.append(cand)
                used[j] = True
                changed = True
                x_right = max(c.x + c.w for c in group)

        if len(group) < min_comps:
            continue
        group = sorted(group, key=lambda c: c.cx)
        xs = np.array([c.cx for c in group], dtype=np.float64)
        ys = np.array([c.cy for c in group], dtype=np.float64)
        if len(group) >= 2 and float(np.ptp(xs)) > 1.0:
            # Fit y = a*x + b; slope a ≈ tan(angle). Image y-down => +slope = clockwise.
            a, _b = np.polyfit(xs, ys, 1)
            angle = float(np.degrees(np.arctan(a)))
        else:
            angle = 0.0
        lines.append(
            TextLine(
                components=group,
                y_mean=float(np.mean(ys)),
                x_min=min(c.x for c in group),
                x_max=max(c.x + c.w for c in group),
                angle_deg=angle,
            )
        )
    return lines


def component_mask(ink_shape: tuple[int, int], comps: list[TextComponent]) -> np.ndarray:
    mask = np.zeros(ink_shape, dtype=np.uint8)
    for c in comps:
        mask[c.y : c.y + c.h, c.x : c.x + c.w] = 255
    return mask

# ---- rotation_detect.py ----
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

# ---- mirror_detect.py ----
def _line_features(ink: np.ndarray) -> dict[str, float]:
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=3)
    w = float(ink.shape[1])
    if len(lines) < 2:
        return {
            "n_lines": float(len(lines)),
            "left_align": 0.0,
            "starts_leftness": 0.5,
            "span_score": 0.0,
            "spacing": 0.0,
        }

    lefts = np.array([ln.x_min for ln in lines], dtype=np.float64)
    spans = np.array([ln.x_max - ln.x_min for ln in lines], dtype=np.float64)
    med = float(np.median(lefts))
    mad = float(np.median(np.abs(lefts - med))) + 1.0
    # Tighter left-edge cluster => higher
    left_align = float(np.clip(w / (mad * 8.0), 0.0, 1.0))
    # Prefer starts in the left half (LTR paragraphs / form labels).
    starts_leftness = float(np.clip(1.0 - (med / (0.55 * w + 1e-9)), 0.0, 1.0))
    span_score = float(np.clip(np.median(spans) / (0.55 * w + 1e-9), 0.0, 1.0))

    gaps: list[float] = []
    for ln in lines:
        xs = sorted(c.x + c.w for c in ln.components)
        for a, b in zip(xs, xs[1:]):
            g = b - a
            if 1 < g < w * 0.25:
                gaps.append(float(g))
    if len(gaps) >= 6:
        g = np.array(gaps, dtype=np.float64)
        spacing = float(np.clip(1.0 / (1.0 + np.std(g) / (np.mean(g) + 1e-9)), 0.0, 1.0))
    else:
        spacing = 0.0

    return {
        "n_lines": float(len(lines)),
        "left_align": left_align,
        "starts_leftness": starts_leftness,
        "span_score": span_score,
        "spacing": spacing,
    }


def _top_heaviness(ink: np.ndarray) -> float:
    h = ink.shape[0]
    top = float(ink[: h // 2].sum())
    bottom = float(ink[h // 2 :].sum())
    return (top - bottom) / (top + bottom + 1e-9)


def structural_mirror_score(ink: np.ndarray) -> dict[str, float]:
    """
    Higher = more consistent with normal LTR document geometry
    (left-aligned line starts, tight left edge, reasonable spans).
    """
    feat = _line_features(ink)
    total = (
        0.40 * feat["left_align"]
        + 0.35 * feat["starts_leftness"]
        + 0.15 * feat["span_score"]
        + 0.10 * feat["spacing"]
    )
    return {**feat, "total": float(total)}


def detect_mirror(
    ink: np.ndarray,
    *,
    win_ratio: float = 1.10,
) -> tuple[bool, float, bool, dict[str, Any]]:
    """
    Returns (mirror, confidence, ambiguous, diagnostics).

    ``mirror=True`` only when the flipped page clearly looks more LTR-like.
    Ambiguous / weak evidence => mirror=False, needs_review.
    """
    normal = structural_mirror_score(ink)
    flipped = structural_mirror_score(cv2.flip(ink, 1))

    n = normal["total"]
    f = flipped["total"]
    best = max(n, f)
    sep = (best - min(n, f)) / (best + 1e-9)

    # Votes that flipped is more LTR-like.
    votes = 0
    if f > n * win_ratio:
        votes += 1
    if flipped["starts_leftness"] > normal["starts_leftness"] + 0.08:
        votes += 1
    if flipped["left_align"] > normal["left_align"] * 1.08:
        votes += 1

    mirror = votes >= 2 and f > n
    ambiguous = (
        sep < 0.07
        or best < 0.15
        or votes <= 1
        or normal["n_lines"] < 3
    )
    conf = float(np.clip(0.20 + 2.8 * sep + 0.12 * votes, 0.0, 1.0))
    if ambiguous:
        conf = min(conf, 0.40)
        mirror = False

    diag = {
        "mirror_scores": {"normal": normal, "flipped": flipped},
        "mirror_separation": sep,
        "mirror_votes": votes,
        "mirror_ambiguous": ambiguous,
        "top_heaviness": _top_heaviness(ink),
    }
    return bool(mirror), float(conf), bool(ambiguous), diag

# ---- tilt_detect.py ----
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

# ---- correct.py ----
def _border_value(image: np.ndarray) -> int | tuple[int, int, int]:
    if image.ndim == 2:
        return 255
    return (255, 255, 255)


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def rotate_coarse_cw(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg == 0:
        return image
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported coarse rotation: {rotation_deg}")


def rotate_fine(
    image: np.ndarray,
    tilt_clockwise_deg: float,
    *,
    expand: bool = True,
) -> np.ndarray:
    """
    Undo clockwise residual skew ``tilt_clockwise_deg``.

    OpenCV ``getRotationMatrix2D`` uses counter-clockwise-positive angles,
    so we pass ``+tilt_clockwise_deg`` to rotate the image CCW and straighten
    clockwise-leaning content.
    """
    if abs(tilt_clockwise_deg) < 1e-4:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    # CCW by +tilt undoes clockwise lean of +tilt.
    matrix = cv2.getRotationMatrix2D(center, float(tilt_clockwise_deg), 1.0)
    if expand:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        size = (new_w, new_h)
    else:
        size = (w, h)
    return cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_border_value(image),
    )


def correct_image(
    image: np.ndarray,
    result: OrientationResult | dict,
    *,
    expand: bool = True,
    apply_tilt: bool = True,
    max_tilt_abs: float | None = None,
) -> np.ndarray:
    """
    Correction order:
      1) coarse rotation
      2) mirror
      3) fine tilt

    Mirror is applied after rotation because mirror features are defined on
    upright (LTR) page geometry. Rotation and horizontal flip are not
    interchangeable for 90/270.
    """
    if isinstance(result, dict):
        mirror = bool(result["mirror"])
        rotation = int(result["rotation"])
        tilt = float(result["tilt"])
    else:
        mirror = result.mirror
        rotation = result.rotation
        tilt = result.tilt

    out = rotate_coarse_cw(image, rotation)
    if mirror:
        out = flip_horizontal(out)

    if apply_tilt:
        if max_tilt_abs is not None and abs(tilt) > max_tilt_abs:
            pass
        else:
            out = rotate_fine(out, tilt, expand=expand)
    return out

# ---- detector.py ----
class PageOrientationDetector:
    """
    Detect coarse rotation (0/90/180/270), horizontal mirror, and residual tilt
    using OpenCV structural signals only (no OCR / no ML models).

    Typical usage::

        detector = PageOrientationDetector()
        result = detector.detect(bgr_or_gray)
        corrected = detector.correct(image, result)
    """

    def __init__(
        self,
        *,
        analysis_max_dimension: int = 1800,
        debug: bool = False,
        debug_dir: str | Path | None = None,
        review_confidence_threshold: float = 0.45,
        max_tilt_to_apply: float | None = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        analysis_max_dimension:
            Longest side of the analysis image (detection only).
        debug:
            If True, write debug artifacts when ``debug_dir`` is set.
        debug_dir:
            Folder for optional debug images/JSON.
        review_confidence_threshold:
            Overall confidence below this forces ``needs_review``.
        max_tilt_to_apply:
            If set, ``correct()`` skips fine tilt when |tilt| exceeds this
            (detection still reports the measured tilt). ``None`` = always apply.
        """
        self.analysis_max_dimension = int(analysis_max_dimension)
        self.debug = bool(debug)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.review_confidence_threshold = float(review_confidence_threshold)
        self.max_tilt_to_apply = max_tilt_to_apply

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """
        Detect orientation. Returns a public dict (see ``OrientationResult``).

        Pipeline: preprocess → rotation → mirror → coarse-correct analysis
        image → fine tilt → confidence / review flags.
        """
        result = self.detect_result(image)
        return result.as_public_dict()

    def detect_result(self, image: np.ndarray) -> OrientationResult:
        if image is None or not hasattr(image, "shape"):
            raise ValueError("detect() requires an OpenCV image (BGR or gray)")

        bundle = preprocess(image, self.analysis_max_dimension)
        ink = bundle.ink_clean
        gray = bundle.clahe

        rotation, rot_conf, rot_amb, rot_diag = detect_rotation(ink, gray)

        # Evaluate mirror on the coarsely rotated analysis image so LTR
        # structural cues are meaningful (especially for 90/270 inputs).
        ink_r = rotate_coarse_cw(ink, rotation)
        gray_r = rotate_coarse_cw(gray, rotation)
        mirror, mir_conf, mir_amb, mir_diag = detect_mirror(ink_r)

        ink_c = flip_horizontal(ink_r) if mirror else ink_r
        gray_c = flip_horizontal(gray_r) if mirror else gray_r

        tilt, tilt_conf, tilt_amb, tilt_diag = detect_tilt(ink_c, gray_c)

        overall = float(
            0.40 * rot_conf + 0.30 * tilt_conf + 0.30 * mir_conf
        )

        # Sparse / blank page?
        ink_density = float(cv2.countNonZero(ink)) / float(ink.size)
        sparse = ink_density < 0.004

        needs_review = bool(
            rot_amb
            or mir_amb
            or tilt_amb
            or sparse
            or overall < self.review_confidence_threshold
        )

        diagnostics: dict[str, Any] = {
            "analysis_size": bundle.analysis_size,
            "scale": bundle.scale,
            "ink_density": ink_density,
            **rot_diag,
            **mir_diag,
            **tilt_diag,
            "sign_convention": {
                "rotation": "clockwise correction degrees to apply",
                "tilt": "clockwise residual skew degrees (positive = CW)",
                "mirror": "True means apply horizontal flip",
            },
        }

        result = OrientationResult(
            rotation=int(rotation),
            tilt=float(round(tilt, 3)),
            mirror=bool(mirror),
            rotation_confidence=float(round(rot_conf, 4)),
            tilt_confidence=float(round(tilt_conf, 4)),
            mirror_confidence=float(round(mir_conf, 4)),
            overall_confidence=float(round(overall, 4)),
            needs_review=needs_review,
            diagnostics=diagnostics,
        )

        if self.debug and self.debug_dir is not None:
            self._write_debug(bundle.ink_clean, gray, result)

        return result

    def correct(
        self,
        image: np.ndarray,
        result: OrientationResult | dict[str, Any],
        *,
        expand: bool = True,
    ) -> np.ndarray:
        """
        Apply corrections at full resolution.

        Order: coarse rotation → mirror → fine tilt (subject to max_tilt_to_apply).
        """
        return correct_image(
            image,
            result,
            expand=expand,
            apply_tilt=True,
            max_tilt_abs=self.max_tilt_to_apply,
        )

    def _write_debug(
        self,
        ink: np.ndarray,
        gray: np.ndarray,
        result: OrientationResult,
    ) -> None:
        assert self.debug_dir is not None
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / "ink_mask.png"), ink)
        cv2.imwrite(str(self.debug_dir / "gray_clahe.png"), gray)
        # JSON diagnostics
        import json

        (self.debug_dir / "result.json").write_text(
            json.dumps(result.to_dict(include_diagnostics=True), indent=2),
            encoding="utf-8",
        )

# ---- azure batch runner ----
# ============================================================
# AZURE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rotation_report.csv")
START_FROM = ""
JPG_SUFFIXES = {".jpg", ".jpeg"}

# Tilt is measured always; applied only when |tilt| <= this (detector setting).
MAX_TILT_TO_APPLY = 5.0

CSV_FIELDS = [
    "folder",
    "filename",
    "rotation_deg",
    "tilt_angle_deg",
    "mirrored",
    "rotation_confidence",
    "tilt_confidence",
    "mirror_confidence",
    "overall_confidence",
    "needs_review",
    "tilt_applied",
    "status",
    "error",
    "output_path",
]


def log(message: str) -> None:
    print(message, flush=True)


def connect_azure():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    log("Connecting to Azure Blob Storage...")
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        log("Azure connection created.")
        return container_client
    except Exception as exc:
        raise SystemExit(f"ERROR connecting to Azure:\n{exc}") from exc


def list_folder_blobs(container_client: Any) -> dict[str, list[str]]:
    log("=" * 70)
    log("Scanning Azure Blob")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log("=" * 70)

    folder_blobs: dict[str, list[str]] = {}
    try:
        for blob in container_client.list_blobs(name_starts_with=PREFIX):
            relative_path = blob.name[len(PREFIX) :]
            parts = relative_path.split("/")
            if len(parts) < 2:
                continue
            folder_name = parts[0]
            filename = parts[-1]
            if Path(filename).suffix.lower() not in JPG_SUFFIXES:
                continue
            folder_blobs.setdefault(folder_name, []).append(blob.name)
    except Exception as exc:
        raise SystemExit(f"ERROR while reading blobs:\n{exc}") from exc
    return folder_blobs


def list_chart_folders(folder_blobs: dict[str, list[str]]) -> list[str]:
    folders = sorted(folder_blobs.keys(), key=lambda name: name.lower())
    if not START_FROM:
        return folders
    start_index = next((i for i, f in enumerate(folders) if f == START_FROM), None)
    if start_index is None:
        raise SystemExit(f"START_FROM folder not found under prefix: {START_FROM}")
    log(f"Starting at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def jpg_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_jpgs(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=jpg_sort_key)
    return images


def download_blob(container_client: Any, blob_name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(container_client.download_blob(blob_name).readall())
    return dest


def process_folder(
    writer: csv.DictWriter,
    csv_file,
    container_client: Any,
    detector: PageOrientationDetector,
    output_dir: Path,
    folder_name: str,
    blob_names: list[str],
) -> None:
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    log(f"  {len(blob_names)} images -> {out_folder}")

    with tempfile.TemporaryDirectory(prefix=f"rotation_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            row = {
                "folder": folder_name,
                "filename": filename,
                "status": "ok",
                "error": "",
            }
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                image = cv2.imread(str(local_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (unreadable/corrupt image)")

                result = detector.detect_result(image)
                corrected = detector.correct(image, result, expand=True)
                out_path = out_folder / filename
                if not cv2.imwrite(str(out_path), corrected):
                    raise ValueError(f"cv2.imwrite failed: {out_path}")

                tilt_applied = (
                    abs(result.tilt) >= 0.15
                    and abs(result.tilt) <= (detector.max_tilt_to_apply or 1e9)
                )
                row.update(
                    {
                        "rotation_deg": result.rotation,
                        "tilt_angle_deg": result.tilt,
                        "mirrored": "Yes" if result.mirror else "No",
                        "rotation_confidence": result.rotation_confidence,
                        "tilt_confidence": result.tilt_confidence,
                        "mirror_confidence": result.mirror_confidence,
                        "overall_confidence": result.overall_confidence,
                        "needs_review": "Yes" if result.needs_review else "No",
                        "tilt_applied": "Yes" if tilt_applied else "No",
                        "output_path": str(out_path),
                    }
                )
                log(
                    f"    rot={result.rotation} tilt={result.tilt} "
                    f"mirror={row['mirrored']} conf={result.overall_confidence:.2f} "
                    f"review={row['needs_review']} -> {out_path.name}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["output_path"] = ""
                for key in CSV_FIELDS:
                    row.setdefault(key, "")
                log(f"    ERROR: {row['error']}")
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            csv_file.flush()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log("=== PAGE ORIENTATION (OpenCV PageOrientationDetector) ===")
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {OUTPUT_DIR}")
    log(f"CSV:    {CSV_PATH}")
    log(f"Max tilt applied: {MAX_TILT_TO_APPLY}°")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / JPG blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    detector = PageOrientationDetector(
        analysis_max_dimension=1800,
        max_tilt_to_apply=MAX_TILT_TO_APPLY,
        debug=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        for folder_name in chart_folders:
            images = list_jpgs(folder_blobs, folder_name)
            if not images:
                log(f"Folder: {folder_name} (no JPGs, skipped)")
                continue
            log(f"Folder: {folder_name}")
            process_folder(
                writer, csv_file, container_client, detector,
                OUTPUT_DIR, folder_name, images,
            )

    log(f"Done. Report: {CSV_PATH}")


if __name__ == "__main__":
    main()
