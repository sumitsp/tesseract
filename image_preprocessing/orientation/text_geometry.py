"""Classical text-geometry primitives shared by the sequential detectors.

These helpers extract ink, glyph-like connected components, and text-line
candidates. They do not decide rotation, mirror, or skew — each detector
stage consumes these signals independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from image_preprocessing.utils.image_utils import (
    INTER_DOWNSAMPLE,
    rotate_bound,
    to_gray,
)


@dataclass
class TextComponent:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    angle_deg: float  # minAreaRect orientation, [0, 180)


@dataclass
class TextLine:
    components: list[TextComponent]
    y_mean: float
    x_min: int
    x_max: int
    angle_deg: float  # local line slant; positive = clockwise (y-down)


@dataclass
class GeometryBundle:
    gray: np.ndarray
    clahe: np.ndarray
    ink: np.ndarray
    ink_clean: np.ndarray
    scale: float
    analysis_size: tuple[int, int]
    components: list[TextComponent]
    lines: list[TextLine]


def resize_for_analysis(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray, 1.0
    scale = max_dim / float(longest)
    out = cv2.resize(
        gray,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=INTER_DOWNSAMPLE,
    )
    return out, scale


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
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
    inv = 255 - thr
    if cv2.countNonZero(inv) < cv2.countNonZero(thr):
        return inv
    return thr if cv2.countNonZero(thr) < cv2.countNonZero(inv) else inv


def combine_ink_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    combined = cv2.bitwise_or(a, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)


def remove_page_borders(ink: np.ndarray, margin_frac: float = 0.02) -> np.ndarray:
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


def remove_ruling_lines(ink: np.ndarray, min_line_len_frac: float = 0.12) -> np.ndarray:
    """Strip long table/form rules so detectors see glyphs, not borders."""
    h, w = ink.shape
    h_len = max(15, int(w * min_line_len_frac))
    v_len = max(15, int(h * min_line_len_frac))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines = cv2.dilate(cv2.bitwise_or(h_lines, v_lines), grow, iterations=1)
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def extract_ink(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = normalize_illumination(gray)
    clahe = apply_clahe(norm)
    ink = combine_ink_masks(_ink_from_adaptive(clahe), _ink_from_otsu(clahe))
    ink_clean = remove_tiny_noise(remove_ruling_lines(remove_page_borders(ink)), min_area=10)
    return clahe, ink, ink_clean


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
    h_img, w_img = ink.shape
    max_area = int(h_img * w_img * max_area_frac)
    max_h = int(h_img * max_h_frac)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps: list[TextComponent] = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if h < min_h or h > max_h or w < 2:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if w > 0.55 * w_img and h < 0.025 * h_img:
            continue
        if h > 0.55 * h_img and w < 0.025 * w_img:
            continue
        fill = area / float(w * h + 1e-9)
        if fill < 0.08 or fill > 0.95:
            continue
        patch = ink[y : y + h, x : x + w]
        angle = _min_area_rect_angle(patch)
        comps.append(
            TextComponent(
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                cx=float(centroids[i, 0]),
                cy=float(centroids[i, 1]),
                angle_deg=angle,
            )
        )
    return comps


def _min_area_rect_angle(patch: np.ndarray) -> float:
    pts = cv2.findNonZero(patch)
    if pts is None or len(pts) < 5:
        return 0.0
    rect = cv2.minAreaRect(pts)
    angle = float(rect[2])
    width, height = rect[1]
    # OpenCV angle is in [0, 90). Convert to the long-axis orientation in [0, 180).
    if width < height:
        angle = angle + 90.0
    return float(angle % 180.0)


def group_text_lines(
    comps: list[TextComponent],
    *,
    y_tol_frac: float = 0.55,
    min_comps: int = 3,
) -> list[TextLine]:
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
            slope, _ = np.polyfit(xs, ys, 1)
            angle = float(np.degrees(np.arctan(slope)))
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


def build_geometry(image: np.ndarray, max_dim: int) -> GeometryBundle:
    gray_full = to_gray(image)
    gray, scale = resize_for_analysis(gray_full, max_dim)
    clahe, ink, ink_clean = extract_ink(gray)
    comps = extract_text_components(ink_clean)
    lines = group_text_lines(comps)
    h, w = gray.shape
    return GeometryBundle(
        gray=gray,
        clahe=clahe,
        ink=ink,
        ink_clean=ink_clean,
        scale=scale,
        analysis_size=(w, h),
        components=comps,
        lines=lines,
    )


def component_mask(shape: tuple[int, int], comps: list[TextComponent], ink: np.ndarray | None = None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for c in comps:
        if ink is not None:
            mask[c.y : c.y + c.h, c.x : c.x + c.w] = ink[c.y : c.y + c.h, c.x : c.x + c.w]
        else:
            mask[c.y : c.y + c.h, c.x : c.x + c.w] = 255
    return mask


def horizontal_alignment_score(ink: np.ndarray) -> float:
    """Higher when ink concentrates into distinct horizontal bands (text lines)."""
    row = ink.sum(axis=1).astype(np.float64)
    col = ink.sum(axis=0).astype(np.float64)
    if row.sum() < 1:
        return 0.0
    row_n = row / (row.sum() + 1e-9)
    col_n = col / (col.sum() + 1e-9)
    row_var = float(np.var(row_n))
    col_var = float(np.var(col_n))
    k = max(5, len(row) // 40)
    peaks = float(np.sort(row)[-k:].sum() / (row.sum() + 1e-9))
    structure = row_var / (col_var + 1e-12)
    return float(structure * (0.5 + 0.5 * peaks))


def morphological_line_score(ink: np.ndarray) -> float:
    h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h)
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v)
    hs = float(cv2.countNonZero(horiz))
    vs = float(cv2.countNonZero(vert))
    return hs / (vs + 1.0)


def projection_score_at_angle(ink: np.ndarray, angle_ccw_deg: float) -> float:
    """Score of text-line alignment after a CCW rotation of ``angle_ccw_deg``."""
    rotated = rotate_bound(ink, angle_ccw_deg, interpolation=cv2.INTER_NEAREST, border=0)
    proj = horizontal_alignment_score(rotated)
    morph = morphological_line_score(rotated)
    return float(proj + 0.25 * morph)


def circular_mean_180(angles: list[float]) -> float:
    if not angles:
        return 0.0
    doubled = np.deg2rad(2.0 * np.array(angles, dtype=np.float64))
    c = float(np.mean(np.cos(doubled)))
    s = float(np.mean(np.sin(doubled)))
    return wrap_180(float(np.rad2deg(np.arctan2(s, c) / 2.0)))


def circular_distance_180(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def wrap_180(angle: float) -> float:
    return float(angle % 180.0)


def wrap_360(angle: float) -> float:
    return float(angle % 360.0)
