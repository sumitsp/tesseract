"""Classical text-geometry primitives shared by the sequential detectors.

These helpers extract ink, glyph-like connected components, and text-line
candidates. They do not decide rotation, mirror, or skew — each detector
stage consumes these signals independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from image_preprocessing.orientation.angle_search import binarize, normalize_illumination

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


def extract_ink(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(normalized gray, raw ink, cleaned ink).

    Binarization lives in ``angle_search`` so every stage shares one ink
    definition; see its ``binarize`` docstring for why CLAHE is absent.
    """
    norm = normalize_illumination(gray)
    ink = binarize(gray)
    ink_clean = remove_tiny_noise(remove_page_borders(ink), min_area=8)
    return norm, ink, ink_clean


def extract_text_components(
    ink: np.ndarray,
    *,
    min_area: int = 8,
    max_area_frac: float = 0.02,
    min_h: int = 3,
    max_h_frac: float = 0.12,
    min_aspect: float = 0.08,
    max_aspect: float = 15.0,
) -> list[TextComponent]:
    """Glyph-scale connected components.

    Thresholds are deliberately permissive. Analysis copies run around 1400 px
    on the long edge, where body text is only 7-10 px tall and a glyph covers
    ~25 px; the stricter limits this replaced kept 49 components on a lab report
    holding roughly 30 text lines, which left line grouping with nothing to work
    with.
    """
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
