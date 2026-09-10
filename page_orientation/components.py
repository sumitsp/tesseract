"""Text-like connected components and approximate text-line grouping (no OCR)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
