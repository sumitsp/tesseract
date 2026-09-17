"""Projection-profile angle estimation — the geometric core of the orientation stages.

Both the arbitrary-rotation stage and the fine-tilt stage measure angles with the
same primitive: project ink pixel coordinates onto a rotated axis and score how
tightly the projection concentrates into bands. Text lines produce a sharp peak
when they are horizontal, so the maximising angle is the page's skew.

Sign convention
---------------
Angles are the *clockwise* offset of content from upright, matching
``utils.image_utils`` (correction = rotate counter-clockwise by that amount).
For a point (x, y) in image coordinates (y down), rotating the image
counter-clockwise by ``a`` maps it to ``y' = -x sin a + y cos a``, so scoring
``y'`` at angle ``a`` asks "how horizontal would text lines be if we corrected
by ``a``". ``a`` is therefore directly the clockwise content offset.

Why coordinates and not warped images
-------------------------------------
Rotating the coordinate list is exact and vectorised: a 90-step sweep over 80k
points costs milliseconds, where 90 ``warpAffine`` calls cost seconds. It also
avoids resampling the mask 90 times, which blurs thin strokes and biases the
result toward whichever angle happens to alias favourably.

Two calibrated details that matter more than they look
------------------------------------------------------
``BIN_SIZE`` is 1.0 px deliberately. At 0.5 px, integer pixel coordinates leave
every other bin empty at exactly 0/90 degrees, which inflates the score there and
pins the answer to 0 regardless of the real skew — an artefact that looks like
excellent accuracy on upright pages and hides real skew everywhere else.

``sum(p**2)`` is used unnormalised by bin count. Multiplying by the bin count to
"normalise for extent" rewards the larger diagonal projection extent instead, and
makes -45 degrees win on any page whose text evidence is weak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from image_preprocessing.utils.image_utils import INTER_DOWNSAMPLE, to_gray

# Histogram bin width in analysis pixels. See module docstring before changing.
BIN_SIZE = 1.0

# Cap on projected points. Beyond this the estimate stops improving and the
# sweep just gets slower.
MAX_POINTS = 80_000

# Below this there is not enough ink to measure an angle from.
MIN_POINTS = 300


@dataclass
class AngleEstimate:
    """Result of one angle search."""

    angle_cw_deg: float | None
    confidence: float
    peak_margin: float
    """(peak - median) / peak over the coarse sweep: how much the best angle
    stands out from a typical angle on this page."""
    runner_up_ratio: float
    """Best score found more than ``AMBIGUITY_SEPARATION_DEG`` away from the
    peak, divided by the peak score. Near 1.0 means two rival structures."""
    n_points: int
    diagnostics: dict = field(default_factory=dict)


# A rival peak must be at least this far from the winner to count as a genuinely
# different structure rather than the shoulder of the same peak.
AMBIGUITY_SEPARATION_DEG = 2.5


# --------------------------------------------------------------------------- ink
def analysis_gray(image: np.ndarray, max_dim: int) -> np.ndarray:
    gray = to_gray(image)
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray
    scale = max_dim / float(longest)
    return cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=INTER_DOWNSAMPLE,
    )


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven scanner lighting by dividing out a blurred background."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 20) | 1)
    bg = np.maximum(cv2.GaussianBlur(gray, (k, k), 0), 1)
    norm = gray.astype(np.float32) / bg.astype(np.float32) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask, white on black.

    No CLAHE. Equalising an already illumination-normalised page amplifies
    paper texture until adaptive thresholding returns the *halo around* text
    rather than the strokes: measured 23% ink coverage on ordinary lab reports,
    where strokes are 3-8%. Those halo bars carry no line structure, which made
    the angle sweep flip between 0 and -5.6 degrees on the same upright page
    depending only on how the page had been pre-rotated.
    """
    norm = normalize_illumination(gray)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    ink = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def drop_non_glyph_components(
    ink: np.ndarray,
    *,
    min_area: int = 8,
    max_area_frac: float = 0.02,
    max_dim_frac: float = 0.35,
) -> np.ndarray:
    """Keep glyph-scale blobs; drop speckles and page-scale structure.

    This is the single most important cleaning step. Page borders, scanner
    frames, the diagonal edge left by an earlier rotation, photos and large
    stamps are all long or large components whose straight edges dominate the
    projection score. Left in, they pull the estimate onto the page outline;
    with them removed the estimate follows the text. Removing them cut the
    worst-case error on sparse pages from 45 degrees to under 6.
    """
    h, w = ink.shape
    max_area = int(h * w * max_area_frac)
    max_extent = int(max(h, w) * max_dim_frac)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if int(stats[i, cv2.CC_STAT_WIDTH]) > max_extent:
            continue
        if int(stats[i, cv2.CC_STAT_HEIGHT]) > max_extent:
            continue
        keep[i] = True
    if not keep.any():
        return np.zeros_like(ink)
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def text_ink(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Analysis-resolution ink mask holding glyph-scale strokes only."""
    return drop_non_glyph_components(binarize(analysis_gray(image, max_dim)))


def ink_points(ink: np.ndarray, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centred float coordinates of ink pixels, subsampled to MAX_POINTS."""
    ys, xs = np.nonzero(ink)
    if len(xs) > MAX_POINTS:
        idx = np.random.default_rng(seed).choice(len(xs), MAX_POINTS, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    if len(xs):
        xs -= xs.mean()
        ys -= ys.mean()
    return xs, ys


# ------------------------------------------------------------------------ scoring
def profile_sharpness(proj: np.ndarray) -> float:
    """Concentration of a 1-D projection: sum of squared normalised bin mass."""
    lo = float(proj.min())
    hi = float(proj.max())
    n_bins = max(8, int((hi - lo) / BIN_SIZE) + 1)
    hist, _ = np.histogram(proj, bins=n_bins, range=(lo, hi))
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist.astype(np.float64) / total
    return float(np.dot(p, p))


def line_score(xs: np.ndarray, ys: np.ndarray, angle_cw_deg: float) -> float:
    """How well text forms horizontal lines after correcting ``angle_cw_deg``."""
    a = math.radians(angle_cw_deg)
    return profile_sharpness(-xs * math.sin(a) + ys * math.cos(a))


def axis_score(xs: np.ndarray, ys: np.ndarray, angle_cw_deg: float) -> float:
    """How well ink aligns to *either* axis after correcting ``angle_cw_deg``.

    Used for the residual search, which runs before the quadrant is known: a
    page turned a quarter turn is perfectly axis-aligned but its lines project
    onto x, not y. Taking the better of the two projections makes the score
    blind to which quadrant the page is in, so the search measures only the
    off-axis residual and leaves the quadrant to OSD.
    """
    a = math.radians(angle_cw_deg)
    sin_a, cos_a = math.sin(a), math.cos(a)
    return max(
        profile_sharpness(-xs * sin_a + ys * cos_a),
        profile_sharpness(xs * cos_a + ys * sin_a),
    )


def horizontal_axis_confidence(image: np.ndarray, max_dim: int) -> float:
    """Confidence that text lines run horizontally rather than vertically.

    This deliberately answers only the axis question. It cannot distinguish
    upright from upside-down, but fine deskew does not need that distinction:
    a line's skew is identical after a 180-degree turn.
    """
    ink = text_ink(image, max_dim)
    xs, ys = ink_points(ink)
    if len(xs) < MIN_POINTS:
        return 0.0
    horizontal = line_score(xs, ys, 0.0)
    vertical = line_score(xs, ys, 90.0)
    return float(horizontal / (horizontal + vertical + 1e-12))


def _refine(
    score_at: Callable[[float], float],
    best: float,
    radius: float,
    steps: tuple[float, ...],
) -> float:
    for step in steps:
        grid = np.arange(best - radius, best + radius + 1e-9, step)
        scores = [score_at(float(a)) for a in grid]
        best = float(grid[int(np.argmax(scores))])
        radius = step
    return best


def search(
    ink: np.ndarray,
    *,
    lo: float,
    hi: float,
    coarse_step: float,
    fine_steps: tuple[float, ...] = (0.1, 0.02),
    scorer: Callable[[np.ndarray, np.ndarray, float], float] = axis_score,
    seed: int = 0,
) -> AngleEstimate:
    """Sweep [lo, hi) coarsely, then refine around the winner."""
    xs, ys = ink_points(ink, seed=seed)
    if len(xs) < MIN_POINTS:
        return AngleEstimate(
            angle_cw_deg=None,
            confidence=0.0,
            peak_margin=0.0,
            runner_up_ratio=1.0,
            n_points=int(len(xs)),
            diagnostics={"reason": "insufficient_ink"},
        )

    def score_at(a: float) -> float:
        return scorer(xs, ys, a)

    coarse = np.arange(lo, hi, coarse_step)
    scores = np.array([score_at(float(a)) for a in coarse])
    peak_idx = int(scores.argmax())
    coarse_best = float(coarse[peak_idx])
    peak = float(scores[peak_idx])
    median = float(np.median(scores))
    peak_margin = (peak - median) / (peak + 1e-12) if peak > 0 else 0.0

    # Rival structure at a genuinely different angle?
    far = np.abs(((coarse - coarse_best + 90.0) % 180.0) - 90.0) > AMBIGUITY_SEPARATION_DEG
    runner_up_ratio = float(scores[far].max() / (peak + 1e-12)) if far.any() and peak > 0 else 0.0

    refined = _refine(score_at, coarse_best, coarse_step, fine_steps)
    return AngleEstimate(
        angle_cw_deg=refined,
        confidence=0.0,  # filled in by the calling stage, which knows its own gates
        peak_margin=float(peak_margin),
        runner_up_ratio=runner_up_ratio,
        n_points=int(len(xs)),
        diagnostics={
            "coarse_best": coarse_best,
            "peak_score": peak,
            "median_score": median,
            "search_range": [lo, hi],
        },
    )


def wrap_pm90(angle: float) -> float:
    """Map an angle into [-45, 45): the residual after removing whole quadrants."""
    return ((angle + 45.0) % 90.0) - 45.0


def wrap_pm180(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0
