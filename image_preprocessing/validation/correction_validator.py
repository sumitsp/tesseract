"""Accept a candidate correction only when it improves geometric metrics.

Every transform is: candidate → validate → accept or reject.
A wrong correction is worse than leaving the page unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.text_geometry import (
    extract_ink,
    extract_text_components,
    group_text_lines,
    horizontal_alignment_score,
)
from image_preprocessing.utils.image_utils import resize_max_dimension, to_gray


@dataclass
class ValidationResult:
    accepted: bool
    before_score: float
    after_score: float
    reason: str


def alignment_score(image: np.ndarray, max_dim: int) -> float:
    gray = to_gray(image)
    analysis, _ = resize_max_dimension(gray, max_dim)
    _clahe, _ink, ink = extract_ink(analysis)
    comps = extract_text_components(ink)
    if len(comps) >= 8:
        mask = np.zeros_like(ink)
        for c in comps:
            mask[c.y : c.y + c.h, c.x : c.x + c.w] = ink[c.y : c.y + c.h, c.x : c.x + c.w]
        work = mask
    else:
        work = ink
    proj = horizontal_alignment_score(work)
    lines = group_text_lines(comps, min_comps=3)
    if len(lines) >= 2:
        angles = np.array([abs(ln.angle_deg) for ln in lines], dtype=np.float64)
        flat = float(np.mean(angles < 4.0))
        n_factor = min(len(lines) / 15.0, 1.0)
    else:
        flat = 0.0
        n_factor = 0.0
    return float(proj + 0.8 * flat + 0.4 * n_factor)


def residual_skew_abs(image: np.ndarray, max_dim: int) -> float | None:
    gray = to_gray(image)
    analysis, _ = resize_max_dimension(gray, max_dim)
    _clahe, _ink, ink = extract_ink(analysis)
    lines = group_text_lines(extract_text_components(ink), min_comps=3)
    if len(lines) < 3:
        return None
    angles = np.array([ln.angle_deg for ln in lines], dtype=np.float64)
    angles = angles[np.abs(angles) <= 20]
    if len(angles) < 3:
        return None
    return float(abs(np.median(angles)))


def validate_candidate(
    before_image: np.ndarray,
    after_image: np.ndarray,
    config: PipelineConfig,
    *,
    min_improvement_ratio: float | None = None,
) -> ValidationResult:
    """Reject the candidate if alignment got worse (or did not improve enough)."""
    before = alignment_score(before_image, config.analysis_max_dimension)
    after = alignment_score(after_image, config.analysis_max_dimension)
    ratio = config.validation_min_improvement_ratio if min_improvement_ratio is None else min_improvement_ratio
    # Allow tiny numeric jitter but never accept a clearly worse page.
    if after + 1e-6 < before * (1.0 - ratio):
        return ValidationResult(
            accepted=False,
            before_score=before,
            after_score=after,
            reason="alignment_worse_than_before",
        )
    return ValidationResult(
        accepted=True,
        before_score=before,
        after_score=after,
        reason="alignment_improved_or_equivalent",
    )


def content_not_cropped(before_shape: tuple[int, ...], after_shape: tuple[int, ...]) -> bool:
    """Rotate-bound must not shrink the canvas below the original min side unreasonably.

    We cannot compare pixel counts directly after rotation (canvas grows). This
    check only flags accidental crops that produce a smaller image than the
    source without a 90° axis swap explanation.
    """
    bh, bw = before_shape[:2]
    ah, aw = after_shape[:2]
    before_area = bh * bw
    after_area = ah * aw
    return after_area + 1 >= int(0.92 * before_area)
