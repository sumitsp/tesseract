"""Stage 7 — accept a candidate correction only if it is measurably not worse.

Every geometric change in the pipeline goes through here:

    candidate -> validate -> better or equal? -> accept, else keep previous

A detector returning an angle is not evidence that applying it helped. This
module re-measures the page after the transform and compares.

The metric is the same horizontal text-line concentration the detectors
maximise (``angle_search.line_score`` at 0 degrees, i.e. "are text lines
horizontal *now*"), so a rotation that genuinely squares the page up scores
higher and one that tilts it away scores lower. Sharing the metric with the
detectors is deliberate: a validator measuring something unrelated cannot tell
whether the thing the detector tried to improve actually improved.

Tolerance, not strict improvement
---------------------------------
A correct 180-degree turn leaves line geometry identical, and resampling costs a
little sharpness, so demanding strict improvement would reject correct
corrections. The gate is therefore "not meaningfully worse", with the small
tolerance in ``validation_regression_tolerance``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.angle_search import (
    ink_points,
    line_score,
    text_ink,
)


@dataclass
class ValidationResult:
    accepted: bool
    before_score: float
    after_score: float
    reason: str


def alignment_score(image: np.ndarray, config: PipelineConfig) -> float:
    """How horizontal the page's text lines are, as it currently stands."""
    ink = text_ink(image, config.analysis_max_dimension)
    xs, ys = ink_points(ink)
    if len(xs) < 300:
        return 0.0
    return line_score(xs, ys, 0.0)


def content_not_cropped(before_shape, after_shape, *, tolerance: float = 0.005) -> bool:
    """A correction may grow the canvas (rotate-bound) but must never shrink it.

    Compares against the rotated bounding box rather than raw area, since a
    quarter turn legitimately swaps width and height.
    """
    bh, bw = before_shape[:2]
    ah, aw = after_shape[:2]
    before_diag = float(bw * bw + bh * bh)
    after_diag = float(aw * aw + ah * ah)
    if after_diag + 1e-9 >= before_diag * (1.0 - tolerance):
        return True
    # Allow the exact quarter-turn swap.
    return (abs(aw - bh) <= 2 and abs(ah - bw) <= 2)


def validate_candidate(
    before: np.ndarray,
    after: np.ndarray,
    config: PipelineConfig,
    *,
    regression_tolerance: float | None = None,
) -> ValidationResult:
    """Accept ``after`` unless it is measurably worse aligned than ``before``."""
    tolerance = (
        config.validation_regression_tolerance
        if regression_tolerance is None
        else regression_tolerance
    )
    if not content_not_cropped(before.shape, after.shape):
        return ValidationResult(False, 0.0, 0.0, "content_cropped")

    before_score = alignment_score(before, config)
    after_score = alignment_score(after, config)
    if before_score <= 0.0 and after_score <= 0.0:
        # Nothing measurable either way: no grounds to reject, and no grounds to
        # claim an improvement.
        return ValidationResult(True, before_score, after_score, "no_measurable_ink")

    if after_score + 1e-12 >= before_score * (1.0 - tolerance):
        return ValidationResult(True, before_score, after_score, "not_worse")
    return ValidationResult(
        False,
        before_score,
        after_score,
        f"alignment_regressed({after_score:.5f}<{before_score:.5f})",
    )
