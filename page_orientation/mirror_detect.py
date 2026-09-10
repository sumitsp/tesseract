"""Horizontal mirror detection via structural comparison — no OCR."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .components import extract_text_components, group_text_lines


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
