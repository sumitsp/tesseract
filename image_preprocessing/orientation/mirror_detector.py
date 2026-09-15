"""Stage 5 — mirror detection AFTER rotation has been corrected.

A page must already be approximately upright. This detector never jointly
optimises rotation + mirror.

Decision
--------
Compare classical LTR text-line evidence on the upright page vs the same
page after a horizontal flip. Optionally confirm with a cheap Tesseract
word-confidence vote on a downscaled copy.

Conservatism
------------
Ambiguous evidence → mirror=UNKNOWN, do not flip.

Special 5-degree business rule
------------------------------
A valid mirror correction is a *pure horizontal flip*. Residual skew from
the previous stage is not a mirror signal: a horizontal flip *negates*
small line angles (a +7° clockwise lean becomes -7°).

Define:

    orig_α  = CCW angle in [-15°, +15°] that best aligns the unflipped page
    flip_α  = CCW angle in [-15°, +15°] that best aligns the flipped page
    expected_flip_α = -orig_α     # skew sign reverses under a horizontal flip
    extra_rotation  = flip_α - expected_flip_α

``extra_rotation`` is the additional rotation the mirrored candidate would
need *beyond* a pure flip (and the expected skew-sign change). If

    abs(extra_rotation) > MIRROR_MAX_EXTRA_ROTATION_DEG (default 5°)

the "mirror" explanation is mixing in residual rotation and is unreliable:

    mirror = UNKNOWN
    mirror_corrected = NO
    warning = "Mirror detection exceeded 5-degree correction threshold"

``extra_rotation`` is a reliability gate only. It is never applied here.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.text_geometry import (
    extract_ink,
    extract_text_components,
    group_text_lines,
    projection_score_at_angle,
)
from image_preprocessing.utils.image_utils import (
    INTER_DOWNSAMPLE,
    bgr_to_pil,
    ensure_bgr,
    flip_horizontal,
    resize_max_dimension,
    to_gray,
)


@dataclass
class MirrorResult:
    mirror: str  # YES | NO | UNKNOWN
    confidence: float | None
    corrected: bool
    extra_rotation_deg: float | None
    warning: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _line_ltr_features(ink: np.ndarray) -> dict[str, float]:
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
            "total": 0.0,
        }
    lefts = np.array([ln.x_min for ln in lines], dtype=np.float64)
    rights = np.array([ln.x_max for ln in lines], dtype=np.float64)
    spans = np.array([ln.x_max - ln.x_min for ln in lines], dtype=np.float64)
    med_l = float(np.median(lefts))
    mad_l = float(np.median(np.abs(lefts - med_l))) + 1.0
    med_r = float(np.median(rights))
    mad_r = float(np.median(np.abs(rights - med_r))) + 1.0
    left_align = float(np.clip(w / (mad_l * 8.0), 0.0, 1.0))
    right_align = float(np.clip(w / (mad_r * 8.0), 0.0, 1.0))
    starts_leftness = float(np.clip(1.0 - (med_l / (0.55 * w + 1e-9)), 0.0, 1.0))
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

    # Prefer a tight LEFT margin over a tight RIGHT margin (LTR pages).
    margin_preference = float(np.clip(0.5 + 0.5 * (left_align - right_align), 0.0, 1.0))
    total = (
        0.30 * left_align
        + 0.25 * starts_leftness
        + 0.20 * margin_preference
        + 0.15 * span_score
        + 0.10 * spacing
    )
    return {
        "n_lines": float(len(lines)),
        "left_align": left_align,
        "right_align": right_align,
        "starts_leftness": starts_leftness,
        "span_score": span_score,
        "spacing": spacing,
        "margin_preference": margin_preference,
        "total": float(total),
    }


def _best_extra_rotation(ink: np.ndarray, search: float = 15.0, step: float = 1.0) -> tuple[float, float]:
    """Angle in [-search, search] that maximises horizontal alignment of ``ink``."""
    best_a, best_s = 0.0, projection_score_at_angle(ink, 0.0)
    a = -search
    while a <= search + 1e-9:
        s = projection_score_at_angle(ink, float(a))
        if s > best_s:
            best_s, best_a = s, float(a)
        a += step
    return float(best_a), float(best_s)


def _tesseract_confidence_vote(image: np.ndarray, flipped: np.ndarray) -> tuple[float | None, float | None]:
    """Mean word confidence original vs flipped. None if Tesseract cannot run."""
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return None, None

    def _mean_conf(img: np.ndarray) -> float | None:
        work, _ = resize_max_dimension(ensure_bgr(img), 900, interpolation=INTER_DOWNSAMPLE)
        pil = bgr_to_pil(work)
        try:
            data = pytesseract.image_to_data(pil, output_type=Output.DICT, config="--psm 6")
        except Exception:
            return None
        confs = [int(c) for c in data.get("conf", []) if str(c) not in {"-1", ""}]
        confs = [c for c in confs if c >= 0]
        if len(confs) < 5:
            return None
        return float(np.mean(confs))

    return _mean_conf(image), _mean_conf(flipped)


def detect_mirror(
    upright_image: np.ndarray,
    config: PipelineConfig,
) -> MirrorResult:
    gray = to_gray(upright_image)
    analysis, _ = resize_max_dimension(gray, config.analysis_max_dimension)
    _clahe, _ink, ink = extract_ink(analysis)
    flipped_ink = flip_horizontal(ink)

    orig_alpha, orig_score = _best_extra_rotation(ink, search=15.0, step=1.0)
    flip_alpha, flip_score = _best_extra_rotation(flipped_ink, search=15.0, step=1.0)
    expected_flip_alpha = -orig_alpha
    extra_rot = float(flip_alpha - expected_flip_alpha)
    diag: dict[str, Any] = {
        "orig_align_angle_ccw": orig_alpha,
        "orig_align_score": orig_score,
        "flip_align_angle_ccw": flip_alpha,
        "flip_align_score": flip_score,
        "expected_flip_align_ccw": expected_flip_alpha,
        "extra_rotation_deg": extra_rot,
        "threshold_deg": config.mirror_max_extra_rotation_deg,
        "rule": (
            "extra_rotation = flip_α - (-orig_α). A pure horizontal flip "
            "negates residual skew, so extra_rotation should be ~0. If "
            "abs(extra_rotation) > MIRROR_MAX_EXTRA_ROTATION_DEG, the "
            "mirrored candidate only looks plausible after a non-trivial "
            "extra rotation and is rejected."
        ),
    }

    if abs(extra_rot) > config.mirror_max_extra_rotation_deg:
        return MirrorResult(
            mirror="UNKNOWN",
            confidence=None,
            corrected=False,
            extra_rotation_deg=round(extra_rot, 2),
            warning="Mirror detection exceeded 5-degree correction threshold",
            diagnostics=diag,
        )

    normal = _line_ltr_features(ink)
    flipped = _line_ltr_features(flipped_ink)
    diag["normal"] = normal
    diag["flipped"] = flipped

    n_lines = min(normal["n_lines"], flipped["n_lines"])
    if n_lines < config.mirror_min_line_count:
        return MirrorResult(
            mirror="UNKNOWN",
            confidence=round(0.2, 4),
            corrected=False,
            extra_rotation_deg=round(extra_rot, 2),
            warning="Mirror evidence insufficient",
            diagnostics=diag,
        )

    n_score, f_score = normal["total"], flipped["total"]
    best = max(n_score, f_score)
    sep = (best - min(n_score, f_score)) / (best + 1e-9)
    diag["separation"] = sep

    votes = 0
    if f_score > n_score * (1.0 + config.mirror_score_margin):
        votes += 1
    if flipped["starts_leftness"] > normal["starts_leftness"] + 0.08:
        votes += 1
    if flipped["left_align"] > normal["left_align"] * 1.08:
        votes += 1
    if flipped.get("margin_preference", 0) > normal.get("margin_preference", 0) + 0.08:
        votes += 1
    diag["structural_votes"] = votes

    if votes == 2:
        tess_n, tess_f = _tesseract_confidence_vote(analysis, flip_horizontal(analysis))
        diag["tesseract_conf_original"] = tess_n
        diag["tesseract_conf_flipped"] = tess_f
        if tess_n is not None and tess_f is not None:
            if tess_f > tess_n + 8:
                votes += 1
                diag["tesseract_vote"] = "flipped"
            elif tess_n > tess_f + 8:
                votes -= 1
                diag["tesseract_vote"] = "original"
            else:
                diag["tesseract_vote"] = "tie"
    else:
        diag["tesseract_vote"] = "skipped_not_borderline"

    want_flip = votes >= 3 and f_score > n_score and sep >= 0.07
    conf = float(np.clip(0.20 + 2.4 * sep + 0.10 * max(votes, 0), 0.0, 1.0))

    ambiguous = (
        sep < 0.07
        or best < 0.15
        or votes < 3
        or conf < config.mirror_confidence_threshold
        or not want_flip
    )
    if ambiguous and not want_flip:
        # Clear "not mirrored" still needs enough evidence to say NO vs UNKNOWN.
        if sep >= 0.07 and n_score > f_score and votes <= 1 and conf >= 0.35:
            return MirrorResult(
                mirror="NO",
                confidence=round(min(max(conf, 0.55), 1.0), 4),
                corrected=False,
                extra_rotation_deg=round(extra_rot, 2),
                warning=None,
                diagnostics=diag,
            )
        return MirrorResult(
            mirror="UNKNOWN",
            confidence=round(min(conf, 0.49), 4),
            corrected=False,
            extra_rotation_deg=round(extra_rot, 2),
            warning="Mirror evidence ambiguous",
            diagnostics=diag,
        )

    if want_flip and conf >= config.mirror_confidence_threshold:
        return MirrorResult(
            mirror="YES",
            confidence=round(conf, 4),
            corrected=True,
            extra_rotation_deg=round(extra_rot, 2),
            warning=None,
            diagnostics=diag,
        )

    return MirrorResult(
        mirror="UNKNOWN",
        confidence=round(min(conf, 0.49), 4),
        corrected=False,
        extra_rotation_deg=round(extra_rot, 2),
        warning="Mirror evidence ambiguous",
        diagnostics=diag,
    )
