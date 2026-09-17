"""Stage 5 — horizontal mirroring, after rotation and before fine tilt.

Runs on a page that is already upright, because every cue used here is defined
on upright left-to-right geometry: a quarter-turned page has no meaningful "left
margin", and rotation and horizontal flip do not commute for 90/270.

Two roles, not one
-------------------
The primary mirror verdict now comes from stage 4B's OSD-based check (see
``osd_direction.detect_quadrant_and_mirror``), which resolves quadrant and
mirror together and is what actually catches most mirrored pages. This
module has two remaining jobs, both driven from ``pipeline.py``:

  * When OSD is confident the page is mirrored, ``detect_mirror`` here is the
    required second opinion before anything is actually flipped — flipping a
    good page is severely destructive, so that one irreversible action needs
    two independent signals to agree, not one.
  * When OSD could not resolve a quadrant at all (and so has no mirror
    opinion either), this module is the sole fallback detector.

Why this stage is built to say NO
---------------------------------
Mirrored pages are rare and flipping a good page is severely destructive, so the
cost of the two errors is nowhere near symmetric. A purely structural detector of
this kind was measured in the sibling ``model-repo`` project reporting mirror on 3
of 12 pages that were not mirrored — a 25% false-positive rate — and that project
ended up recording its verdict but never applying it.

So structural evidence here is only a *gate*, never the decision. Ordinary pages
fail the gate immediately and cost nothing beyond two cheap projections. Only a
page that looks genuinely reversed pays for a Tesseract confirmation pass, and
only agreement between the two lines of evidence flips anything. Anything short
of that is reported UNKNOWN and left untouched.

Structural cue
--------------
Left-to-right text has tightly clustered line *starts* on the left and a ragged
right edge. Mirroring swaps that. Measured as the spread of line left-edges
versus right-edges, which is a text-line property rather than a pixel-density
one — a left/right ink balance is not usable on its own, since page layout, not
reading direction, decides which half of a form holds more ink.

The 5-degree rule
-----------------
A horizontal flip maps a text tilt of alpha to exactly -alpha; nothing else about
the geometry may change. So measure the deskew angle on the page as it stands
(``alpha_original``) and on the flipped candidate (``alpha_flipped``). For a
genuine mirror::

    alpha_flipped == -alpha_original

and the residual that a flip cannot account for is::

    extra_rotation = alpha_flipped - (-alpha_original)
                   = alpha_flipped + alpha_original

When ``|extra_rotation| > mirror_max_extra_rotation_deg`` (5 degrees) the mirror
hypothesis needs a rotation on top of the flip to explain what is on the page,
which means the evidence is not a clean mirror. The page is then NOT flipped,
mirror is reported UNKNOWN, and the row is warned. This is the documented
interpretation of the requirement; it is a rejection rule, never a licence to
rotate by up to 5 degrees here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from image_preprocessing.config import PipelineConfig
from image_preprocessing.orientation.angle_search import (
    line_score,
    search,
    text_ink,
)
from image_preprocessing.orientation.text_geometry import (
    extract_text_components,
    group_text_lines,
)
from image_preprocessing.utils.image_utils import flip_horizontal

LOGGER = logging.getLogger(__name__)


@dataclass
class MirrorResult:
    mirror: str  # YES | NO | UNKNOWN
    corrected: bool
    confidence: float
    warning: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _edge_alignment(ink: np.ndarray) -> tuple[float, int]:
    """How much tighter line starts cluster than line ends.

    Positive means left-edges are the tidier margin, i.e. normal LTR. Returns
    (score, n_lines).
    """
    comps = extract_text_components(ink)
    lines = group_text_lines(comps, min_comps=4)
    if len(lines) < 3:
        return 0.0, len(lines)
    lefts = np.array([ln.x_min for ln in lines], dtype=np.float64)
    rights = np.array([ln.x_max for ln in lines], dtype=np.float64)

    def spread(v: np.ndarray) -> float:
        return float(np.median(np.abs(v - np.median(v)))) + 1.0

    left_spread = spread(lefts)
    right_spread = spread(rights)
    # >0 when the left margin is tighter than the right.
    score = (right_spread - left_spread) / (right_spread + left_spread)
    return float(score), len(lines)


def _ocr_strength(image: np.ndarray) -> tuple[int, float]:
    """(confident word count, mean confidence). (0, 0.0) if unavailable."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return 0, 0.0
    try:
        # psm 6 = uniform block. psm 3's layout pass silently rotates vertical
        # text, which muddies a left/right comparison.
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6")
    except Exception as exc:
        LOGGER.debug("Mirror OCR check unavailable: %s", exc)
        return 0, 0.0
    confs = []
    for conf, text in zip(data.get("conf", []), data.get("text", [])):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        text = (text or "").strip()
        if conf >= 60 and len(text) >= 2 and any(ch.isalnum() for ch in text):
            confs.append(conf)
    return len(confs), (float(np.mean(confs)) if confs else 0.0)


def _deskew_angle(image: np.ndarray, config: PipelineConfig) -> float | None:
    ink = text_ink(image, config.analysis_max_dimension)
    est = search(
        ink,
        lo=-config.max_skew_angle,
        hi=config.max_skew_angle + 1e-9,
        coarse_step=config.skew_coarse_step_deg,
        fine_steps=(0.1,),
        scorer=line_score,
    )
    return est.angle_cw_deg


def detect_mirror(upright_image: np.ndarray, config: PipelineConfig) -> MirrorResult:
    """Decide whether ``upright_image`` is horizontally mirrored."""
    ink = text_ink(upright_image, config.analysis_max_dimension)
    normal_score, n_lines = _edge_alignment(ink)
    flipped_score, _ = _edge_alignment(flip_horizontal(ink))

    diagnostics: dict = {
        "edge_alignment_normal": round(normal_score, 4),
        "edge_alignment_flipped": round(flipped_score, 4),
        "line_count": n_lines,
    }

    if n_lines < config.mirror_min_line_count:
        return MirrorResult(
            mirror="UNKNOWN",
            corrected=False,
            confidence=0.0,
            warning="Mirror not determined: too few text lines",
            diagnostics=diagnostics,
        )

    # Gate. The flipped page must look clearly more LTR-like than the page as it
    # stands before anything more expensive happens.
    structural_margin = flipped_score - normal_score
    diagnostics["structural_margin"] = round(structural_margin, 4)
    if structural_margin < config.mirror_score_margin:
        return MirrorResult(
            mirror="NO",
            corrected=False,
            confidence=round(float(np.clip(0.55 + structural_margin, 0.0, 1.0)), 4),
            diagnostics=diagnostics,
        )

    # The 5-degree rule, before spending an OCR pass.
    alpha_original = _deskew_angle(upright_image, config)
    flipped_image = flip_horizontal(upright_image)
    alpha_flipped = _deskew_angle(flipped_image, config)
    if alpha_original is None or alpha_flipped is None:
        return MirrorResult(
            mirror="UNKNOWN",
            corrected=False,
            confidence=0.0,
            warning="Mirror not determined: insufficient text geometry",
            diagnostics=diagnostics,
        )
    extra_rotation = float(alpha_flipped + alpha_original)
    diagnostics["alpha_original"] = round(float(alpha_original), 3)
    diagnostics["alpha_flipped"] = round(float(alpha_flipped), 3)
    diagnostics["extra_rotation_deg"] = round(extra_rotation, 3)
    if abs(extra_rotation) > config.mirror_max_extra_rotation_deg:
        return MirrorResult(
            mirror="UNKNOWN",
            corrected=False,
            confidence=0.0,
            warning="Mirror detection exceeded 5-degree correction threshold",
            diagnostics=diagnostics,
        )

    # Confirmation. Real text reads far better the right way round; a mirrored
    # page yields garbage regardless of how tidy its margins look.
    normal_words, normal_conf = _ocr_strength(upright_image)
    flipped_words, flipped_conf = _ocr_strength(flipped_image)
    diagnostics["ocr_normal"] = [normal_words, round(normal_conf, 1)]
    diagnostics["ocr_flipped"] = [flipped_words, round(flipped_conf, 1)]

    if normal_words + flipped_words == 0:
        return MirrorResult(
            mirror="UNKNOWN",
            corrected=False,
            confidence=0.0,
            warning="Mirror not confirmed: no OCR evidence available",
            diagnostics=diagnostics,
        )

    normal_strength = normal_words * max(normal_conf, 1.0)
    flipped_strength = flipped_words * max(flipped_conf, 1.0)
    total = normal_strength + flipped_strength
    ocr_margin = (flipped_strength - normal_strength) / (total + 1e-9)
    diagnostics["ocr_margin"] = round(ocr_margin, 4)

    if ocr_margin < config.mirror_ocr_margin:
        return MirrorResult(
            mirror="NO",
            corrected=False,
            confidence=round(float(np.clip(0.5 + abs(ocr_margin), 0.0, 1.0)), 4),
            diagnostics=diagnostics,
        )

    confidence = float(np.clip(0.45 + 0.5 * ocr_margin + 0.5 * structural_margin, 0.0, 1.0))
    if confidence < config.mirror_confidence_threshold:
        return MirrorResult(
            mirror="UNKNOWN",
            corrected=False,
            confidence=round(confidence, 4),
            warning="Mirror evidence too weak to act on; page not flipped",
            diagnostics=diagnostics,
        )
    return MirrorResult(
        mirror="YES",
        corrected=True,
        confidence=round(confidence, 4),
        diagnostics=diagnostics,
    )
