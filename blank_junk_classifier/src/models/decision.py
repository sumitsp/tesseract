"""Map model probabilities → KEEP / BLANK / JUNK flags.

Nothing is deleted. Low-confidence cases are flagged KEEP with
review_required=True (safe default: never call clinical content JUNK/BLANK
when unsure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing.taxonomy import Flag, to_flag


@dataclass
class DecisionConfig:
    min_flag_confidence: float = 0.70
    junk_blank_min_confidence: float = 0.85
    keep_veto_prob: float = 0.10


@dataclass
class PageDecision:
    flag: Flag
    confidence: float
    review_required: bool
    p_keep: float
    p_blank: float
    p_junk: float
    reason: str

    @property
    def page_type(self) -> str:
        return self.flag

    @property
    def keep_delete(self) -> str:
        return self.flag


def _bucket_probs(labels: list[str], proba: np.ndarray) -> dict[str, float]:
    buckets = {"KEEP": 0.0, "BLANK": 0.0, "JUNK": 0.0}
    for lab, p in zip(labels, proba):
        try:
            flag = to_flag(lab)
        except KeyError:
            continue
        if flag is None:
            continue
        buckets[flag] += float(p)
    return buckets


def decide_from_proba(
    proba: np.ndarray,
    labels: list[str],
    *,
    config: DecisionConfig | None = None,
) -> PageDecision:
    config = config or DecisionConfig()
    buckets = _bucket_probs(labels, proba)
    p_keep, p_blank, p_junk = buckets["KEEP"], buckets["BLANK"], buckets["JUNK"]

    scores = {"KEEP": p_keep, "BLANK": p_blank, "JUNK": p_junk}
    best: Flag = max(scores, key=scores.get)  # type: ignore[assignment]
    confidence = float(scores[best])
    reason = "argmax"
    review = False
    flag: Flag = best

    if flag in {"JUNK", "BLANK"} and p_keep > config.keep_veto_prob:
        flag = "KEEP"
        confidence = float(p_keep)
        review = True
        reason = "keep_veto"

    if flag in {"JUNK", "BLANK"} and confidence < config.junk_blank_min_confidence:
        flag = "KEEP"
        confidence = float(max(p_keep, confidence))
        review = True
        reason = "junk_blank_low_confidence"

    if confidence < config.min_flag_confidence:
        flag = "KEEP"
        review = True
        reason = "below_review_threshold"

    return PageDecision(
        flag=flag,
        confidence=float(confidence),
        review_required=review,
        p_keep=float(p_keep),
        p_blank=float(p_blank),
        p_junk=float(p_junk),
        reason=reason,
    )
