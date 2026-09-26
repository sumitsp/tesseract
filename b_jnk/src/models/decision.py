"""Map model probabilities → KEEP / BLANK / JUNK flags.

Applies Blank & Junk elimination protocol:
  - Model scores are primary
  - Typology patterns feed audit tags
  - Mandatory clinical-image / demographic retention hard-blocks JUNK/BLANK

Nothing is deleted from the packet by this module — flags + audit tags only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing.protocol import ProtocolHit, analyze_protocol, must_retain
from src.preprocessing.taxonomy import Flag, to_flag


@dataclass
class DecisionConfig:
    min_flag_confidence: float = 0.70
    junk_blank_min_confidence: float = 0.85
    keep_veto_prob: float = 0.10
    # Protocol: never allow JUNK/BLANK when retention signals fire
    enforce_retention_safeguards: bool = True


@dataclass
class PageDecision:
    flag: Flag
    confidence: float
    review_required: bool
    p_keep: float
    p_blank: float
    p_junk: float
    reason: str
    audit_tag: str | None = None
    protocol_evidence: tuple[str, ...] = ()

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
        buckets[flag] += max(0.0, float(p))
    total = sum(buckets.values())
    if total > 1.0 + 1e-6:
        for k in buckets:
            buckets[k] /= total
    for k in buckets:
        buckets[k] = max(0.0, min(1.0, buckets[k]))
    return buckets


def _default_audit(flag: Flag, hit: ProtocolHit) -> str | None:
    if hit.audit_tag:
        return hit.audit_tag
    return {"KEEP": "KEEP", "BLANK": "BLANK", "JUNK": "JUNK"}[flag]


def decide_from_proba(
    proba: np.ndarray,
    labels: list[str],
    *,
    ocr_text: str = "",
    config: DecisionConfig | None = None,
) -> PageDecision:
    config = config or DecisionConfig()
    buckets = _bucket_probs(labels, np.asarray(proba, dtype=float).ravel())
    p_keep, p_blank, p_junk = buckets["KEEP"], buckets["BLANK"], buckets["JUNK"]
    hit = analyze_protocol(ocr_text)

    scores = {"KEEP": p_keep, "BLANK": p_blank, "JUNK": p_junk}
    best: Flag = max(scores, key=scores.get)  # type: ignore[assignment]
    confidence = float(scores[best])
    reason = "argmax"
    review = False
    flag: Flag = best

    # --- Protocol mandatory retention (never junk/blank) ---
    if config.enforce_retention_safeguards and must_retain(hit) and flag in {"JUNK", "BLANK"}:
        flag = "KEEP"
        confidence = float(max(p_keep, confidence, 0.99))
        review = False  # protocol is definitive KEEP
        if hit.retain_clinical_image:
            reason = "protocol_retain_clinical_image"
        else:
            reason = "protocol_retain_demographic"

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

    if confidence < config.min_flag_confidence and reason == "argmax":
        flag = "KEEP"
        review = True
        reason = "below_review_threshold"

    # If model said KEEP but typology screams system-blank / fax, still allow
    # model KEEP (safe). Audit tag carries typology for QC.

    confidence = max(0.0, min(1.0, float(confidence)))
    audit = _default_audit(flag, hit)
    # If we forced KEEP via retention, stamp the retention audit tag
    if reason.startswith("protocol_retain"):
        audit = (
            "KEEP_CLINICAL_IMAGE"
            if hit.retain_clinical_image
            else "KEEP_DEMOGRAPHIC"
        )

    return PageDecision(
        flag=flag,
        confidence=confidence,
        review_required=review,
        p_keep=float(p_keep),
        p_blank=float(p_blank),
        p_junk=float(p_junk),
        reason=reason,
        audit_tag=audit,
        protocol_evidence=tuple(hit.evidence),
    )
