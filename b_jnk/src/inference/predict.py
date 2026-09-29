"""Inference API — flags pages as KEEP / BLANK / JUNK (+ protocol audit tags)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.explainability.evidence import explain_page
from src.features.ocr_features import top_evidence
from src.models.classifiers import FlatClassifier
from src.models.decision import DecisionConfig, decide_from_proba
from src.preprocessing.page_subclass import (
    JUNK_SUBTYPES,
    assign_blank_subtype,
    assign_junk_subtype,
    junk_trigger_groups,
)
from src.preprocessing.protocol import analyze_protocol
from src.preprocessing.routing import route_empty_or_unreadable
from src.preprocessing.taxonomy import JUNK_CLASSES


@dataclass
class InferenceResult:
    page_id: str
    flag: str  # KEEP | BLANK | JUNK
    confidence: float
    review_required: bool
    audit_tag: str  # typology / retention subtype for chain of custody
    top_evidence: str
    model_version: str
    decision_reason: str
    p_keep: float | None = None
    p_blank: float | None = None
    p_junk: float | None = None
    subclass: str = ""  # BLANK_* / JUNK_* subtype; empty for KEEP
    subclass_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def page_type(self) -> str:
        return self.flag

    @property
    def keep_delete(self) -> str:
        return self.flag


def _subclass(flag: str, text: str, *, model_tag: str | None, structurally_empty: bool) -> tuple[str, str]:
    if flag == "JUNK":
        s = assign_junk_subtype(text, model_tag=model_tag, model_picks_mixed=model_tag in JUNK_SUBTYPES)
    elif flag == "BLANK":
        s = assign_blank_subtype(text, structurally_empty=structurally_empty)
    else:
        return "", ""
    return s.subclass, s.reason


def _best_junk_label(labels: list[str], proba) -> str | None:
    scored = [(float(p), lab) for lab, p in zip(labels, proba) if lab in JUNK_CLASSES and lab != "JUNK"]
    return max(scored)[1] if scored else None


class PageClassifierService:
    def __init__(
        self,
        model: FlatClassifier,
        *,
        model_version: str,
        decision: DecisionConfig | None = None,
        min_dictionary_words: int = 2,
        route_short_pages: bool = True,
    ) -> None:
        self.model = model
        self.model_version = model_version
        self.decision = decision or DecisionConfig()
        self.min_dictionary_words = min_dictionary_words
        self.route_short_pages = route_short_pages

    def predict_one(
        self,
        page_id: str,
        ocr_text: str,
        *,
        content_meta: dict[str, Any] | None = None,
    ) -> InferenceResult:
        route = route_empty_or_unreadable(
            ocr_text,
            min_dictionary_words=self.min_dictionary_words,
            content_meta=content_meta,
            route_short_text=self.route_short_pages,
        )
        if route.routed:
            hit = analyze_protocol(ocr_text)
            flag = route.flag
            audit = route.audit_tag or "UNREADABLE_OCR"
            review = route.review_required
            # Retention safeguards still win over blank routing
            if hit.retain_clinical_image:
                flag = "KEEP"
                audit = "KEEP_CLINICAL_IMAGE"
                review = False
            elif hit.retain_demographic:
                flag = "KEEP"
                audit = "KEEP_DEMOGRAPHIC"
                # Gibberish that happens to contain "dob" is not a real face sheet.
                review = route.reason == "gibberish_ocr"
            if route.reason == "gibberish_ocr" and flag == "JUNK":
                subclass, subclass_reason = "JUNK_OTHERS", "gibberish"
            else:
                subclass, subclass_reason = _subclass(
                    flag,
                    ocr_text,
                    model_tag=None,
                    structurally_empty=bool((content_meta or {}).get("is_structurally_empty")),
                )
            return InferenceResult(
                page_id=page_id,
                flag=flag,
                confidence=route.confidence,
                review_required=review,
                audit_tag=audit,
                top_evidence=route.reason or "empty_or_unreadable_ocr",
                model_version=self.model_version,
                decision_reason=f"pre_model_route:{route.reason}",
                subclass=subclass,
                subclass_reason=subclass_reason,
            )

        labels = list(self.model.labels)
        proba = self.model.predict_proba([ocr_text])[0]
        decision = decide_from_proba(
            proba,
            labels,
            ocr_text=ocr_text,
            config=self.decision,
        )
        evidence = explain_page(self.model, ocr_text, decision.flag)
        if not evidence:
            evidence = top_evidence(ocr_text)
        if decision.protocol_evidence:
            evidence = list(decision.protocol_evidence) + evidence

        review = decision.review_required
        reason = decision.reason
        if decision.flag == "KEEP" and not review:
            hits = junk_trigger_groups(ocr_text)
            if hits:
                review = True
                reason = f"{reason}+junk_trigger_review"
                evidence = [f"junk_trigger:{h}" for h in hits] + evidence

        model_tag = _best_junk_label(labels, proba) or decision.audit_tag
        subclass, subclass_reason = _subclass(
            decision.flag, ocr_text, model_tag=model_tag, structurally_empty=False
        )
        return InferenceResult(
            page_id=page_id,
            flag=decision.flag,
            confidence=round(decision.confidence, 4),
            review_required=review,
            audit_tag=decision.audit_tag or decision.flag,
            top_evidence="; ".join(evidence[:8]),
            model_version=self.model_version,
            decision_reason=reason,
            p_keep=round(decision.p_keep, 4),
            p_blank=round(decision.p_blank, 4),
            p_junk=round(decision.p_junk, 4),
            subclass=subclass,
            subclass_reason=subclass_reason,
        )

    def predict_batch(self, pages: list[dict[str, Any]]) -> list[InferenceResult]:
        return [
            self.predict_one(
                p["page_id"],
                p.get("ocr_text", ""),
                content_meta=p.get("content_meta"),
            )
            for p in pages
        ]
