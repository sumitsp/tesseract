"""Inference API — flags pages as KEEP / BLANK / JUNK (+ protocol audit tags)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.explainability.evidence import explain_page
from src.features.ocr_features import top_evidence
from src.models.classifiers import FlatClassifier
from src.models.decision import DecisionConfig, decide_from_proba
from src.preprocessing.protocol import analyze_protocol
from src.preprocessing.routing import route_empty_or_unreadable


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def page_type(self) -> str:
        return self.flag

    @property
    def keep_delete(self) -> str:
        return self.flag


class PageClassifierService:
    def __init__(
        self,
        model: FlatClassifier,
        *,
        model_version: str,
        decision: DecisionConfig | None = None,
        min_dictionary_words: int = 2,
    ) -> None:
        self.model = model
        self.model_version = model_version
        self.decision = decision or DecisionConfig()
        self.min_dictionary_words = min_dictionary_words

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
                review = False
            return InferenceResult(
                page_id=page_id,
                flag=flag,
                confidence=route.confidence,
                review_required=review,
                audit_tag=audit,
                top_evidence=route.reason or "empty_or_unreadable_ocr",
                model_version=self.model_version,
                decision_reason=f"pre_model_route:{route.reason}",
            )

        proba = self.model.predict_proba([ocr_text])[0]
        decision = decide_from_proba(
            proba,
            list(self.model.labels),
            ocr_text=ocr_text,
            config=self.decision,
        )
        evidence = explain_page(self.model, ocr_text, decision.flag)
        if not evidence:
            evidence = top_evidence(ocr_text)
        if decision.protocol_evidence:
            evidence = list(decision.protocol_evidence) + evidence
        return InferenceResult(
            page_id=page_id,
            flag=decision.flag,
            confidence=round(decision.confidence, 4),
            review_required=decision.review_required,
            audit_tag=decision.audit_tag or decision.flag,
            top_evidence="; ".join(evidence[:8]),
            model_version=self.model_version,
            decision_reason=decision.reason,
            p_keep=round(decision.p_keep, 4),
            p_blank=round(decision.p_blank, 4),
            p_junk=round(decision.p_junk, 4),
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
