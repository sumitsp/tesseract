"""Inference API — flags pages as KEEP / BLANK / JUNK (CSV-friendly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.explainability.evidence import explain_page
from src.features.ocr_features import top_evidence
from src.models.classifiers import FlatClassifier
from src.models.decision import DecisionConfig, decide_from_proba
from src.preprocessing.routing import route_empty_or_unreadable


@dataclass
class InferenceResult:
    page_id: str
    flag: str  # KEEP | BLANK | JUNK
    confidence: float
    review_required: bool
    top_evidence: str
    model_version: str
    decision_reason: str
    p_keep: float | None = None
    p_blank: float | None = None
    p_junk: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Legacy aliases for older scripts
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

    def predict_one(self, page_id: str, ocr_text: str) -> InferenceResult:
        route = route_empty_or_unreadable(
            ocr_text, min_dictionary_words=self.min_dictionary_words
        )
        if route.routed:
            # Empty/unreadable OCR: flag KEEP + review (never auto-BLANK)
            return InferenceResult(
                page_id=page_id,
                flag="KEEP",
                confidence=route.confidence,
                review_required=True,
                top_evidence=route.reason or "empty_or_unreadable_ocr",
                model_version=self.model_version,
                decision_reason=f"pre_model_route:{route.reason}",
            )

        proba = self.model.predict_proba([ocr_text])[0]
        decision = decide_from_proba(proba, list(self.model.labels), config=self.decision)
        evidence = explain_page(self.model, ocr_text, decision.flag)
        if not evidence:
            evidence = top_evidence(ocr_text)
        return InferenceResult(
            page_id=page_id,
            flag=decision.flag,
            confidence=round(decision.confidence, 4),
            review_required=decision.review_required,
            top_evidence="; ".join(evidence),
            model_version=self.model_version,
            decision_reason=decision.reason,
            p_keep=round(decision.p_keep, 4),
            p_blank=round(decision.p_blank, 4),
            p_junk=round(decision.p_junk, 4),
        )

    def predict_batch(self, pages: list[dict[str, str]]) -> list[InferenceResult]:
        return [self.predict_one(p["page_id"], p.get("ocr_text", "")) for p in pages]
