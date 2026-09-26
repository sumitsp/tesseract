"""Pre-model routing for empty / unreadable OCR."""

from __future__ import annotations

from dataclasses import dataclass

from src.features.ocr_features import count_dictionary_words


@dataclass
class RouteDecision:
    routed: bool
    reason: str | None
    confidence: float


def route_empty_or_unreadable(
    ocr_text: str,
    *,
    min_dictionary_words: int = 2,
) -> RouteDecision:
    """Empty OCR cannot safely be flagged BLANK (could be clinical image).

    Caller should emit flag=KEEP with review_required=True.
    """
    text = (ocr_text or "").strip()
    if not text:
        return RouteDecision(routed=True, reason="empty_ocr", confidence=1.0)
    n_dict = count_dictionary_words(text)
    if n_dict < min_dictionary_words and len(text) < 40:
        return RouteDecision(
            routed=True,
            reason="insufficient_dictionary_words",
            confidence=0.9,
        )
    return RouteDecision(routed=False, reason=None, confidence=0.0)
