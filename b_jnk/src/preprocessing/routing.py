"""Pre-model routing for empty / image-only / unreadable pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.ocr_features import count_dictionary_words


@dataclass
class RouteDecision:
    routed: bool
    reason: str | None
    confidence: float
    audit_tag: str | None = None


def route_empty_or_unreadable(
    ocr_text: str,
    *,
    min_dictionary_words: int = 2,
    content_meta: dict[str, Any] | None = None,
) -> RouteDecision:
    """Route pages that should not go through the text classifier alone.

    Absolute blank is inferred from Docling structure when available
    (empty texts/pictures/tables/body) — not from hardcoded OCR phrases.

    Empty text / structural blank → KEEP + review (could be clinical image OCR miss).
    Pictures present but no texts → KEEP + review (possible clinical image).
    """
    meta = content_meta or {}

    if meta.get("has_pictures") and int(meta.get("n_texts") or 0) == 0:
        return RouteDecision(
            routed=True,
            reason="docling_image_only_no_text",
            confidence=0.95,
            audit_tag="KEEP_IMAGE_NO_OCR",
        )

    if meta.get("is_structurally_empty"):
        return RouteDecision(
            routed=True,
            reason="docling_structurally_empty",
            confidence=1.0,
            audit_tag="BLANK_ABSOLUTE_CANDIDATE",
        )

    text = (ocr_text or "").strip()
    if not text:
        return RouteDecision(
            routed=True,
            reason="empty_ocr",
            confidence=1.0,
            audit_tag="BLANK_ABSOLUTE_CANDIDATE",
        )

    n_dict = count_dictionary_words(text)
    if n_dict < min_dictionary_words and len(text) < 40:
        return RouteDecision(
            routed=True,
            reason="insufficient_dictionary_words",
            confidence=0.9,
            audit_tag="UNREADABLE_OCR",
        )
    return RouteDecision(routed=False, reason=None, confidence=0.0)
