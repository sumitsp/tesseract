"""Pre-model routing for empty / image-only / absolute-blank pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.features.ocr_features import count_dictionary_words

Flag = Literal["KEEP", "BLANK", "JUNK"]


@dataclass
class RouteDecision:
    routed: bool
    reason: str | None
    confidence: float
    flag: Flag = "KEEP"
    review_required: bool = True
    audit_tag: str | None = None


def route_empty_or_unreadable(
    ocr_text: str,
    *,
    min_dictionary_words: int = 2,
    content_meta: dict[str, Any] | None = None,
) -> RouteDecision:
    """Route pages that should not go through the text classifier alone.

    Absolute blank = Docling structure with no texts, pictures, tables, forms,
    or body children (not a hardcoded markdown string).

    - Structurally empty → flag=BLANK (absolute blank)
    - Pictures but no texts → KEEP + review (possible clinical image)
    - Empty text without structure → KEEP + review (unsafe to auto-blank)
    """
    meta = content_meta or {}

    if meta.get("has_pictures") and int(meta.get("n_texts") or 0) == 0:
        return RouteDecision(
            routed=True,
            reason="docling_image_only_no_text",
            confidence=0.95,
            flag="KEEP",
            review_required=True,
            audit_tag="KEEP_IMAGE_NO_OCR",
        )

    if meta.get("is_structurally_empty"):
        return RouteDecision(
            routed=True,
            reason="docling_structurally_empty",
            confidence=1.0,
            flag="BLANK",
            review_required=False,
            audit_tag="BLANK_ABSOLUTE",
        )

    text = (ocr_text or "").strip()
    if not text:
        # No Docling structure — cannot prove absolute blank vs image OCR miss
        return RouteDecision(
            routed=True,
            reason="empty_ocr_no_structure",
            confidence=1.0,
            flag="KEEP",
            review_required=True,
            audit_tag="BLANK_ABSOLUTE_CANDIDATE",
        )

    n_dict = count_dictionary_words(text)
    if n_dict < min_dictionary_words and len(text) < 40:
        return RouteDecision(
            routed=True,
            reason="insufficient_dictionary_words",
            confidence=0.9,
            flag="KEEP",
            review_required=True,
            audit_tag="UNREADABLE_OCR",
        )
    return RouteDecision(routed=False, reason=None, confidence=0.0)
