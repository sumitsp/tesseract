"""OCR text normalization for n-grams and pattern features."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    # Non-breaking spaces and soft hyphens from bad OCR dumps
    text = text.replace("\xa0", " ").replace("\u00ad", "")
    return text


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_for_ngrams(text: str, *, mask_digits: bool = True) -> str:
    """Lowercase + whitespace collapse; optionally mask digits for templates."""
    text = collapse_whitespace(normalize_unicode(text)).lower()
    if mask_digits:
        text = _DIGIT.sub("0", text)
    return text


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]{1,}", text or "")
