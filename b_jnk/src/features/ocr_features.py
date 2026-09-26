"""Hand-crafted OCR page features + keyword pattern flags.

Keywords are features for the model — never the final decision rule.
"""

from __future__ import annotations

import math
import re
from typing import Any

from src.preprocessing.text_normalize import (
    collapse_whitespace,
    normalize_unicode,
    tokenize_words,
)

# Lightweight English+common clinical/admin lexicon for "dictionary-like" ratio.
# Not exhaustive; used only as a density feature.
_DICT = frozenset(
    """
    patient name date birth sex male female address phone fax medical record
    records request discharge summary history assessment plan diagnosis
    medication medications allergies allergy vital signs encounter procedure
    lab laboratory imaging results visit hospital clinical note notes
    confidential transmission transmittal cover letter insurance member
    demographic demographics registration face sheet facesheet provider
    physician nurse allergy blood pressure temperature weight height
    admission admitted discharged diagnosis diagnoses treatment therapy
    radiology pathology cardiology oncology dermatology ophthalmology
    endoscopy colonoscopy wound lesion retinal scan photograph photo
    barcode separator printer test page blank system error
    """.split()
)

PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    "fax": (r"\bfax\b", r"transmission", r"transmittal", r"pages?\s*sent"),
    "request_cover": (
        r"medical records? request",
        r"request for medical records",
        r"cover letter",
        r"fulfillment",
        r"e-?request",
        r"pull list",
        r"records retrieval",
    ),
    "confidential": (r"confidential", r"hipaa", r"protected health"),
    "demographic": (
        r"date of birth",
        r"\bdob\b",
        r"patient demographics?",
        r"registration",
        r"face\s*sheet",
        r"account number",
        r"member id",
        r"registro del portal",
    ),
    "clinical": (
        r"discharge summary",
        r"history and physical",
        r"\bh&p\b",
        r"assessment and plan",
        r"\bdiagnosis\b",
        r"\bdiagnoses\b",
        r"medication",
        r"allerg",
        r"vital signs?",
        r"\bencounter\b",
        r"\bprocedure\b",
        r"\blab(oratory)?\b",
        r"imaging",
        r"clinical notes?",
        r"hospital course",
        r"visit diagnosis",
        r"progress note",
    ),
    "vendor_admin": (
        r"advantmed",
        r"datavant",
        r"ciox",
        r"optum",
        r"wellmed",
        r"mro\b",
        r"verisma",
        r"sharecare",
        r"unitedhealthcare",
        r"hedis",
    ),
    "insurance_card": (
        r"subscriber",
        r"copay",
        r"out[- ]of[- ]pocket",
        r"\brx\b",
        r"policy\s*#",
        r"member id\s*#",
    ),
    "separator": (r"\bbarcode\b", r"batch\s*separator", r"divider"),
}

_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_PAGE_NUM = re.compile(r"\bpage\s*\d+\s*(?:of|/)\s*\d+\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_URL = re.compile(r"https?://|www\.", re.I)
_REPEATED = re.compile(r"(.)\1{4,}")


def _compile_groups() -> dict[str, list[re.Pattern[str]]]:
    return {
        name: [re.compile(p, re.I) for p in pats]
        for name, pats in PATTERN_GROUPS.items()
    }


_COMPILED = _compile_groups()


FEATURE_NAMES: list[str] = [
    "char_count",
    "word_count",
    "line_count",
    "avg_word_len",
    "numeric_char_ratio",
    "alpha_char_ratio",
    "special_char_ratio",
    "date_count",
    "phone_count",
    "page_number_count",
    "email_url_count",
    "dictionary_word_ratio",
    "repeated_char_ratio",
    "log_char_count",
    "has_meaningful_words",
] + [f"pat_{name}" for name in PATTERN_GROUPS]


def extract_features(ocr_text: str) -> dict[str, float]:
    raw = normalize_unicode(ocr_text or "")
    text = collapse_whitespace(raw)
    chars = len(text)
    words = tokenize_words(text)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    n_words = len(words)
    alpha = sum(1 for c in text if c.isalpha())
    digit = sum(1 for c in text if c.isdigit())
    special = chars - alpha - digit - text.count(" ")
    dict_hits = sum(1 for w in words if w.lower() in _DICT)
    repeated = sum(len(m.group(0)) for m in _REPEATED.finditer(text))

    feats: dict[str, float] = {
        "char_count": float(chars),
        "word_count": float(n_words),
        "line_count": float(len(lines)),
        "avg_word_len": float(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "numeric_char_ratio": digit / chars if chars else 0.0,
        "alpha_char_ratio": alpha / chars if chars else 0.0,
        "special_char_ratio": max(special, 0) / chars if chars else 0.0,
        "date_count": float(len(_DATE.findall(text))),
        "phone_count": float(len(_PHONE.findall(text))),
        "page_number_count": float(len(_PAGE_NUM.findall(text))),
        "email_url_count": float(len(_EMAIL.findall(text)) + len(_URL.findall(text))),
        "dictionary_word_ratio": dict_hits / n_words if n_words else 0.0,
        "repeated_char_ratio": repeated / chars if chars else 0.0,
        "log_char_count": math.log1p(chars),
        "has_meaningful_words": 1.0 if dict_hits >= 2 else 0.0,
    }
    for name, patterns in _COMPILED.items():
        feats[f"pat_{name}"] = float(sum(1 for p in patterns if p.search(text)))
    return feats


def feature_vector(ocr_text: str) -> list[float]:
    feats = extract_features(ocr_text)
    return [feats[n] for n in FEATURE_NAMES]


def top_evidence(ocr_text: str, limit: int = 5) -> list[str]:
    """Interpretable evidence strings for audit output."""
    text = collapse_whitespace(normalize_unicode(ocr_text or ""))
    if not text:
        return ["empty_ocr"]
    hits: list[str] = []
    for name, patterns in _COMPILED.items():
        for p in patterns:
            m = p.search(text)
            if m:
                hits.append(f"{name}:{m.group(0).lower()}")
    # Prefer unique group labels
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        key = h.split(":", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    if not out:
        feats = extract_features(ocr_text)
        if feats["word_count"] < 5:
            out.append("very_short_ocr")
        elif feats["pat_request_cover"] == 0 and feats["pat_clinical"] == 0:
            out.append("no_strong_pattern_match")
    return out


def count_dictionary_words(ocr_text: str) -> int:
    words = tokenize_words(normalize_unicode(ocr_text or ""))
    return sum(1 for w in words if w.lower() in _DICT)
