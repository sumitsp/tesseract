"""Blank & Junk elimination protocol — typology + mandatory retention.

Coarse output remains KEEP | BLANK | JUNK (flagging only). Audit subtypes
provide chain-of-custody tags for QC. Mandatory retention rules hard-block
JUNK/BLANK when clinical-image or demographic signals are present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.preprocessing.text_normalize import collapse_whitespace, normalize_unicode

# ---------------------------------------------------------------------------
# 3.1 Blank typology
# ---------------------------------------------------------------------------
BLANK_PATTERNS: dict[str, tuple[str, ...]] = {
    "BLANK_ABSOLUTE": (),  # empty OCR — handled by routing, not regex
    "BLANK_TECHNICAL": (
        r"scan(?:ning)?\s*noise",
        r"border\s*shadow",
        r"background\s*speck",
    ),
    "BLANK_SYSTEM": (
        r"this page intentionally left blank",
        r"intentionally left blank",
        r"page left blank",
        r"no content on this page",
    ),
    "BLANK_HEADER_FOOTER_ONLY": (
        r"page\s*\d+\s*(?:of|/)\s*\d+",
        r"confidential\s*-\s*do not",
    ),
}

# ---------------------------------------------------------------------------
# 3.2 Junk typology (administrative / non-clinical)
# ---------------------------------------------------------------------------
JUNK_PATTERNS: dict[str, tuple[str, ...]] = {
    "JUNK_COVER_REQUEST": (
        r"medical records?\s*request",
        r"request for (?:medical )?records",
        r"records?\s*retrieval",
        r"cover letter",
        r"copy service",
        r"fulfillment",
        r"e-?request",
        r"pull list",
        r"transmittal\s*(?:sheet|form|letter)?",
    ),
    "JUNK_FAX_TRANSMISSION": (
        r"transmission\s*(?:successful|ok|complete|report|result)",
        r"fax\s*(?:transmission|report|status|log)",
        r"pages?\s*sent",
        r"delivery\s*(?:receipt|confirmation)",
        r"\btx\s*report\b",
    ),
    "JUNK_SEPARATOR_BARCODE": (
        r"\bbarcode\b",
        r"batch\s*separator",
        r"document\s*divider",
        r"patch\s*code",
        r"scanner\s*calibration",
        r"separator\s*sheet",
    ),
    "JUNK_PRINTER_SYSTEM_TEST": (
        r"printer\s*(?:test|diagnostic|configuration|alignment)",
        r"hardware\s*(?:test|alignment)",
        r"configuration\s*printout",
        r"print(?:er)?\s*test\s*page",
        r"printed by .+\d{1,2}/\d{1,2}/\d{2,4}",  # weak: footer crumbs
    ),
    "JUNK_POSTAL_MAIL": (
        r"certified\s*mail",
        r"postal\s*receipt",
        r"courier",
        r"shipping\s*label",
        r"usps\b",
        r"fedex\b",
        r"ups\b",
        r"tracking\s*(?:number|#|no)",
    ),
    "JUNK_INSURANCE_ID": (
        r"driver'?s?\s*license",
        r"\bdl\s*#",
        r"social\s*security",
        r"\bssn\b",
        r"health\s*insurance\s*card",
        r"member\s*id\s*#",
        r"subscriber\s*id",
        r"copay",
        r"out[- ]of[- ]pocket",
    ),
    "JUNK_BLACK_SCAN_DEFECT": (
        r"inverted\s*(?:scan|image|page)",
        r"black\s*(?:scan|background)\s*defect",
        r"unreadable\s*(?:white\s*)?font",
        r"scan(?:ning)?\s*malfunction",
    ),
    "JUNK_NON_CLINICAL_PHOTO": (
        r"patient\s*(?:photo|headshot|portrait)",
        r"facial\s*(?:photo|image|headshot)",
        r"facility\s*(?:photo|picture)",
        r"non[- ]clinical\s*(?:photo|image)",
    ),
}

# ---------------------------------------------------------------------------
# Mandatory retention — NEVER mark JUNK/BLANK
# ---------------------------------------------------------------------------
CLINICAL_IMAGE_PATTERNS: tuple[str, ...] = (
    r"wound\s*(?:photo|image|tracking|care)",
    r"dermatolog(?:y|ic)\s*(?:photo|image|lesion)",
    r"lesion\s*(?:photo|image|capture)",
    r"endoscop(?:y|ic)",
    r"colonoscop(?:y|ic)",
    r"retinal\s*(?:scan|image|photo|photo)",
    r"fundus\s*(?:photo|image|exam)",
    r"ophthalmolog",
    r"clinical\s*(?:photo|image|imaging)",
    r"oct\s*(?:nerve|mac|scan)",
    r"slit\s*lamp",
)

DEMOGRAPHIC_PATTERNS: tuple[str, ...] = (
    r"patient\s*demographics?",
    r"\bdemographics?\b",
    r"registration\s*(?:sheet|form|page)?",
    r"face\s*sheet",
    r"\bfacesheet\b",
    r"member\s*verification",
    r"patient\s*(?:information|info)\s*(?:sheet|form)?",
    r"guarantor",
    r"emergency\s*contact",
    r"preferred\s*(?:name|language|pharmacy)",
    r"medical\s*record\s*number",
    r"\bmrn\b",
    r"date of birth",
    r"\bdob\b",
)


def _compile(pats: tuple[str, ...] | list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in pats]


_BLANK = {k: _compile(v) for k, v in BLANK_PATTERNS.items()}
_JUNK = {k: _compile(v) for k, v in JUNK_PATTERNS.items()}
_CLINICAL_IMAGE = _compile(CLINICAL_IMAGE_PATTERNS)
_DEMOGRAPHIC = _compile(DEMOGRAPHIC_PATTERNS)


@dataclass
class ProtocolHit:
    audit_tag: str | None
    retain_clinical_image: bool
    retain_demographic: bool
    evidence: list[str]


def _first_hits(
    text: str, groups: dict[str, list[re.Pattern[str]]], limit: int = 3
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tag, patterns in groups.items():
        for p in patterns:
            m = p.search(text)
            if m:
                out.append((tag, m.group(0)))
                break
        if len(out) >= limit:
            break
    return out


def analyze_protocol(ocr_text: str) -> ProtocolHit:
    text = collapse_whitespace(normalize_unicode(ocr_text or ""))
    clinical = any(p.search(text) for p in _CLINICAL_IMAGE)
    demo = any(p.search(text) for p in _DEMOGRAPHIC)

    evidence: list[str] = []
    audit: str | None = None

    if clinical:
        evidence.append("retain:clinical_image")
        audit = "KEEP_CLINICAL_IMAGE"
    if demo:
        evidence.append("retain:demographic")
        if audit is None:
            audit = "KEEP_DEMOGRAPHIC"

    junk_hits = _first_hits(text, _JUNK)
    blank_hits = _first_hits(text, _BLANK)

    # Retention wins over junk/blank typology for audit tag
    if not clinical and not demo:
        if junk_hits:
            audit = junk_hits[0][0]
            evidence.extend(f"{t}:{m.lower()}" for t, m in junk_hits[:2])
        elif blank_hits:
            audit = blank_hits[0][0]
            evidence.extend(f"{t}:{m.lower()}" for t, m in blank_hits[:2])
        elif not text.strip():
            audit = "BLANK_ABSOLUTE"
            evidence.append("empty_ocr")

    return ProtocolHit(
        audit_tag=audit,
        retain_clinical_image=clinical,
        retain_demographic=demo,
        evidence=evidence,
    )


def must_retain(hit: ProtocolHit) -> bool:
    """Protocol § retention safeguards — never junk/blank these pages."""
    return hit.retain_clinical_image or hit.retain_demographic
