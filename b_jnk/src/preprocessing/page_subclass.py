"""JUNK / BLANK subclass rules shared by synthetic-data checking, training and inference.

The model decides KEEP vs BLANK vs JUNK. These rules only name the subtype of a
page that is already BLANK or JUNK, so training labels and inference subtypes
always agree. When a page carries triggers of several subtypes, its main
purpose wins: the subtype named first in the page heading, else the one with
the most triggers. The typology list order (Invoice, Cover Page, Record
Request, Instructions, Letter/Fax, Others) only breaks exact ties.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

JUNK_SUBTYPES = (
    "JUNK_INVOICE",
    "JUNK_COVER_PAGE",
    "JUNK_RECORD_REQUEST",
    "JUNK_INSTRUCTIONS",
    "JUNK_LETTER_FAX",
    "JUNK_OTHERS",
)
BLANK_SUBTYPES = (
    "BLANK_ABSOLUTE",
    "BLANK_TECHNICAL",
    "BLANK_SYSTEM",
    "BLANK_HEADER_FOOTER_ONLY",
)

SHORT_PAGE_MAX_WORDS = 19
HEADING_WORDS = 30

INVOICE_STRONG = (
    "invoice", "superbill", "billing statement", "statement of account", "remittance",
    "amount due", "balance due", "total due", "payment due",
)
INVOICE_WEAK = (
    "bill to", "ship to", "unit price", "charge description", "payments received",
    "revenue reconciliation",
)
TRIGGERS: dict[str, tuple[str, ...]] = {
    "JUNK_COVER_PAGE": ("accept", "unaccept", "cover page"),
    "JUNK_RECORD_REQUEST": (
        "medical records request", "record request", "records requested", "transmittal sheet",
        "records attached",
        "risk adjustment request", "audit fulfillment",
    ),
    "JUNK_INSTRUCTIONS": (
        "what to send", "provide documentation", "instructions for sending",
        "instructions for submission", "instructions for records", "please send",
        "documents to include", "documents to send", "documents to attach",
    ),
    "JUNK_LETTER_FAX": (
        "cover letter", "fax transmission", "fax cover", "facsimile", "fax sheet", "this fax",
        "intended recipient",
    ),
}
PROXIMITY_NAME = "records~request"
RECORD_WORDS = frozenset({"records"})
REQUEST_WORDS = frozenset({"request", "requests", "requested", "requesting"})
PROXIMITY_WINDOW = 5

SYSTEM_PHRASES = (
    "this page intentionally left blank", "intentionally left blank", "page left blank",
    "no content on this page",
)
SIGNATURE_RE = re.compile(r"signed by|electronically signed|attestation", re.I)
CLINICAL_SECTION_RE = re.compile(
    r"(?<![a-z])(encounters|medications?|allergies|problems?|vital signs|results|procedures?|"
    r"labs?|imaging|diagnos[ie]s|assessment|immunizations?|history|plan of (?:care|treatment))(?![a-z])",
    re.I,
)

MODEL_TAG_TO_SUBTYPE: dict[str, str] = {
    **{s: s for s in JUNK_SUBTYPES},
    "JUNK_INSURANCE_ID": "JUNK_INVOICE",
    "JUNK_COVER_REQUEST": "JUNK_RECORD_REQUEST",
    "JUNK_FAX_TRANSMISSION": "JUNK_LETTER_FAX",
    "JUNK_SEPARATOR_BARCODE": "JUNK_OTHERS",
    "JUNK_PRINTER_SYSTEM_TEST": "JUNK_OTHERS",
    "JUNK_POSTAL_MAIL": "JUNK_OTHERS",
    "JUNK_BLACK_SCAN_DEFECT": "JUNK_OTHERS",
    "JUNK_NON_CLINICAL_PHOTO": "JUNK_OTHERS",
    "JUNK": "JUNK_OTHERS",
}


def _phrase_re(phrase: str) -> re.Pattern[str]:
    # Word-start match; trailing letters allowed ("invoiced", "accepted").
    return re.compile(r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, phrase.split())), re.I)


MATCHERS: dict[str, re.Pattern[str]] = {
    p: _phrase_re(p)
    for p in (*INVOICE_STRONG, *INVOICE_WEAK, *(q for v in TRIGGERS.values() for q in v))
}


def word_count(text: str) -> int:
    return len((text or "").split())


def _records_request_start(text: str) -> int | None:
    """Char offset of the first "records" ... "request" pair within the window, else None."""
    tokens = [(m.group(0), m.start()) for m in re.finditer(r"[a-z0-9]+", (text or "").lower())]
    rec = [i for i, (w, _) in enumerate(tokens) if w in RECORD_WORDS]
    req = [i for i, (w, _) in enumerate(tokens) if w in REQUEST_WORDS]
    pairs = [min(a, b) for a in rec for b in req if abs(a - b) <= PROXIMITY_WINDOW]
    return tokens[min(pairs)][1] if pairs else None


def records_near_request(text: str) -> bool:
    return _records_request_start(text) is not None


def scan(text: str) -> tuple[dict[str, list[str]], set[str]]:
    """Trigger hits per group ("strong", "weak", JUNK_* subtypes) and the flat set."""
    text = text or ""
    groups: dict[str, list[str]] = {
        "strong": [p for p in INVOICE_STRONG if MATCHERS[p].search(text)],
        "weak": [p for p in INVOICE_WEAK if MATCHERS[p].search(text)],
    }
    for subtype, phrases in TRIGGERS.items():
        groups[subtype] = [p for p in phrases if MATCHERS[p].search(text)]
    if records_near_request(text):
        groups["JUNK_RECORD_REQUEST"].append(PROXIMITY_NAME)
    flat = {p for hits in groups.values() for p in hits}
    return groups, flat


def _candidate_hits(text: str) -> dict[str, list[str]]:
    """Subtypes whose trigger rule is satisfied, with the phrases that satisfied it."""
    groups, _ = scan(text)
    out: dict[str, list[str]] = {}
    if groups["strong"] or len(groups["weak"]) >= 2:
        out["JUNK_INVOICE"] = groups["strong"] + (groups["weak"] if len(groups["weak"]) >= 2 else [])
    if groups["JUNK_COVER_PAGE"] and word_count(text) <= SHORT_PAGE_MAX_WORDS:
        out["JUNK_COVER_PAGE"] = groups["JUNK_COVER_PAGE"]
    for s in ("JUNK_RECORD_REQUEST", "JUNK_INSTRUCTIONS", "JUNK_LETTER_FAX"):
        if groups[s]:
            out[s] = groups[s]
    return out


def junk_trigger_groups(text: str) -> list[str]:
    """Junk subtypes whose triggers fire on the page (for reviewing KEEP pages)."""
    return list(_candidate_hits(text))


def _word_index(text: str, phrase: str) -> int:
    """Word position of the phrase's first occurrence (large if absent)."""
    if phrase == PROXIMITY_NAME:
        pos = _records_request_start(text)
    else:
        m = MATCHERS[phrase].search(text)
        pos = m.start() if m else None
    return len(text[:pos].split()) if pos is not None else 10**9


def main_purpose(text: str, candidates: dict[str, list[str]]) -> tuple[str, str]:
    """Pick one subtype among several triggered ones: earliest in the heading, else most
    triggers, else typology list order."""
    first = {s: min((_word_index(text, p), p) for p in hits) for s, hits in candidates.items()}
    order = {s: i for i, s in enumerate(JUNK_SUBTYPES)}
    in_heading = [s for s in candidates if first[s][0] < HEADING_WORDS]
    if in_heading:
        best = min(in_heading, key=lambda s: (first[s][0], order[s]))
        return best, f"heading:{first[best][1]}"
    best = min(candidates, key=lambda s: (-len(candidates[s]), order[s]))
    return best, f"most_triggers:{'+'.join(candidates[best][:3])}"


def is_gibberish(text: str) -> bool:
    tokens = [t.strip(".,;:!?()[]{}'\"|-_*") for t in (text or "").split()]
    tokens = [t for t in tokens if t and not re.fullmatch(r"[\d/\-.:#$%]+", t)]
    if len(tokens) < 20:
        return False
    wordlike = sum(
        1 for t in tokens if re.fullmatch(r"[A-Za-z][a-z]+|[A-Z]{2,}", t) and re.search(r"[aeiouyAEIOUY]", t)
    )
    return wordlike / len(tokens) < 0.5


@dataclass(frozen=True)
class Subclass:
    subclass: str
    reason: str


def assign_junk_subtype(text: str, model_tag: str | None = None, *, model_picks_mixed: bool = False) -> Subclass:
    """Every JUNK page gets exactly one subtype.

    ``model_picks_mixed``: when several subtypes are triggered and the model's
    subtype is one of them, trust the model's reading of the page's purpose.
    """
    text = text or ""
    wc = word_count(text)
    candidates = _candidate_hits(text)
    if len(candidates) == 1:
        subtype, hits = next(iter(candidates.items()))
        return Subclass(subtype, f"trigger:{hits[0]}")
    if candidates:
        tag = MODEL_TAG_TO_SUBTYPE.get(model_tag or "")
        if model_picks_mixed and tag in candidates:
            return Subclass(tag, f"model_purpose:{model_tag}")
        subtype, reason = main_purpose(text, candidates)
        return Subclass(subtype, reason)
    if wc <= SHORT_PAGE_MAX_WORDS and not SIGNATURE_RE.search(text) and not CLINICAL_SECTION_RE.search(text):
        return Subclass("JUNK_OTHERS", "short_page")
    if is_gibberish(text):
        return Subclass("JUNK_OTHERS", "gibberish")
    if model_tag and model_tag in MODEL_TAG_TO_SUBTYPE:
        return Subclass(MODEL_TAG_TO_SUBTYPE[model_tag], f"model:{model_tag}")
    return Subclass("JUNK_OTHERS", "default")


def assign_blank_subtype(text: str, *, structurally_empty: bool = False) -> Subclass:
    text = text or ""
    if structurally_empty or not text.strip():
        return Subclass("BLANK_ABSOLUTE", "structurally_empty" if structurally_empty else "empty_text")
    low = " ".join(text.lower().split())
    for phrase in SYSTEM_PHRASES:
        if phrase in low:
            return Subclass("BLANK_SYSTEM", f"phrase:{phrase}")
    if re.search(r"[A-Za-z]{3,}", text):
        return Subclass("BLANK_HEADER_FOOTER_ONLY", "letterhead_or_footer_words")
    return Subclass("BLANK_TECHNICAL", "marks_only")
