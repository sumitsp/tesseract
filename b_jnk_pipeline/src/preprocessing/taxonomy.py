"""Coarse page flags: KEEP / BLANK / JUNK.

Fine labels from the dataset (RETAIN_*, BLANK_*, JUNK_*) are mapped into these
three flags for training. Inference only emits these flags — nothing is deleted;
pages are flagged for downstream review/routing.
"""

from __future__ import annotations

from typing import Literal

Flag = Literal["KEEP", "BLANK", "JUNK"]

FLAGS: tuple[Flag, ...] = ("KEEP", "BLANK", "JUNK")

# Optional fine labels still accepted in JSONL and collapsed to FLAGS.
RETAIN_CLASSES = (
    "RETAIN_CLINICAL",
    "RETAIN_DEMOGRAPHIC",
    "RETAIN_CLINICAL_IMAGE",
    "KEEP",
)
BLANK_CLASSES = (
    "BLANK_ABSOLUTE",
    "BLANK_TECHNICAL",
    "BLANK_SYSTEM",
    "BLANK_HEADER_FOOTER_ONLY",
    "BLANK",
)
JUNK_CLASSES = (
    "JUNK_COVER_REQUEST",
    "JUNK_FAX_TRANSMISSION",
    "JUNK_SEPARATOR_BARCODE",
    "JUNK_PRINTER_SYSTEM_TEST",
    "JUNK_POSTAL_MAIL",
    "JUNK_INSURANCE_ID",
    "JUNK_BLACK_SCAN_DEFECT",
    "JUNK_NON_CLINICAL_PHOTO",
    "JUNK",
)
REVIEW_CLASSES = ("UNCERTAIN_REVIEW",)

ALL_CLASSES = RETAIN_CLASSES + BLANK_CLASSES + JUNK_CLASSES + REVIEW_CLASSES

# Legacy JSONL field keep_delete is KEEP | DELETE | REVIEW
ACTION_BY_CLASS: dict[str, str] = {
    **{c: "KEEP" for c in RETAIN_CLASSES},
    **{c: "DELETE" for c in BLANK_CLASSES},
    **{c: "DELETE" for c in JUNK_CLASSES},
    **{c: "REVIEW" for c in REVIEW_CLASSES},
}

NOT_LEARNABLE_FROM_TEXT = (
    "BLANK_ABSOLUTE",
    "JUNK_BLACK_SCAN_DEFECT",
    "JUNK_NON_CLINICAL_PHOTO",
)

PARTIAL_FROM_TEXT = (
    "RETAIN_CLINICAL_IMAGE",
    "BLANK_TECHNICAL",
    "JUNK_SEPARATOR_BARCODE",
    "BLANK_HEADER_FOOTER_ONLY",
    "JUNK_INSURANCE_ID",
)


def to_flag(label: str) -> Flag | None:
    """Map any fine/coarse label to KEEP/BLANK/JUNK. REVIEW → None."""
    if label in RETAIN_CLASSES or label == "KEEP":
        return "KEEP"
    if label in BLANK_CLASSES or label == "BLANK":
        return "BLANK"
    if label in JUNK_CLASSES or label == "JUNK":
        return "JUNK"
    if label in REVIEW_CLASSES:
        return None
    raise KeyError(f"Unknown label: {label}")


def action_for(page_type: str) -> Flag | Literal["REVIEW"]:
    """Backward-compatible name: returns flag or REVIEW."""
    if page_type in REVIEW_CLASSES:
        return "REVIEW"
    flag = to_flag(page_type)
    return flag if flag is not None else "REVIEW"


def coarse_bucket(page_type: str) -> str:
    """KEEP / BLANK / JUNK / REVIEW — used by hierarchical trainer."""
    if page_type in REVIEW_CLASSES:
        return "REVIEW"
    flag = to_flag(page_type)
    if flag is not None:
        return flag
    raise KeyError(page_type)
