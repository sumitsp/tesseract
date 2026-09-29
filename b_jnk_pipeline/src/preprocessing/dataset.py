"""Load and validate OCR page JSONL records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.preprocessing.taxonomy import ACTION_BY_CLASS, ALL_CLASSES, Flag, to_flag


@dataclass
class PageRecord:
    page_id: str
    ocr_text: str
    primary_class: str
    keep_delete: str
    confidence: str
    secondary_type: str | None = None
    hard_negative: bool = False
    hard_negative_type: list[str] = field(default_factory=list)
    ambiguous: bool = False
    review_required: bool = False
    train_eligible: bool = False
    document_id: str | None = None
    batch_id: str | None = None
    facility: str | None = None
    vendor: str | None = None
    template_group: str | None = None
    split_group: str | None = None
    text_hash: str | None = None
    ocr_char_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def group_key(self) -> str:
        return self.split_group or self.template_group or self.batch_id or self.page_id

    @property
    def flag(self) -> Flag | None:
        """Coarse training/eval target: KEEP | BLANK | JUNK (None for REVIEW)."""
        return to_flag(self.primary_class)


def _sha1_norm(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def parse_record(obj: dict[str, Any]) -> PageRecord:
    label = obj.get("label") or {}
    ann = obj.get("annotation") or {}
    meta = obj.get("meta") or {}
    primary = label.get("primary_class")
    keep = label.get("keep_delete")
    if primary not in ALL_CLASSES:
        raise ValueError(f"{obj.get('page_id')}: unknown class {primary}")
    expected = ACTION_BY_CLASS[primary]
    if keep != expected:
        raise ValueError(
            f"{obj.get('page_id')}: keep_delete={keep} disagrees with "
            f"{primary} (expected {expected})"
        )
    text = obj.get("ocr_text")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"{obj.get('page_id')}: ocr_text must be a string")
    return PageRecord(
        page_id=str(obj["page_id"]),
        ocr_text=text,
        primary_class=primary,
        keep_delete=keep,
        confidence=str(label.get("confidence") or ""),
        secondary_type=label.get("secondary_type"),
        hard_negative=bool(ann.get("hard_negative")),
        hard_negative_type=list(ann.get("hard_negative_type") or []),
        ambiguous=bool(ann.get("ambiguous")),
        review_required=bool(ann.get("review_required")),
        train_eligible=bool(obj.get("train_eligible")),
        document_id=meta.get("document_id"),
        batch_id=meta.get("batch_id"),
        facility=meta.get("facility"),
        vendor=meta.get("vendor"),
        template_group=meta.get("template_group"),
        split_group=meta.get("split_group"),
        text_hash=meta.get("text_hash") or _sha1_norm(text),
        ocr_char_count=int(meta.get("ocr_char_count") or len(text)),
        raw=obj,
    )


def load_jsonl(path: Path | str) -> list[PageRecord]:
    path = Path(path)
    rows: list[PageRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(parse_record(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    _assert_no_label_conflicts(rows)
    return rows


def _assert_no_label_conflicts(rows: Iterable[PageRecord]) -> None:
    by_hash: dict[str, set[str]] = {}
    for r in rows:
        # Empty OCR is identical across absolute blanks / unknown review pages —
        # allow multiple labels (routing decides at inference).
        if not (r.ocr_text or "").strip():
            continue
        by_hash.setdefault(r.text_hash or "", set()).add(r.primary_class)
    conflicts = {h: cs for h, cs in by_hash.items() if h and len(cs) > 1}
    if conflicts:
        raise ValueError(
            "Identical OCR text (text_hash) has conflicting labels: "
            + ", ".join(f"{h}->{sorted(cs)}" for h, cs in conflicts.items())
        )


def filter_training_rows(
    rows: list[PageRecord],
    *,
    require_train_eligible: bool = True,
    include_medium_keep: bool = True,
) -> list[PageRecord]:
    """Select rows eligible for supervised training.

    REVIEW class pages are never training targets.
    Empty OCR is kept out of the classifier (handled by pre-model routing).

    ``include_medium_keep`` optionally admits KEEP pages that annotators marked
    MEDIUM/ambiguous (v0.2 demographics). Labels are not changed; this only
    widens the training filter for bootstrap experiments.
    """
    out: list[PageRecord] = []
    for r in rows:
        if r.primary_class == "UNCERTAIN_REVIEW":
            continue
        if not (r.ocr_text or "").strip():
            continue
        is_bootstrap_keep = (
            include_medium_keep
            and r.keep_delete == "KEEP"
            and r.confidence in {"MEDIUM", "HIGH"}
        )
        is_bootstrap_blank = (
            include_medium_keep
            and r.flag == "BLANK"
            and r.confidence in {"MEDIUM", "HIGH"}
            and bool((r.ocr_text or "").strip())
        )
        if (r.ambiguous or r.review_required) and not (
            is_bootstrap_keep or is_bootstrap_blank
        ):
            continue
        if (
            require_train_eligible
            and not r.train_eligible
            and not is_bootstrap_keep
            and not is_bootstrap_blank
        ):
            continue
        out.append(r)
    return out
