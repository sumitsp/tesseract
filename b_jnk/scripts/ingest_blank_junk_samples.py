#!/usr/bin/env python3
"""Ingest human blank/junk OCR dumps into ehr_pages_v0.3.jsonl and retrain.

Uses Blank & Junk elimination protocol typology for primary_class / flag /
secondary audit subtype. All pages in these dumps are DROP (BLANK or JUNK),
not KEEP — per annotator.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.protocol import analyze_protocol  # noqa: E402
from src.preprocessing.taxonomy import ACTION_BY_CLASS  # noqa: E402

PAGE_SPLIT = re.compile(r"=====+\s*(\d+)\.([A-Za-z0-9]+)\s*=====", re.I)

# Sources (attachment copies land under data/raw/blank_junk_sources_v0.3/)
SOURCE_DIR = ROOT / "data/raw/blank_junk_sources_v0.3"
ATTACH = Path(
    "/Users/sumit/.cursor/projects/Users-sumit-Algodel-tesseract/attachments/"
    "b6599b97-a36f-46e8-bb96-49eef332783c"
)
JSONL = ROOT / "data/raw/ehr_pages_v0.3.jsonl"


def _sha1(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _clean(text: str) -> str:
    text = (text or "").replace("\u2028", "\n").replace("\xa0", " ")
    # Docling placeholder is not real content for training text features
    if text.strip().lower() in {"[no text detected]", "no text detected"}:
        return ""
    return text.strip()


def split_pages(raw: str) -> list[tuple[str, str, str]]:
    parts = PAGE_SPLIT.split(raw)
    out: list[tuple[str, str, str]] = []
    for i in range(1, len(parts), 3):
        stem, ext, body = parts[i], parts[i + 1], parts[i + 2] if i + 2 < len(parts) else ""
        out.append((stem, ext.lower(), _clean(body)))
    return out


def assign_labels(ocr: str) -> tuple[str, str, str]:
    """Return (primary_class, flag, secondary_type) for a blank/junk page."""
    hit = analyze_protocol(ocr)
    text = ocr or ""
    n = len(text)
    low = text.lower()

    # Absolute / near-empty
    if n == 0:
        return "BLANK_ABSOLUTE", "BLANK", "BLANK_ABSOLUTE"
    if n < 25 and not re.search(r"[a-zA-Z]{4,}", text):
        return "BLANK_TECHNICAL", "BLANK", "BLANK_TECHNICAL"

    if hit.audit_tag and hit.audit_tag.startswith("BLANK_"):
        return hit.audit_tag, "BLANK", hit.audit_tag
    if hit.audit_tag and hit.audit_tag.startswith("JUNK_"):
        return hit.audit_tag, "JUNK", hit.audit_tag

    # Short facility crumbs / END OF ENCOUNTER / Accept ICD — separators / crumbs
    if n < 80:
        if re.search(r"end of encounter|accept|unaccept|icd-?10|discharge\s*summary", low):
            return "JUNK_SEPARATOR_BARCODE", "JUNK", "JUNK_SEPARATOR_BARCODE"
        if re.search(r"page\s*\d+|printed by", low):
            return "BLANK_HEADER_FOOTER_ONLY", "BLANK", "BLANK_HEADER_FOOTER_ONLY"
        return "BLANK_TECHNICAL", "BLANK", "BLANK_TECHNICAL"

    # Upload receipts / portal confirmations
    if re.search(r"files have been received|confirmation number|start new u", low):
        return "JUNK_PRINTER_SYSTEM_TEST", "JUNK", "JUNK_PRINTER_SYSTEM_TEST"

    # HEDIS / measure guide packets inside request dumps
    if re.search(r"measure acronym|what to send|hedis|member demographic sheet", low):
        return "JUNK_COVER_REQUEST", "JUNK", "JUNK_COVER_REQUEST"

    # Default admin junk if looks like request/fax-ish long text
    if re.search(r"medical records|request|transmittal|advantmed|ciox|mro|optum|wellmed", low):
        return "JUNK_COVER_REQUEST", "JUNK", "JUNK_COVER_REQUEST"

    return "JUNK_COVER_REQUEST", "JUNK", "JUNK_COVER_REQUEST"


def make_record(
    *,
    page_id: str,
    ocr: str,
    source_file: str,
    document_id: str,
    batch_id: str,
) -> dict:
    primary, flag, secondary = assign_labels(ocr)
    keep_delete = ACTION_BY_CLASS.get(primary) or (
        "DELETE" if flag in {"BLANK", "JUNK"} else "KEEP"
    )
    return {
        "page_id": page_id,
        "ocr_text": ocr,
        "label": {
            "primary_class": primary,
            "keep_delete": keep_delete,
            "flag": flag,
            "confidence": "HIGH",
            "secondary_type": secondary,
        },
        "annotation": {
            "evidence_from_ocr": (ocr[:160] or "[empty]").replace("\n", " "),
            "reason_for_label": (
                "Human blank/junk sample packet; typed via Blank & Junk "
                f"elimination protocol → {secondary}"
            ),
            "hard_negative": False,
            "hard_negative_type": [],
            "ambiguous": False,
            "review_required": False,
            "candidate_classes": [primary],
            "label_source": "human_blank_junk_packet",
            "label_version": "v0.3.2",
        },
        "meta": {
            "source_file": source_file,
            "document_id": document_id,
            "batch_id": batch_id,
            "facility": None,
            "vendor": None,
            "template_group": document_id,
            "split_group": document_id,
            "text_hash": _sha1(ocr),
            "ocr_char_count": len(ocr),
        },
        "train_eligible": True,
    }


def main() -> int:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if ATTACH.exists():
        for src in (ATTACH / "Untitled__12_.txt", ATTACH / "Untitled__13_.txt"):
            if src.exists():
                (SOURCE_DIR / src.name).write_bytes(src.read_bytes())

    new_rows: list[dict] = []
    for path in sorted(SOURCE_DIR.glob("Untitled__*.txt")):
        pages = split_pages(path.read_text(encoding="utf-8", errors="replace"))
        doc_id = f"bjunk_{path.stem}"
        for stem, ext, ocr in pages:
            pid = f"v03_{doc_id}_p{int(stem):03d}.{ext}"
            new_rows.append(
                make_record(
                    page_id=pid,
                    ocr=ocr,
                    source_file=path.name,
                    document_id=doc_id,
                    batch_id="blank_junk_samples_v0.3.2",
                )
            )
        print(f"{path.name}: {len(pages)} pages")

    if not new_rows:
        print("No blank/junk source files found")
        return 2

    existing: list[dict] = []
    if JSONL.exists():
        for line in JSONL.open(encoding="utf-8"):
            line = line.strip()
            if line:
                existing.append(json.loads(line))

    # Drop prior ingest of these packets, then append
    drop_batches = {"blank_junk_samples_v0.3.2"}
    drop_prefixes = ("v03_bjunk_Untitled__12_", "v03_bjunk_Untitled__13_")
    kept = [
        r
        for r in existing
        if (r.get("meta") or {}).get("batch_id") not in drop_batches
        and not str(r.get("page_id", "")).startswith(drop_prefixes)
    ]

    # Dedup by text_hash among new vs kept (prefer new blank/junk labels over old KEEP)
    by_hash = {
        (r.get("meta") or {}).get("text_hash"): r
        for r in kept
        if (r.get("ocr_text") or "").strip()
    }
    final = []
    used = set()
    # Prefer new rows when hash collides
    for r in new_rows:
        h = (r.get("meta") or {}).get("text_hash")
        by_hash[h] = r
    for r in kept:
        h = (r.get("meta") or {}).get("text_hash")
        if not (r.get("ocr_text") or "").strip():
            final.append(r)
            continue
        if h in used:
            continue
        final.append(by_hash.get(h, r))
        used.add(h)
    for r in new_rows:
        h = (r.get("meta") or {}).get("text_hash")
        if not (r.get("ocr_text") or "").strip():
            # keep empty blanks (multiple absolute blanks allowed)
            final.append(r)
            continue
        if h in used:
            continue
        final.append(r)
        used.add(h)

    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with JSONL.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    flags = Counter((r.get("label") or {}).get("flag") for r in new_rows)
    subtypes = Counter((r.get("label") or {}).get("secondary_type") for r in new_rows)
    print("new flags", dict(flags))
    print("new subtypes", dict(subtypes))
    print(f"Wrote {len(final)} total rows → {JSONL} (added {len(new_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
