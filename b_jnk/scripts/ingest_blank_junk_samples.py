#!/usr/bin/env python3
"""Ingest Untitled__12_/13_ OCR dumps into ehr_pages_v0.3.jsonl.

Labels come from curated annotations in
``data/labels/bjunk_packet_labels_v0.3.json`` (human page review).
Clinical medicine / chart pages in those dumps are KEEP — never inferred
from keyword heuristics.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.taxonomy import ACTION_BY_CLASS  # noqa: E402

PAGE_SPLIT = re.compile(r"=====+\s*(\d+)\.([A-Za-z0-9]+)\s*=====", re.I)

SOURCE_DIR = ROOT / "data/raw/blank_junk_sources_v0.3"
ATTACH = Path(
    "/Users/sumit/.cursor/projects/Users-sumit-Algodel-tesseract/attachments/"
    "b6599b97-a36f-46e8-bb96-49eef332783c"
)
JSONL = ROOT / "data/raw/ehr_pages_v0.3.jsonl"
LABELS = ROOT / "data/labels/bjunk_packet_labels_v0.3.json"
BATCH_ID = "blank_junk_samples_v0.3.3"


def _sha1(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _clean(text: str) -> str:
    text = (text or "").replace("\u2028", "\n").replace("\xa0", " ")
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


def load_curated() -> dict[str, dict]:
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    return dict(data.get("pages") or {})


def make_record(
    *,
    page_id: str,
    file_key: str,
    ocr: str,
    source_file: str,
    document_id: str,
    curated: dict[str, dict],
) -> dict:
    lab = curated.get(file_key)
    if lab is None:
        raise KeyError(
            f"Missing curated label for {file_key} in {LABELS.name}. "
            "Add a page review entry — do not invent keyword rules."
        )
    primary = lab["primary_class"]
    flag = lab["flag"]
    hard_neg = bool(lab.get("hard_negative"))
    note = lab.get("note") or f"curated {flag}/{primary}"
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
            "secondary_type": primary,
        },
        "annotation": {
            "evidence_from_ocr": (ocr[:160] or "[empty]").replace("\n", " "),
            "reason_for_label": note,
            "hard_negative": hard_neg,
            "hard_negative_type": ["clinical_vs_admin_junk"] if hard_neg else [],
            "ambiguous": False,
            "review_required": False,
            "candidate_classes": [primary],
            "label_source": "curated_bjunk_packet_v0.3.3",
            "label_version": "v0.3.3",
        },
        "meta": {
            "source_file": source_file,
            "document_id": document_id,
            "batch_id": BATCH_ID,
            "facility": None,
            "vendor": None,
            "template_group": document_id,
            "split_group": document_id,
            "text_hash": _sha1(ocr),
            "ocr_char_count": len(ocr),
        },
        "train_eligible": True,
    }


def _prefer_keep(a: dict, b: dict) -> dict:
    """On identical OCR, keep clinical KEEP over admin JUNK."""
    fa = (a.get("label") or {}).get("flag")
    fb = (b.get("label") or {}).get("flag")
    if fa == "KEEP" and fb != "KEEP":
        return a
    if fb == "KEEP" and fa != "KEEP":
        return b
    return b  # newer annotation wins when same flag family


def main() -> int:
    curated = load_curated()
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
            # Untitled__12_.txt → Untitled__12__p001.png
            file_key = f"{path.stem}_p{int(stem):03d}.{ext}"
            pid = f"v03_{doc_id}_p{int(stem):03d}.{ext}"
            new_rows.append(
                make_record(
                    page_id=pid,
                    file_key=file_key,
                    ocr=ocr,
                    source_file=path.name,
                    document_id=doc_id,
                    curated=curated,
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

    drop_batches = {"blank_junk_samples_v0.3.2", BATCH_ID}
    drop_prefixes = ("v03_bjunk_Untitled__12_", "v03_bjunk_Untitled__13_")
    kept = [
        r
        for r in existing
        if (r.get("meta") or {}).get("batch_id") not in drop_batches
        and not str(r.get("page_id", "")).startswith(drop_prefixes)
    ]

    by_hash: dict[str, dict] = {}
    for r in kept:
        h = (r.get("meta") or {}).get("text_hash") or ""
        if not (r.get("ocr_text") or "").strip():
            continue
        by_hash[h] = r
    for r in new_rows:
        h = (r.get("meta") or {}).get("text_hash") or ""
        if not (r.get("ocr_text") or "").strip():
            continue
        if h in by_hash:
            by_hash[h] = _prefer_keep(by_hash[h], r)
        else:
            by_hash[h] = r

    final: list[dict] = []
    used: set[str] = set()
    for r in kept:
        if not (r.get("ocr_text") or "").strip():
            final.append(r)
            continue
        h = (r.get("meta") or {}).get("text_hash") or ""
        if h in used:
            continue
        chosen = by_hash.get(h, r)
        final.append(chosen)
        used.add(h)
    for r in new_rows:
        if not (r.get("ocr_text") or "").strip():
            final.append(r)
            continue
        h = (r.get("meta") or {}).get("text_hash") or ""
        if h in used:
            continue
        final.append(by_hash.get(h, r))
        used.add(h)

    JSONL.parent.mkdir(parents=True, exist_ok=True)
    with JSONL.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter

    flags = Counter((r.get("label") or {}).get("flag") for r in new_rows)
    print("new flags", dict(flags))
    print(
        "KEEP hard-negatives",
        sum(1 for r in new_rows if (r.get("annotation") or {}).get("hard_negative")),
    )
    print(f"Wrote {len(final)} total rows → {JSONL} (packet {len(new_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
