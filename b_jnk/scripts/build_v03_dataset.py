#!/usr/bin/env python3
"""Build ehr_pages_v0.3.jsonl: v0.2 blank/junk + new KEEP clinical OCR.

Sources:
  - data/raw/ehr_pages_v0.2.jsonl  (existing BLANK/JUNK/REVIEW + 2 KEEP)
  - data/raw/keep_sources_v0.3/    (KEEP OCR dumps, page-split on ===== N.jpg =====)

Inference/training use coarse flags KEEP|BLANK|JUNK via taxonomy.to_flag().
New KEEP pages are labeled primary_class=KEEP, keep_delete=KEEP.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE_SPLIT = re.compile(r"=====+\s*(\d+)\.(?:jpg|png|jpeg)\s*=====", re.I)
PACKET_SPLIT = re.compile(r"(?im)^\s*\d+\)\s*(?======)")
MRN_RE = re.compile(r"\bMRN[:\s#]*([0-9A-Za-z\-]{4,})", re.I)
PRN_RE = re.compile(r"\bPRN[:\s#]*([0-9A-Za-z\-]{4,})", re.I)
ACCT_RE = re.compile(r"\bAcct[#:\s]*([0-9]{3,})", re.I)
CHART_RE = re.compile(r"\bChart[#:\s]*([0-9]{3,})", re.I)


def _sha1_norm(text: str) -> str:
    norm = " ".join((text or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _clean_ocr(text: str) -> str:
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.replace("\xa0", " ")
    # Drop trailing human labels like "all keep"
    text = re.sub(r"(?im)^\s*all keep\s*$", "", text)
    text = re.sub(r"(?im)^\s*these are all real documents ocr\s*$", "", text)
    text = re.sub(r"(?im)^\s*are they enough\?+\s*$", "", text)
    return text.strip()


def _guess_doc_key(text: str, fallback: str) -> str:
    head = text[:800]
    for rx in (MRN_RE, PRN_RE, ACCT_RE, CHART_RE):
        m = rx.search(head)
        if m:
            return f"{fallback}_{m.group(1)}"
    return fallback


def _guess_facility(text: str) -> str | None:
    head = " ".join(text[:500].split())
    needles = [
        ("FAIRLAWN DERMATOLOGY", "fairlawn_derm"),
        ("EXCELLENCE IN EYECARE", "excellence_eyecare"),
        ("Luketic Eye Center", "luketic_eye"),
        ("BHARAT J SHAH", "bharat_shah"),
        ("Cleveland Clinic", "cleveland_clinic"),
        ("Practice Fusion", "practice_fusion"),
        ("Discharge Summary", "discharge_summary"),
    ]
    for needle, slug in needles:
        if needle.lower() in head.lower():
            return slug
    return None


def split_pages(raw: str) -> list[tuple[str, str]]:
    """Return list of (page_stem, ocr_text)."""
    parts = PAGE_SPLIT.split(raw)
    if len(parts) < 3:
        body = _clean_ocr(raw)
        return [("1", body)] if body else []
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        stem = parts[i]
        body = _clean_ocr(parts[i + 1] if i + 1 < len(parts) else "")
        if body:
            out.append((stem, body))
    return out


def extract_paste_text(path: Path) -> str:
    """Handle raw OCR paste or transcript-wrapped JSON blob."""
    raw = path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("{") and "<user_query>" in raw:
        try:
            obj = json.loads(raw)
            # Cursor transcript message shape
            content = obj.get("content")
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text") or "")
                raw = "\n".join(texts)
        except json.JSONDecodeError:
            pass
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", raw, re.S)
    if m:
        raw = m.group(1)
    return raw


def parse_keep_paste(path: Path) -> list[dict]:
    text = extract_paste_text(path)
    # Split into numbered packets: "1) ===== 1.jpg ====="
    chunks = PACKET_SPLIT.split(text)
    # If no packet markers, treat whole as one packet
    if len(chunks) <= 1:
        packets = [("paste_packet", text)]
    else:
        # re.split with capturing may leave preamble; rebuild with markers
        packets = []
        # Find starts
        starts = list(PACKET_SPLIT.finditer(text))
        if not starts:
            packets = [("paste_packet", text)]
        else:
            for i, m in enumerate(starts):
                end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
                chunk = text[m.start() : end]
                packets.append((f"paste_packet_{i+1}", chunk))

    records: list[dict] = []
    for pkt_name, chunk in packets:
        pages = split_pages(chunk)
        if not pages:
            continue
        facility = _guess_facility(pages[0][1]) or "paste_clinical"
        doc_key = _guess_doc_key(pages[0][1], facility)
        for stem, ocr in pages:
            records.append(
                _keep_record(
                    page_id=f"v03_{doc_key}_p{int(stem):02d}",
                    ocr=ocr,
                    source_file=path.name,
                    document_id=doc_key,
                    batch_id="keep_paste_v0.3",
                    facility=facility,
                    template_group=doc_key,
                    split_group=doc_key,
                )
            )
    return records


def parse_keep_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pages = split_pages(text)
    records: list[dict] = []
    # Group consecutive pages; new doc when page stem resets to 1 (after first)
    groups: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for stem, ocr in pages:
        if cur and stem == "1":
            groups.append(cur)
            cur = []
        cur.append((stem, ocr))
    if cur:
        groups.append(cur)

    for gi, group in enumerate(groups, start=1):
        facility = _guess_facility(group[0][1]) or path.stem.lower()
        doc_key = _guess_doc_key(group[0][1], f"{path.stem}_d{gi:02d}")
        for stem, ocr in group:
            records.append(
                _keep_record(
                    page_id=f"v03_{doc_key}_p{int(stem):02d}",
                    ocr=ocr,
                    source_file=path.name,
                    document_id=doc_key,
                    batch_id=path.stem,
                    facility=facility,
                    template_group=doc_key,
                    split_group=doc_key,
                )
            )
    return records


def _keep_record(
    *,
    page_id: str,
    ocr: str,
    source_file: str,
    document_id: str,
    batch_id: str,
    facility: str | None,
    template_group: str,
    split_group: str,
) -> dict:
    text_hash = _sha1_norm(ocr)
    n = len(ocr)
    # Short signature-only pages stay KEEP but skip training
    train_eligible = n >= 80
    return {
        "page_id": page_id,
        "ocr_text": ocr,
        "label": {
            "primary_class": "KEEP",
            "keep_delete": "KEEP",
            "flag": "KEEP",
            "confidence": "HIGH",
            "secondary_type": "clinical_ocr_human_labeled",
        },
        "annotation": {
            "evidence_from_ocr": ocr[:160].replace("\n", " "),
            "reason_for_label": (
                "Human-labeled KEEP: clinical note / progress note / encounter OCR. "
                "Flagging only — do not delete."
            ),
            "hard_negative": False,
            "hard_negative_type": [],
            "ambiguous": False,
            "review_required": False,
            "candidate_classes": ["KEEP"],
            "label_source": "human_ocr_text_only",
            "label_version": "v0.3",
        },
        "meta": {
            "source_file": source_file,
            "document_id": document_id,
            "batch_id": batch_id,
            "facility": facility,
            "vendor": None,
            "template_group": template_group,
            "split_group": split_group,
            "text_hash": text_hash,
            "ocr_char_count": n,
        },
        "train_eligible": train_eligible,
    }


def upgrade_v02(obj: dict) -> dict:
    """Carry v0.2 rows forward; attach coarse flag."""
    from src.preprocessing.taxonomy import to_flag

    out = json.loads(json.dumps(obj))  # deep copy
    primary = out["label"]["primary_class"]
    flag = to_flag(primary)
    out["label"]["flag"] = flag
    out["annotation"] = dict(out.get("annotation") or {})
    out["annotation"]["label_version"] = "v0.3"
    # Prefix page_id to avoid collision with new KEEP ids
    pid = str(out["page_id"])
    if not pid.startswith("v02_"):
        out["page_id"] = f"v02_{pid}"
    # Bootstrap: admit sparse BLANK pages into the train pool for 3-class learning
    if flag == "BLANK" and (out.get("ocr_text") or "").strip():
        out["train_eligible"] = True
        out["annotation"]["ambiguous"] = False
        out["annotation"]["review_required"] = False
        out["label"]["confidence"] = out["label"].get("confidence") or "MEDIUM"
    return out


def main() -> int:
    v02_path = ROOT / "data/raw/ehr_pages_v0.2.jsonl"
    keep_dir = ROOT / "data/raw/keep_sources_v0.3"
    paste_path = ROOT / "data/raw/keep_paste_v0.3.txt"
    out_path = ROOT / "data/raw/ehr_pages_v0.3.jsonl"
    label_map_path = ROOT / "data/labels/label_map_v0.3.json"

    keep_dir.mkdir(parents=True, exist_ok=True)

    # Copy attachments into keep_sources if present
    attach = Path(
        "/Users/sumit/.cursor/projects/Users-sumit-Algodel-tesseract/attachments/"
        "b6599b97-a36f-46e8-bb96-49eef332783c"
    )
    if attach.exists():
        for src in sorted(attach.glob("Untitled__*.txt")):
            dest = keep_dir / src.name
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                dest.write_bytes(src.read_bytes())

    records: list[dict] = []

    # 1) v0.2 blank/junk/review (+ thin KEEP)
    with v02_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(upgrade_v02(json.loads(line)))

    # 2) KEEP from paste + attachment dumps
    keep_records: list[dict] = []
    if paste_path.exists():
        keep_records.extend(parse_keep_paste(paste_path))
    for path in sorted(keep_dir.glob("*.txt")):
        keep_records.extend(parse_keep_file(path))

    # Dedup by text_hash (prefer KEEP over older if conflict — shouldn't)
    seen: dict[str, dict] = {}
    order: list[str] = []
    for r in records + keep_records:
        h = (r.get("meta") or {}).get("text_hash") or _sha1_norm(r.get("ocr_text", ""))
        if h in seen and h != "da39a3ee5e6b":  # allow multiple empty OCR
            # Prefer KEEP / newer label_version
            old = seen[h]
            old_flag = (old.get("label") or {}).get("flag")
            new_flag = (r.get("label") or {}).get("flag")
            if old_flag != "KEEP" and new_flag == "KEEP":
                seen[h] = r
            continue
        if h not in seen:
            order.append(h)
        seen[h] = r

    # Rebuild preserving order of first-seen, but empty OCR pages all kept from v02
    final: list[dict] = []
    used_nonempty: set[str] = set()
    empties: list[dict] = []
    for r in records + keep_records:
        h = (r.get("meta") or {}).get("text_hash") or _sha1_norm(r.get("ocr_text", ""))
        if not (r.get("ocr_text") or "").strip():
            empties.append(r)
            continue
        if h in used_nonempty:
            continue
        # take canonical from seen
        final.append(seen[h])
        used_nonempty.add(h)
    final.extend(empties)

    # Human flag corrections (e.g. printer-footer pages mis-labeled KEEP)
    corr_path = ROOT / "data/labels/label_corrections_v0.3.json"
    if corr_path.exists():
        corr = json.loads(corr_path.read_text(encoding="utf-8"))
        by_id = corr.get("by_page_id") or {}
        n_corr = 0
        for r in final:
            flag = by_id.get(r["page_id"])
            if not flag:
                continue
            r["label"]["primary_class"] = flag
            r["label"]["flag"] = flag
            r["label"]["keep_delete"] = (
                "DELETE" if flag in {"BLANK", "JUNK"} else "KEEP"
            )
            r["label"]["confidence"] = "HIGH"
            r["annotation"]["label_version"] = "v0.3.1"
            r["train_eligible"] = bool((r.get("ocr_text") or "").strip())
            n_corr += 1
        print(f"Applied {n_corr} label corrections from {corr_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    flags = Counter((r.get("label") or {}).get("flag") for r in final)
    kd = Counter((r.get("label") or {}).get("keep_delete") for r in final)
    primary = Counter((r.get("label") or {}).get("primary_class") for r in final)
    keep_new = sum(
        1
        for r in final
        if (r.get("annotation") or {}).get("label_version") == "v0.3"
        and (r.get("label") or {}).get("flag") == "KEEP"
        and str(r["page_id"]).startswith("v03_")
    )

    label_map = {
        "version": "v0.3",
        "flags": ["KEEP", "BLANK", "JUNK"],
        "note": (
            "Training/inference use coarse flags only. Fine classes from v0.2 are "
            "retained in primary_class and mapped via taxonomy.to_flag()."
        ),
        "flag_by_primary_class": {
            "KEEP": "KEEP",
            "RETAIN_CLINICAL": "KEEP",
            "RETAIN_DEMOGRAPHIC": "KEEP",
            "RETAIN_CLINICAL_IMAGE": "KEEP",
            "BLANK": "BLANK",
            "BLANK_ABSOLUTE": "BLANK",
            "BLANK_TECHNICAL": "BLANK",
            "BLANK_SYSTEM": "BLANK",
            "BLANK_HEADER_FOOTER_ONLY": "BLANK",
            "JUNK": "JUNK",
            "JUNK_COVER_REQUEST": "JUNK",
            "JUNK_FAX_TRANSMISSION": "JUNK",
            "JUNK_SEPARATOR_BARCODE": "JUNK",
            "JUNK_PRINTER_SYSTEM_TEST": "JUNK",
            "JUNK_POSTAL_MAIL": "JUNK",
            "JUNK_INSURANCE_ID": "JUNK",
            "JUNK_BLACK_SCAN_DEFECT": "JUNK",
            "JUNK_NON_CLINICAL_PHOTO": "JUNK",
            "UNCERTAIN_REVIEW": None,
        },
        "counts": {
            "total_pages": len(final),
            "by_flag": dict(flags),
            "by_keep_delete": dict(kd),
            "new_keep_pages": keep_new,
            "by_primary_class": dict(primary),
        },
    }
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    print(json.dumps(label_map["counts"], indent=2))
    print(f"Wrote {out_path} ({len(final)} rows)")
    print(f"Wrote {label_map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
