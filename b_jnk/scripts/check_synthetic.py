#!/usr/bin/env python3
"""Check Claude-generated synthetic batches and build the clean training file.

  python scripts/check_synthetic.py synthetic/raw/KEEP-004.jsonl   # check files
  python scripts/check_synthetic.py --write                        # all of synthetic/raw -> checked/clean.jsonl

The batch id comes from the file name (e.g. KEEP-001.jsonl). Rows that break a
rule are reported and left out of clean.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.page_subclass import (  # noqa: E402
    CLINICAL_SECTION_RE,
    SHORT_PAGE_MAX_WORDS,
    SIGNATURE_RE,
    TRIGGERS,
    assign_blank_subtype,
    assign_junk_subtype,
    scan,
)

RAW_DIR = ROOT / "synthetic" / "raw"
CLEAN_PATH = ROOT / "synthetic" / "checked" / "clean.jsonl"

PREFIX_LABEL = {
    "KEEP": "KEEP", "BLNK": "BLANK", "INV": "JUNK_INVOICE", "COV": "JUNK_COVER_PAGE",
    "RREQ": "JUNK_RECORD_REQUEST", "INST": "JUNK_INSTRUCTIONS", "FAX": "JUNK_LETTER_FAX",
    "OTH": "JUNK_OTHERS",
}
KINDS = {
    "KEEP": ["office visit/progress note", "H&P", "discharge summary", "ED note", "consult note",
             "operative/procedure note", "telephone encounter", "nursing note/flowsheet", "medication list",
             "medication reconciliation", "prescription/refill history", "MAR", "allergy list", "problem list",
             "immunization record", "lab results", "pathology report", "radiology report",
             "ECG/echo/stress test", "ophthalmology exam", "dermatology exam with lesion photo captions",
             "wound care note with photo captions", "endoscopy/colonoscopy report with images",
             "retinal scan report", "PT/OT/behavioral therapy note", "care plan", "after-visit summary",
             "provider referral/consult letter", "face sheet", "registration sheet", "demographics page",
             "patient portal registration form", "signature/attestation page",
             "short clinical continuation page", "chart table of contents", "EMR contents/index page",
             "clinical section divider page"],
    "BLNK": ["absolute", "technical", "system", "header_footer"],
    "INV": ["invoice", "superbill", "billing statement", "statement of account", "remittance advice",
            "patient balance statement", "weak-phrases-only invoice", "insurance card copy",
            "driver's license copy", "SSN card copy"],
    "COV": ["accept sheet", "unaccept sheet", "accept with code", "cover page title sheet",
            "cover page with batch/packet id"],
    "RREQ": ["vendor request letter", "HEDIS chart request", "risk adjustment request", "e-request letter",
             "pull list", "records transmittal sheet", "records-attached release note", "audit fulfillment page",
             "partial fulfillment letter", "request acknowledgement form", "payer letter to provider"],
    "INST": ['HEDIS measure "what to send" tables', "submission/upload instructions", "portal instructions",
             "documents-to-include checklists", "record copy guidelines"],
    "FAX": ["fax cover sheet", "facsimile transmittal sheet", "fax transmission report/confirmation log",
            "generic cover letter", "confidentiality notice page"],
    "OTH": ["short admin crumb", "gibberish OCR page", "barcode/patch separator", "scanner calibration target",
            "printer test/config page", "postal receipt", "shipping label", "courier envelope",
            "black/inverted scan remnant", "non-clinical photo caption", "portal upload receipt"],
}
ID_CARD_KINDS = {"insurance card copy", "driver's license copy", "SSN card copy"}
BLANK_KIND = {
    "absolute": "BLANK_ABSOLUTE", "technical": "BLANK_TECHNICAL", "system": "BLANK_SYSTEM",
    "header_footer": "BLANK_HEADER_FOOTER_ONLY",
}
CLINICAL_HEADINGS_RE = re.compile(
    r"(?<![a-z])(medications|allergies|vital signs|chief complaint|hpi|history of present illness|"
    r"review of systems|physical exam|assessment|diagnosis|plan of care|plan of treatment|problem list|"
    r"active problems|immunizations|progress note|reason for visit|impression|subjective|objective)(?![a-z])",
    re.I,
)
LONG_JUNK = {"JUNK_INVOICE", "JUNK_RECORD_REQUEST", "JUNK_INSTRUCTIONS", "JUNK_LETTER_FAX"}


def norm_kind(kind: str) -> str:
    return str(kind or "").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')


def focus_kinds(prefix: str, n: int) -> list[str]:
    kinds = KINDS[prefix]
    if prefix == "BLNK":
        return kinds
    return [kinds[((n - 1) * 4 + j) % len(kinds)] for j in range(4)]


def check_row(r: dict, prefix: str, label: str, focus: list[str]) -> list[str]:
    problems: list[str] = []
    text = r.get("text", "")
    if not isinstance(text, str):
        return ["text is not a string"]
    wc = len(text.split())
    groups, found = scan(text)
    kind = norm_kind(r.get("page_kind"))

    if r.get("label") != label:
        problems.append(f"label {r.get('label')!r} (batch is {label})")
    want_flag = "KEEP" if label == "KEEP" else "BLANK" if label == "BLANK" else "JUNK"
    if r.get("flag") != want_flag:
        problems.append(f"flag {r.get('flag')!r}")
    if kind not in focus:
        problems.append(f"page_kind {r.get('page_kind')!r} not in this batch's focus kinds")
    if r.get("word_count") != wc:
        problems.append(f"word_count says {r.get('word_count')} real {wc}")
    claimed = set(r.get("triggers_found") or [])
    if claimed != found:
        problems.append(f"triggers_found claimed {sorted(claimed)} real {sorted(found)}")
    if r.get("noise") not in {"clean", "light", "heavy"}:
        problems.append(f"noise {r.get('noise')!r}")
    lang = r.get("language")
    if lang not in {"en", "es"}:
        problems.append(f"language {lang!r}")
    elif lang == "es" and label not in {"KEEP", "BLANK", "JUNK_OTHERS"}:
        problems.append("Spanish not allowed for this label")
    if wc > 600:
        problems.append(f"{wc} words > 600")

    if label == "KEEP":
        if found and not r.get("hard_negative"):
            problems.append("has junk trigger but hard_negative=false")
        return problems

    if label == "BLANK":
        want = assign_blank_subtype(text).subclass
        if BLANK_KIND.get(kind) != want:
            problems.append(f"page_kind {kind} but text is {want}")
        if found:
            problems.append(f"BLANK has triggers {sorted(found)}")
        heads = sorted({m.group(1).lower() for m in CLINICAL_HEADINGS_RE.finditer(text)})
        if heads:
            problems.append(f"BLANK has clinical headings {heads}")
        if kind == "technical":
            toks = text.split()
            if not 1 <= len(toks) <= 5 or any(len(t) > 1 and re.search(r"[A-Za-z0-9]", t) for t in toks):
                problems.append("technical must be 1-5 single-char/symbol tokens")
        return problems

    if label == "JUNK_INVOICE" and kind in ID_CARD_KINDS:
        if found:
            problems.append(f"ID card copy has triggers {sorted(found)}")
        got = assign_junk_subtype(text, model_tag="JUNK_INSURANCE_ID").subclass
        if got != "JUNK_INVOICE":
            problems.append(f"ID card copy resolves to {got}")
    else:
        got = assign_junk_subtype(text).subclass
        if got != label:
            problems.append(f"rules assign {got}, not {label}")
        others = [g for g in TRIGGERS if g != label and groups[g]]
        if label == "JUNK_COVER_PAGE" or wc > SHORT_PAGE_MAX_WORDS:
            others = [o for o in others if o != "JUNK_COVER_PAGE"]
        if label != "JUNK_INVOICE" and groups["strong"]:
            others.append(f"invoice {groups['strong']}")
        if label != "JUNK_INVOICE" and len(groups["weak"]) >= 2:
            others.append(f"2+ weak invoice {groups['weak']}")
        if others:
            problems.append(f"other-subtype triggers: {others}")
    if label in LONG_JUNK and wc < 20:
        problems.append(f"only {wc} words (needs 20+)")
    if label == "JUNK_COVER_PAGE" and wc > SHORT_PAGE_MAX_WORDS:
        problems.append(f"{wc} words (cover page must be <=19)")
    if label == "JUNK_OTHERS" and wc <= SHORT_PAGE_MAX_WORDS:
        if groups["JUNK_COVER_PAGE"]:
            problems.append("short Others has accept/unaccept/cover page")
        if CLINICAL_SECTION_RE.search(text):
            problems.append("short Others names a clinical section")
        if SIGNATURE_RE.search(text):
            problems.append("short Others looks like a signature page")
    if "table of contents" in text.lower():
        problems.append("JUNK page says 'table of contents'")
    return problems


def check_file(path: Path, *, verbose: bool = True) -> tuple[list[dict], int]:
    """Return (rows that passed, problem count)."""
    batch = path.stem
    m = re.fullmatch(r"([A-Z]+)-(\d{3})", batch)
    if not m or m.group(1) not in PREFIX_LABEL:
        print(f"{path.name}: cannot read batch id from file name")
        return [], 1
    prefix, n = m.group(1), int(m.group(2))
    label = PREFIX_LABEL[prefix]
    focus = focus_kinds(prefix, n)

    rows, bad = [], 0
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  {path.name} line {ln}: invalid JSON ({e})")
            bad += 1

    out = [f"== {path.name}  label={label}  rows={len(rows)}"]
    if len(rows) != 25:
        out.append(f"  expected 25 rows, got {len(rows)}")
        bad += 1
    if [r.get("id") for r in rows] != [f"{batch}-{i:04d}" for i in range(1, len(rows) + 1)]:
        out.append(f"  ids not {batch}-0001..{len(rows):04d} in order")
        bad += 1
    openings = collections.Counter(" ".join(r.get("text", "").split()[:8]) for r in rows)
    for opening, count in openings.items():
        if count > 1 and len(opening.split()) >= 8:
            out.append(f"  {count} rows share opening: {opening!r}")
            bad += 1

    passed = []
    for r in rows:
        problems = check_row(r, prefix, label, focus)
        if problems:
            bad += 1
            out.append(f"  {r.get('id')}: " + " | ".join(problems))
        else:
            passed.append(r)

    kinds = collections.Counter(norm_kind(r.get("page_kind")) for r in rows)
    missing = [k for k in focus if k not in kinds]
    lens = sorted(len(str(r.get("text", "")).split()) for r in rows) or [0]
    out.append(f"  kinds {dict(kinds)}" + (f"  MISSING {missing}" if missing else ""))
    out.append(
        f"  noise {dict(collections.Counter(r.get('noise') for r in rows))}  "
        f"language {dict(collections.Counter(r.get('language') for r in rows))}  "
        f"words min/median/max {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}"
    )
    if label == "KEEP":
        out.append(f"  hard negatives {sum(bool(r.get('hard_negative')) for r in rows)}/{len(rows)}")
    out.append("  OK" if bad == 0 else f"  {bad} problem(s)")
    if verbose or bad:
        print("\n".join(out))
    return passed, bad


def write_clean(files: list[Path]) -> int:
    kept: list[dict] = []
    total_bad = 0
    for path in files:
        passed, bad = check_file(path, verbose=False)
        total_bad += bad
        kept.extend(passed)

    by_text: dict[str, set[str]] = collections.defaultdict(set)
    for r in kept:
        by_text[" ".join(r["text"].lower().split())].add(r["label"])
    conflicts = {t for t, labels in by_text.items() if len(labels) > 1}

    seen: set[tuple[str, str]] = set()
    clean: list[dict] = []
    dropped_dupes = dropped_conflicts = 0
    for r in kept:
        key_text = " ".join(r["text"].lower().split())
        if key_text in conflicts:
            dropped_conflicts += 1
            continue
        key = (key_text, r["label"])
        if key in seen:
            dropped_dupes += 1
            continue
        seen.add(key)
        clean.append(
            {k: r[k] for k in ("id", "label", "flag", "page_kind", "text", "hard_negative", "noise", "language")}
        )

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_PATH.open("w", encoding="utf-8") as fh:
        for r in clean:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"\nChecked {len(files)} files: {len(kept)} rows passed, {total_bad} problem(s).\n"
        f"Dropped {dropped_dupes} duplicate text(s), {dropped_conflicts} cross-label conflict(s).\n"
        f"Wrote {len(clean)} rows -> {CLEAN_PATH}\n"
        f"Labels: {dict(collections.Counter(r['label'] for r in clean))}"
    )
    return 1 if total_bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--write", action="store_true", help=f"check all files and write {CLEAN_PATH.name}")
    args = ap.parse_args()
    files = args.files or sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No batch files given and none in {RAW_DIR}")
    if args.write:
        return write_clean(files)
    return 1 if sum(check_file(p)[1] for p in files) else 0


if __name__ == "__main__":
    raise SystemExit(main())
