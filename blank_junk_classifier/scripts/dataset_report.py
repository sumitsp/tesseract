#!/usr/bin/env python3
"""Generate dataset quality report JSON (no training)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.dataset import filter_training_rows, load_jsonl  # noqa: E402
from src.preprocessing.taxonomy import ALL_CLASSES, NOT_LEARNABLE_FROM_TEXT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(ROOT / "data/raw/ehr_pages_v0.2.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports/dataset_quality_v0.2.json"))
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    lens = [len(r.ocr_text or "") for r in rows]
    train = filter_training_rows(rows, require_train_eligible=True, include_medium_keep=True)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_hash[r.text_hash or ""].append(r.page_id)
    dups = {h: ids for h, ids in by_hash.items() if h and len(ids) > 1}

    class_counts = Counter(r.primary_class for r in rows)
    missing = [c for c in ALL_CLASSES if class_counts[c] == 0]

    report = {
        "total_pages": len(rows),
        "flags": dict(Counter(r.flag for r in rows if r.flag)),
        "legacy_keep_delete": dict(Counter(r.keep_delete for r in rows)),
        "pages_per_fine_class": dict(class_counts),
        "missing_fine_classes": missing,
        "train_eligible_after_filter": len(train),
        "train_flags": dict(Counter(r.flag for r in train if r.flag)),
        "train_fine_classes": dict(Counter(r.primary_class for r in train)),
        "empty_ocr": sum(1 for n in lens if n == 0),
        "short_ocr_lt_50": sum(1 for n in lens if n < 50),
        "ocr_char_stats": {
            "min": min(lens),
            "median": statistics.median(lens),
            "p90": sorted(lens)[int(0.9 * (len(lens) - 1))],
            "max": max(lens),
        },
        "duplicate_groups": len(dups),
        "duplicate_pages": sum(len(v) for v in dups.values()),
        "hard_negatives": sum(1 for r in rows if r.hard_negative),
        "hard_negative_types": dict(
            Counter(t for r in rows for t in r.hard_negative_type)
        ),
        "ambiguous_pages": sum(1 for r in rows if r.ambiguous),
        "not_learnable_from_text_present": {
            c: class_counts[c] for c in NOT_LEARNABLE_FROM_TEXT
        },
        "nonblank_junk_ge_300": sum(
            1
            for r in rows
            if r.flag == "JUNK"
            and len(r.ocr_text) >= 300
        ),
        "template_groups": dict(Counter(r.template_group for r in rows)),
        "sufficient_for_production": False,
        "blocking_gaps": [
            "Only 2 BLANK pages — BLANK flag is undertrained",
            "JUNK still mostly cover/fax templates from v0.2",
            "Some KEEP packets are long single-patient dumps (capped per split_group)",
        ],
        "recommendation": (
            "v0.3 has substantial KEEP clinical OCR. Output is flagging CSV "
            "(KEEP/BLANK/JUNK) — nothing is deleted. Collect more BLANK and "
            "diverse JUNK before trusting those flags in production."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
