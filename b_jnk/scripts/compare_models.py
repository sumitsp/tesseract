#!/usr/bin/env python3
"""Compare baseline vs embedding eval reports on safety metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=str(ROOT / "reports/tfidf_flat_eval.json"),
    )
    ap.add_argument(
        "--embedding",
        default=str(ROOT / "reports/embedding_flat_eval.json"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "reports/model_comparison.json"),
    )
    args = ap.parse_args()

    base = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).exists() else None
    emb = json.loads(Path(args.embedding).read_text()) if Path(args.embedding).exists() else None

    def slice_metrics(m: dict | None) -> dict | None:
        if not m:
            return None
        return {
            "model": m.get("model"),
            "n": m.get("n"),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "false_flag": m.get("false_flag"),
            "missed_drop": m.get("missed_drop"),
            "flag_distribution": m.get("flag_distribution"),
            "dataset_warning": m.get("dataset_warning"),
        }

    comparison = {
        "selection_rule": (
            "Choose the model with the fewest KEEP→BLANK/JUNK false flags at a "
            "fixed review budget; break ties on correct BLANK/JUNK rate. Do not "
            "select on accuracy alone."
        ),
        "baseline": slice_metrics(base),
        "embedding": slice_metrics(emb),
        "production_ready": False,
        "recommendation": (
            "Neither model is production-ready on v0.2. Prefer TF-IDF baseline as "
            "the default scaffolding (fast, interpretable evidence). Re-run after "
            "collecting more KEEP pages. Output is flagging CSV only — nothing is deleted."
        ),
    }
    Path(args.out).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
