"""Safety-first evaluation for KEEP / BLANK / JUNK flagging."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.preprocessing.taxonomy import FLAGS, to_flag


def false_flag_report(
    y_true: list[str],
    y_pred_flag: list[str],
) -> dict[str, Any]:
    """Primary safety metric: predicted BLANK/JUNK when truth was KEEP."""
    total_keep = 0
    false_flags = 0
    by_pred: dict[str, int] = {"BLANK": 0, "JUNK": 0}
    for truth, pred in zip(y_true, y_pred_flag):
        t = to_flag(truth) if truth not in FLAGS else truth  # type: ignore[arg-type]
        if t != "KEEP":
            continue
        total_keep += 1
        if pred in {"BLANK", "JUNK"}:
            false_flags += 1
            by_pred[pred] = by_pred.get(pred, 0) + 1
    rate = (false_flags / total_keep) if total_keep else None
    return {
        "keep_pages": total_keep,
        "false_blank_or_junk": false_flags,
        "false_flag_rate": rate,
        "by_predicted_flag": by_pred,
        "note": (
            "Cannot estimate false-flag rate reliably until KEEP pages "
            "exist in the evaluation split."
            if total_keep == 0
            else None
        ),
    }


def missed_drop_report(
    y_true: list[str],
    y_pred_flag: list[str],
) -> dict[str, Any]:
    """BLANK/JUNK truth that was flagged KEEP (safe miss — not a deletion)."""
    drop_true = 0
    kept = 0
    for truth, pred in zip(y_true, y_pred_flag):
        t = to_flag(truth) if truth not in FLAGS else truth  # type: ignore[arg-type]
        if t not in {"BLANK", "JUNK"}:
            continue
        drop_true += 1
        if pred == "KEEP":
            kept += 1
    return {
        "blank_junk_truth_pages": drop_true,
        "incorrectly_kept": kept,
        "missed_drop_rate": (kept / drop_true) if drop_true else None,
    }


def _as_flag(label: str) -> str:
    if label in FLAGS:
        return label
    mapped = to_flag(label)
    return mapped if mapped is not None else "KEEP"


def evaluate_predictions(
    y_true: list[str],
    y_pred: list[str],
    y_pred_flag: list[str] | None = None,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate flag predictions. ``y_true`` / ``y_pred`` should be KEEP|BLANK|JUNK."""
    y_true_f = [_as_flag(t) for t in y_true]
    y_pred_f = [_as_flag(p) for p in (y_pred_flag if y_pred_flag is not None else y_pred)]
    labels = labels or sorted(set(y_true_f) | set(y_pred_f))
    report = classification_report(
        y_true_f, y_pred_f, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels).tolist()
    return {
        "n": len(y_true_f),
        "accuracy": float(accuracy_score(y_true_f, y_pred_f)) if y_true_f else None,
        "macro_f1": float(f1_score(y_true_f, y_pred_f, average="macro", zero_division=0))
        if y_true_f
        else None,
        "weighted_f1": float(
            f1_score(y_true_f, y_pred_f, average="weighted", zero_division=0)
        )
        if y_true_f
        else None,
        "per_class": {c: report[c] for c in labels if c in report},
        "confusion_matrix": {"labels": labels, "matrix": cm},
        "false_flag": false_flag_report(y_true_f, y_pred_f),
        "missed_drop": missed_drop_report(y_true_f, y_pred_f),
        "flag_distribution": dict(Counter(y_pred_f)),
        "truth_distribution": dict(Counter(y_true_f)),
    }


def write_report(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
