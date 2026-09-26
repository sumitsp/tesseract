#!/usr/bin/env python3
"""Train embedding + logistic regression classifier and evaluate safety metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError as exc:
        raise SystemExit("Install PyYAML or pass --config configs/default.json") from exc


from src.evaluation.metrics import evaluate_predictions, write_report  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.classifiers import train_embedding_flat  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402
from src.models.split import describe_split, group_train_test_split  # noqa: E402
from src.preprocessing.dataset import filter_training_rows, load_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    args = ap.parse_args()
    cfg = _load_cfg(Path(args.config))

    rows = load_jsonl(ROOT / cfg["paths"]["raw_jsonl"])
    train_pool = filter_training_rows(
        rows,
        require_train_eligible=cfg["training"]["require_train_eligible"],
        include_medium_keep=cfg["training"]["include_medium_keep"],
    )
    train, test = group_train_test_split(
        train_pool,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        max_pages_per_template_train=cfg["split"]["max_pages_per_template_train"],
    )
    print("Split:", json.dumps(describe_split(train, test), indent=2))

    try:
        model = train_embedding_flat(
            train,
            model_name=cfg["embeddings"]["model_name"],
            retain_cost=cfg["training"]["retain_delete_cost_ratio"],
        )
    except ImportError as exc:
        print(exc)
        return 2

    out = ROOT / cfg["paths"]["models_dir"] / "embedding_flat.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved {out}")

    service = PageClassifierService(
        model,
        model_version=cfg["model_version"] + "+embed",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
    )
    eval_rows = test if test else train
    y_true, y_pred = [], []
    for r in eval_rows:
        if r.flag is None:
            continue
        p = service.predict_one(r.page_id, r.ocr_text)
        y_true.append(r.flag)
        y_pred.append(p.flag)
    metrics = evaluate_predictions(y_true, y_pred)
    metrics["model"] = model.name
    metrics["dataset_warning"] = (
        "Embedding comparison is exploratory on v0.2; KEEP coverage is inadequate."
    )
    report = ROOT / cfg["paths"]["reports_dir"] / "embedding_flat_eval.json"
    write_report(report, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
