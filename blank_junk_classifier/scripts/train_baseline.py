#!/usr/bin/env python3
"""Train TF-IDF baseline on KEEP / BLANK / JUNK flags; evaluate safety metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_predictions, write_report  # noqa: E402
from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.classifiers import train_hierarchical, train_tfidf_flat  # noqa: E402
from src.models.decision import DecisionConfig, decide_from_proba  # noqa: E402
from src.models.split import describe_split, group_train_test_split  # noqa: E402
from src.preprocessing.dataset import filter_training_rows, load_jsonl  # noqa: E402


def _load_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError as exc:
        raise SystemExit(
            "Install PyYAML or pass --config configs/default.json"
        ) from exc


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
    print(f"Loaded {len(rows)} pages; train pool {len(train_pool)}")
    if len(train_pool) < 4:
        print("ERROR: train pool too small. Collect more labeled data.")
        return 2

    train, test = group_train_test_split(
        train_pool,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        max_pages_per_template_train=cfg["split"]["max_pages_per_template_train"],
    )
    split_info = describe_split(train, test)
    print("Split:", json.dumps(split_info, indent=2))
    if split_info["overlap_groups"]:
        print("ERROR: template leakage across split", split_info["overlap_groups"])
        return 3

    tfidf_cfg = cfg["tfidf"]
    model = train_tfidf_flat(
        train,
        classifier=tfidf_cfg["classifier"],
        retain_cost=cfg["training"]["retain_delete_cost_ratio"],
        word_ngram_range=tuple(tfidf_cfg["word_ngram_range"]),
        char_ngram_range=tuple(tfidf_cfg["char_ngram_range"]),
        word_max_features=tfidf_cfg["word_max_features"],
        char_max_features=tfidf_cfg["char_max_features"],
        min_df=tfidf_cfg["min_df"],
    )
    models_dir = ROOT / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "tfidf_flat.joblib"
    model.save(str(model_path))
    print(f"Saved {model_path}")

    decision = DecisionConfig(**cfg["decision"])
    service = PageClassifierService(
        model,
        model_version=cfg["model_version"] + "+tfidf",
        decision=decision,
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
    )

    eval_rows = test if test else train
    y_true, y_pred = [], []
    for r in eval_rows:
        if r.flag is None:
            continue
        out = service.predict_one(r.page_id, r.ocr_text)
        y_true.append(r.flag)
        y_pred.append(out.flag)

    metrics = evaluate_predictions(y_true, y_pred)
    metrics["model"] = model.name
    metrics["split"] = split_info
    metrics["dataset_warning"] = (
        "v0.3 KEEP coverage is much better; BLANK is still tiny (n≈2). "
        "Treat BLANK predictions as provisional."
    )
    report_path = ROOT / cfg["paths"]["reports_dir"] / "tfidf_flat_eval.json"
    write_report(report_path, metrics)
    print(json.dumps(metrics, indent=2))

    try:
        hier = train_hierarchical(
            train, retain_cost=cfg["training"]["retain_delete_cost_ratio"]
        )
        hier_path = models_dir / "tfidf_hierarchical_meta.json"
        hier_path.write_text(
            json.dumps(
                {
                    "name": hier.name,
                    "labels": hier.labels,
                    "note": "Coarse KEEP/DROP then BLANK/JUNK",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        y_true_h, y_pred_h = [], []
        for r in eval_rows:
            if r.flag is None:
                continue
            route = service.predict_one(r.page_id, r.ocr_text)
            if route.decision_reason.startswith("pre_model_route"):
                y_true_h.append(r.flag)
                y_pred_h.append(route.flag)
                continue
            _best, mat = hier.predict_proba_page_types([r.ocr_text])
            d = decide_from_proba(mat[0], hier.labels, config=decision)
            y_true_h.append(r.flag)
            y_pred_h.append(d.flag)
        h_metrics = evaluate_predictions(y_true_h, y_pred_h)
        h_metrics["model"] = hier.name
        write_report(
            ROOT / cfg["paths"]["reports_dir"] / "tfidf_hierarchical_eval.json",
            h_metrics,
        )
        print("Hierarchical:", json.dumps(h_metrics.get("false_flag"), indent=2))
    except ValueError as exc:
        print(f"Hierarchical skipped: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
