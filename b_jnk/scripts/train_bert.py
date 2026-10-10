#!/usr/bin/env python3
"""Train the 8-label BERT page classifier and test it on real pages only.

  .venv/bin/python scripts/train_bert.py               # evaluate, then train + save final model
  .venv/bin/python scripts/train_bert.py --skip-eval   # train + save final model only
  .venv/bin/python scripts/train_bert.py --eval-only   # evaluate only, save nothing

Evaluation (every number comes from real pages the model did not train on):
  1. synthetic_only  - train on synthetic pages, test on ALL real pages
  2. real_2fold      - train on synthetic + half the real pages, test on the other half, then swap
The final model trains on synthetic + all real pages and is saved to models/bert_page.
Synthetic pages are never used for testing: BERT can learn Claude's writing style.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import PageClassifierService  # noqa: E402
from src.models.bert_classifier import (  # noqa: E402
    BERT_LABELS,
    BertPageClassifier,
    encode_head_tail,
    pick_device,
    save_bert,
)
from src.models.decision import DecisionConfig  # noqa: E402
from src.preprocessing.dataset import filter_training_rows, load_jsonl  # noqa: E402
from src.preprocessing.page_subclass import JUNK_SUBTYPES, assign_junk_subtype  # noqa: E402
from src.preprocessing.taxonomy import to_flag  # noqa: E402

LABEL_INDEX = {lab: i for i, lab in enumerate(BERT_LABELS)}
REAL_PACKET_BATCH_PREFIX = "blank_junk_samples"


@dataclass
class Example:
    uid: str
    text: str
    label: str
    source: str  # synthetic | real
    group: str = ""
    reason: str = ""
    fold: int = -1


def _load_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def flag_of(label: str) -> str:
    return to_flag(label) or "KEEP"


def synthetic_examples(cfg: dict) -> list[Example]:
    path = ROOT / cfg["bert"]["synthetic_clean"]
    if not path.exists():
        raise SystemExit(f"{path} missing - run: python scripts/check_synthetic.py --write")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["text"].strip():
            continue  # absolute blanks are decided from Docling structure before the model
        out.append(Example(r["id"], r["text"], r["label"], "synthetic", group="synthetic"))
    return out


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def annotated_examples(cfg: dict) -> list[Example]:
    """Image labels joined to Document Intelligence text. Missing file = none."""
    rel = cfg["bert"].get("annotated_jsonl") or "data/raw/annotated_pages.jsonl"
    path = ROOT / rel
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text") or "").strip()
        label = str(row.get("label") or "")
        if not text or label not in LABEL_INDEX:
            continue
        out.append(Example(
            str(row.get("id") or ""),
            text,
            label,
            "annotated",
            group=str(row.get("group") or row.get("id") or ""),
            reason="image_label",
        ))
    return out


def new_annotated(annotated: list[Example], already: list[Example]) -> tuple[list[Example], int, int]:
    """Drop a page whose text is already in training. A conflicting label is counted apart."""
    seen: dict[str, set[str]] = {}
    for example in already:
        seen.setdefault(_norm_text(example.text), set()).add(example.label)
    kept: list[Example] = []
    conflicts = 0
    duplicates = 0
    for example in annotated:
        labels = seen.get(_norm_text(example.text))
        if not labels:
            kept.append(example)
            seen.setdefault(_norm_text(example.text), set()).add(example.label)
            continue
        if example.label in labels:
            duplicates += 1
        else:
            conflicts += 1
    return kept, duplicates, conflicts


def real_examples(cfg: dict) -> list[Example]:
    path = ROOT / cfg["paths"]["raw_jsonl"]
    if not path.is_file():
        print(f"real pages file not on this machine: {path}", flush=True)
        return []
    rows = load_jsonl(path)
    pool = filter_training_rows(
        rows,
        require_train_eligible=cfg["training"]["require_train_eligible"],
        include_medium_keep=cfg["training"]["include_medium_keep"],
    )
    out = []
    for r in pool:
        flag = r.flag
        if flag == "KEEP":
            label, reason = "KEEP", "flag:KEEP"
        elif flag == "BLANK":
            label, reason = "BLANK", f"flag:BLANK ({r.primary_class})"
        elif r.primary_class in JUNK_SUBTYPES:
            label, reason = r.primary_class, "human_subtype"
        else:
            s = assign_junk_subtype(r.ocr_text, model_tag=r.primary_class)
            label, reason = s.subclass, s.reason
        packet = (r.batch_id or "").startswith(REAL_PACKET_BATCH_PREFIX)
        out.append(Example(r.page_id, r.ocr_text, label, "real", group=r.group_key, reason=reason,
                           fold=-2 if packet else -1))
    return out


def assign_folds(real: list[Example]) -> None:
    """Two folds. Chart documents stay whole (pages share headers); the blank/junk
    sample packets are collections of unrelated sheets, so they split page by page."""
    by_label: dict[str, list[Example]] = collections.defaultdict(list)
    groups: dict[str, list[Example]] = collections.defaultdict(list)
    for e in real:
        (by_label[e.label] if e.fold == -2 else groups[e.group]).append(e)
    for items in by_label.values():
        for i, e in enumerate(sorted(items, key=lambda x: x.uid)):
            e.fold = i % 2
    sizes = [sum(1 for e in real if e.fold == f) for f in (0, 1)]
    for g in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        f = 0 if sizes[0] <= sizes[1] else 1
        for e in groups[g]:
            e.fold = f
        sizes[f] += len(groups[g])


def write_real_labels(real: list[Example]) -> Path:
    path = ROOT / "reports" / "bert_real_labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_id", "fold", "label", "label_reason", "text_preview"])
        for e in sorted(real, key=lambda x: (x.label, x.uid)):
            w.writerow([e.uid, e.fold, e.label, e.reason, " ".join(e.text.split())[:200]])
    return path


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def length_batches(examples: list[Example], batch_size: int, rng: random.Random) -> list[list[Example]]:
    """Batches of similar length (less padding), in random order."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    batches = []
    chunk = batch_size * 50
    for start in range(0, len(idx), chunk):
        part = sorted(idx[start : start + chunk], key=lambda i: len(examples[i].text))
        batches.extend([examples[i] for i in part[j : j + batch_size]] for j in range(0, len(part), batch_size))
    rng.shuffle(batches)
    return batches


def train_model(
    train: list[Example],
    cfg: dict,
    device,
    *,
    epochs: int,
    base_model: str,
    tag: str,
    resume_dir: Path | None = None,
    learning_rate: float | None = None,
    keep_weight: float | None = None,
):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    bc = cfg["bert"]
    seed_all(bc["seed"])
    rng = random.Random(bc["seed"])
    if resume_dir is not None:
        print(f"[{tag}] continuing from saved model {resume_dir}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(resume_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(resume_dir)).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(BERT_LABELS),
            id2label=dict(enumerate(BERT_LABELS)),
            label2id=LABEL_INDEX,
        ).to(device)

    weights = torch.ones(len(BERT_LABELS))
    weights[LABEL_INDEX["KEEP"]] = float(bc["keep_weight"] if keep_weight is None else keep_weight)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    lr = float(bc["learning_rate"] if learning_rate is None else learning_rate)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(bc["weight_decay"]))
    steps_per_epoch = math.ceil(len(train) / bc["batch_size"])
    total = steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(optim, int(total * float(bc["warmup_ratio"])), total)

    counts = collections.Counter(e.label for e in train)
    print(f"[{tag}] training on {len(train)} pages {dict(counts)} | {epochs} epochs x {steps_per_epoch} steps", flush=True)
    model.train()
    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        for batch in length_batches(train, bc["batch_size"], rng):
            enc = encode_head_tail(tokenizer, [e.text for e in batch], bc["max_length"], bc["head_share"])
            enc = {k: v.to(device) for k, v in enc.items()}
            y = torch.tensor([LABEL_INDEX[e.label] for e in batch], device=device)
            loss = loss_fn(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"[{tag}] epoch {epoch} step {step}/{total} loss {running / 100:.4f}", flush=True)
                running = 0.0
        print(f"[{tag}] epoch {epoch} done in {time.time() - t0:.0f}s", flush=True)
    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer, device, test: list[Example], cfg: dict, tag: str) -> tuple[dict, list[dict]]:
    bc = cfg["bert"]
    clf = BertPageClassifier(
        model, tokenizer, labels=list(BERT_LABELS), max_length=bc["max_length"],
        head_share=bc["head_share"], device=device, meta={"base_model": bc["base_model"]},
    )
    service = PageClassifierService(
        clf,
        model_version="eval",
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
        route_short_pages=not bc.get("decide_short_pages", False),
    )
    proba = clf.predict_proba([e.text for e in test])
    raw_pred = [BERT_LABELS[int(i)] for i in proba.argmax(axis=1)]

    rows = []
    for e, raw, p in zip(test, raw_pred, proba):
        out = service.predict_one(e.uid, e.text)
        rows.append({
            "run": tag,
            "page_id": e.uid,
            "true_label": e.label,
            "true_flag": flag_of(e.label),
            "bert_label": raw,
            "bert_conf": round(float(p.max()), 4),
            "flag": out.flag,
            "subclass": out.subclass,
            "review_required": int(out.review_required),
            "decision_reason": out.decision_reason,
            "text_preview": " ".join(e.text.split())[:160],
        })
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    keep = [r for r in rows if r["true_flag"] == "KEEP"]
    drop = [r for r in rows if r["true_flag"] != "KEEP"]
    junk = [r for r in rows if r["true_flag"] == "JUNK"]
    keep_lost = [r["page_id"] for r in keep if r["flag"] != "KEEP"]
    missed = [r for r in drop if r["flag"] == "KEEP"]
    junk_flagged = [r for r in junk if r["flag"] == "JUNK"]
    confusion: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        confusion[r["true_label"]][r["bert_label"]] += 1
    return {
        "pages": n,
        "true_counts": dict(collections.Counter(r["true_label"] for r in rows)),
        "bert_raw_label_accuracy": round(sum(r["bert_label"] == r["true_label"] for r in rows) / n, 4) if n else None,
        "bert_raw_flag_accuracy": round(sum(flag_of(r["bert_label"]) == r["true_flag"] for r in rows) / n, 4) if n else None,
        "pipeline_flag_accuracy": round(sum(r["flag"] == r["true_flag"] for r in rows) / n, 4) if n else None,
        "keep_pages_lost": len(keep_lost),
        "keep_pages_lost_ids": keep_lost,
        "keep_pages_sent_to_review": sum(1 for r in keep if r["flag"] == "KEEP" and r["review_required"]),
        "junk_or_blank_pages": len(drop),
        "junk_or_blank_caught": len(drop) - len(missed),
        "junk_or_blank_missed": len(missed),
        "missed_but_sent_to_review": sum(1 for r in missed if r["review_required"]),
        "missed_silently_ids": [r["page_id"] for r in missed if not r["review_required"]],
        "junk_subtype_correct": sum(1 for r in junk_flagged if r["subclass"] == r["true_label"]),
        "junk_subtype_checked": len(junk_flagged),
        "confusion_true_vs_bert": {k: dict(v) for k, v in confusion.items()},
    }


def print_summary(tag: str, s: dict) -> None:
    print(
        f"\n=== {tag}: {s['pages']} real pages ===\n"
        f"  KEEP pages lost (flagged JUNK/BLANK): {s['keep_pages_lost']}  {s['keep_pages_lost_ids']}\n"
        f"  KEEP pages sent to review:            {s['keep_pages_sent_to_review']}\n"
        f"  junk/blank caught: {s['junk_or_blank_caught']}/{s['junk_or_blank_pages']}  "
        f"(missed {s['junk_or_blank_missed']}, of which {s['missed_but_sent_to_review']} went to review)\n"
        f"  junk subtype correct: {s['junk_subtype_correct']}/{s['junk_subtype_checked']}\n"
        f"  BERT raw 8-label accuracy {s['bert_raw_label_accuracy']}, flag accuracy {s['bert_raw_flag_accuracy']}, "
        f"pipeline flag accuracy {s['pipeline_flag_accuracy']}",
        flush=True,
    )


def free(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def continue_saved_model(cfg: dict, device, epochs: int) -> int:
    """Train the already-saved model on the new annotated pages and save it again."""
    bc = cfg["bert"]
    model_dir = ROOT / bc["model_dir"]
    if not (model_dir / "bjnk_meta.json").is_file():
        raise SystemExit(f"No saved model at {model_dir}")
    annotated = annotated_examples(cfg)
    if not annotated:
        raise SystemExit(f"No annotated pages in {ROOT / bc.get('annotated_jsonl', 'data/raw/annotated_pages.jsonl')}")
    # A short, small step. The saved weights stay; these pages nudge them.
    epochs = min(epochs, 2)
    model, tok = train_model(
        annotated, cfg, device, epochs=epochs, base_model=bc["base_model"], tag="continue",
        resume_dir=model_dir, learning_rate=1e-5, keep_weight=1.0,
    )
    out_dir = model_dir
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    meta = json.loads((model_dir / "bjnk_meta.json").read_text(encoding="utf-8"))
    meta.update({
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "continued_from": str(model_dir),
        "continue_epochs": epochs,
        "continue_pages": len(annotated),
        "continue_labels": dict(collections.Counter(e.label for e in annotated)),
    })
    save_bert(model, tok, tmp_dir, meta)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    print(f"\nSaved continued model -> {out_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--base-model")
    ap.add_argument("--device", default="auto")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--skip-eval", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    mode.add_argument("--continue-model", action="store_true",
                      help="Keep models/bert_page and train it further on the annotated pages")
    args = ap.parse_args()

    cfg = _load_cfg(Path(args.config))
    bc = cfg["bert"]
    epochs = args.epochs or bc["epochs"]
    base_model = args.base_model or bc["base_model"]
    device = pick_device(args.device)
    print(f"device={device} base_model={base_model} epochs={epochs}", flush=True)

    if args.continue_model:
        return continue_saved_model(cfg, device, epochs)

    synth = synthetic_examples(cfg)
    real = real_examples(cfg)
    annotated, dupes, conflicts = new_annotated(annotated_examples(cfg), synth + real)
    assign_folds(real)
    labels_csv = write_real_labels(real)
    print(f"synthetic pages: {len(synth)} {dict(collections.Counter(e.label for e in synth))}")
    print(f"real pages:      {len(real)} {dict(collections.Counter(e.label for e in real))}")
    print(
        f"annotated pages: {len(annotated)} {dict(collections.Counter(e.label for e in annotated))} "
        f"(skipped {dupes} already in training, {conflicts} conflicting)"
    )
    print(f"real folds:      {dict(collections.Counter((e.fold, flag_of(e.label)) for e in real))}")
    print(f"real page labels -> {labels_csv}", flush=True)

    copies = int(bc["real_copies"])
    report: dict = {"base_model": base_model, "epochs": epochs, "device": str(device), "runs": {}}
    all_rows: list[dict] = []

    if not args.skip_eval:
        model, tok = train_model(synth, cfg, device, epochs=epochs, base_model=base_model, tag="synthetic_only")
        s, rows = evaluate(model, tok, device, real, cfg, "synthetic_only")
        print_summary("synthetic_only (trained on synthetic, tested on all real pages)", s)
        report["runs"]["synthetic_only"] = s
        all_rows += rows
        free(model)

        fold_rows: list[dict] = []
        for test_fold in (0, 1):
            train = synth + [e for e in real if e.fold != test_fold] * copies
            test = [e for e in real if e.fold == test_fold]
            tag = f"real_2fold_test{test_fold}"
            model, tok = train_model(train, cfg, device, epochs=epochs, base_model=base_model, tag=tag)
            s, rows = evaluate(model, tok, device, test, cfg, tag)
            print_summary(tag, s)
            report["runs"][tag] = s
            fold_rows += rows
            free(model)
        combined = summarize(fold_rows)
        print_summary("real_2fold combined (every real page tested once)", combined)
        report["runs"]["real_2fold"] = combined
        all_rows += fold_rows

        pred_path = ROOT / "reports" / "bert_eval_pages.csv"
        with pred_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"per-page results -> {pred_path}")

    if not args.eval_only:
        train = synth + real * copies + annotated * copies
        model, tok = train_model(train, cfg, device, epochs=epochs, base_model=base_model, tag="final")
        out_dir = ROOT / bc["model_dir"]
        tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        meta = {
            "labels": list(BERT_LABELS),
            "max_length": bc["max_length"],
            "head_share": bc["head_share"],
            "base_model": base_model,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "train_pages": {
                "synthetic": len(synth),
                "real": len(real),
                "annotated": len(annotated),
                "real_copies": copies,
            },
            "eval": {k: {kk: v[kk] for kk in ("pages", "keep_pages_lost", "junk_or_blank_caught",
                                               "junk_or_blank_pages", "junk_subtype_correct", "junk_subtype_checked")}
                     for k, v in report["runs"].items()},
        }
        save_bert(model, tok, tmp_dir, meta)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
        print(f"\nSaved final model -> {out_dir}", flush=True)
        report["final_model"] = str(out_dir)

    eval_path = ROOT / "reports" / "bert_eval.json"
    eval_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
