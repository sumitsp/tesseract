#!/usr/bin/env python3
"""Flag local OCR pages as KEEP / BLANK / JUNK → Excel (nothing is deleted).

Reads JSONL/JSON/CSV/XLSX (page_id + ocr_text) or a folder of .txt files.
Prints progress to the terminal while processing.
"""

from __future__ import annotations

import argparse
import csv
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


from src.inference.predict import InferenceResult, PageClassifierService  # noqa: E402
from src.models.classifiers import FlatClassifier  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402


COLUMNS = [
    "page_id",
    "flag",
    "confidence",
    "review_required",
    "p_keep",
    "p_blank",
    "p_junk",
    "decision_reason",
    "top_evidence",
    "model_version",
]


def _load_pages(path: Path) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".ocr"} and p.is_file()
        )
        for fp in files:
            pages.append(
                {
                    "page_id": str(fp.relative_to(path)),
                    "ocr_text": fp.read_text(encoding="utf-8", errors="replace"),
                }
            )
        return pages

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        import pandas as pd

        df = pd.read_excel(path)
        cols = {c.lower(): c for c in df.columns}
        if "page_id" not in cols or "ocr_text" not in cols:
            raise SystemExit(f"{path}: Excel must have page_id and ocr_text columns")
        for _, row in df.iterrows():
            pages.append(
                {
                    "page_id": str(row[cols["page_id"]]),
                    "ocr_text": "" if pd.isna(row[cols["ocr_text"]]) else str(row[cols["ocr_text"]]),
                }
            )
        return pages

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pages.append(
                    {
                        "page_id": str(row.get("page_id", "")),
                        "ocr_text": row.get("ocr_text", "") or "",
                    }
                )
        return pages

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pages
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("pages") or payload.get("data") or [payload]
        for obj in payload:
            pages.append(
                {"page_id": str(obj["page_id"]), "ocr_text": obj.get("ocr_text", "")}
            )
        return pages

    # JSONL (default)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pages.append(
                {"page_id": str(obj["page_id"]), "ocr_text": obj.get("ocr_text", "")}
            )
    return pages


def _row_from_result(r: InferenceResult) -> dict:
    row = r.to_dict()
    row["review_required"] = int(bool(row["review_required"]))
    for k in ("p_keep", "p_blank", "p_junk"):
        if row.get(k) is None:
            row[k] = ""
    return {k: row.get(k, "") for k in COLUMNS}


def _write_excel(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font

        wb = Workbook()
        ws = wb.active
        ws.title = "page_flags"
        ws.append(COLUMNS)
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in rows:
            ws.append([row.get(c, "") for c in COLUMNS])
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            letter = col[0].column_letter
            width = max(len(str(c.value or "")) for c in col[:50])
            ws.column_dimensions[letter].width = min(max(width + 2, 10), 48)
        wb.save(path)
        return
    except ImportError:
        pass

    try:
        import pandas as pd

        pd.DataFrame(rows, columns=COLUMNS).to_excel(path, index=False, sheet_name="page_flags")
        return
    except Exception as exc:
        raise SystemExit(
            "Excel output needs openpyxl (pip install openpyxl).\n"
            "Or pass --output reports/page_flags.csv"
        ) from exc


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Flag OCR pages as KEEP / BLANK / JUNK → Excel (flagging only)"
    )
    ap.add_argument("--model", default=str(ROOT / "models/tfidf_flat.joblib"))
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument(
        "--input",
        required=True,
        help="JSONL/JSON/CSV/XLSX with page_id+ocr_text, or a folder of .txt files",
    )
    ap.add_argument(
        "--output",
        default=str(ROOT / "reports/page_flags.xlsx"),
        help="Output path (.xlsx default; use .csv for CSV)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    print(f"Loading model: {args.model}", flush=True)
    cfg = _load_cfg(Path(args.config))
    model = FlatClassifier.load(args.model)
    service = PageClassifierService(
        model,
        model_version=cfg["model_version"],
        decision=DecisionConfig(**cfg["decision"]),
        min_dictionary_words=cfg["routing"]["min_dictionary_words"],
    )

    print(f"Reading input: {in_path}", flush=True)
    pages = _load_pages(in_path)
    total = len(pages)
    if total == 0:
        raise SystemExit("No pages found in input")
    print(f"Flagging {total} pages…", flush=True)

    rows: list[dict] = []
    n_keep = n_blank = n_junk = n_review = 0
    for i, page in enumerate(pages, start=1):
        r = service.predict_one(page["page_id"], page.get("ocr_text", ""))
        rows.append(_row_from_result(r))
        if r.flag == "KEEP":
            n_keep += 1
        elif r.flag == "BLANK":
            n_blank += 1
        elif r.flag == "JUNK":
            n_junk += 1
        if r.review_required:
            n_review += 1
        review = " review" if r.review_required else ""
        print(
            f"[{i}/{total}] {r.page_id} → {r.flag} "
            f"(conf={r.confidence:.3f}{review})",
            flush=True,
        )

    if out.suffix.lower() in {".xlsx", ".xlsm"}:
        _write_excel(out, rows)
    else:
        _write_csv(out, rows)

    print(
        f"Done. Wrote {total} rows → {out}\n"
        f"  KEEP={n_keep}  BLANK={n_blank}  JUNK={n_junk}  review_required={n_review}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
