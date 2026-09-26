#!/usr/bin/env python3
"""Flag local OCR pages as KEEP / BLANK / JUNK → Excel (nothing is deleted).

Edit INPUT_PATH below, then run:
  python scripts/infer.py

Excel is saved next to the input (same folder) and written from the start —
updated after every page so you don't wait until the end.

Supports:
  - Docling+RapidOCR JSON: {"1.jpg": {"markdown": "...", "document": {...}}, ...}
  - RapidOCR .txt: ===== 1.jpg =====\\n text ...
  - Folder of those files / chart subfolders
  - JSONL/CSV/XLSX with page_id + ocr_text

Prints progress to the terminal while processing.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Edit this path only — Excel is saved in the same folder as INPUT_PATH
# ---------------------------------------------------------------------------
INPUT_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_docling")
# Examples:
# INPUT_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_docling\52743839_44976074\52743839_44976074.json")
# INPUT_PATH = Path.home() / "Desktop" / "Imaging" / "docling_format_and_rapid"
# INPUT_PATH = ROOT / "data" / "samples" / "sample_pages.jsonl"

MODEL_PATH = ROOT / "models" / "tfidf_flat.joblib"
CONFIG_PATH = ROOT / "configs" / "default.json"
# ---------------------------------------------------------------------------

PAGE_SPLIT_TXT = re.compile(r"=====+\s*([^\n=]+?)\s*=====+")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}


def _load_cfg(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError as exc:
        raise SystemExit("Install PyYAML or use configs/default.json") from exc


from src.inference.predict import InferenceResult, PageClassifierService  # noqa: E402
from src.models.classifiers import FlatClassifier  # noqa: E402
from src.models.decision import DecisionConfig  # noqa: E402


COLUMNS = [
    "page_id",
    "flag",
    "audit_tag",
    "confidence",
    "review_required",
    "p_keep",
    "p_blank",
    "p_junk",
    "decision_reason",
    "top_evidence",
    "model_version",
]


def _natural_page_key(name: str):
    stem = Path(name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def _clean_ocr_text(text: str) -> str:
    """Normalize Docling markdown / OCR dumps for the text classifier."""
    if not text:
        return ""
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\xa0", " ")
    text = re.sub(r"<!--\s*image\s*-->", " ", text, flags=re.I)
    text = re.sub(r"\[no text detected\]", " ", text, flags=re.I)
    text = re.sub(r"\[ERROR extracting[^\]]*\]", " ", text, flags=re.I)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_from_page_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_ocr_text(value)
    if not isinstance(value, dict):
        return _clean_ocr_text(str(value))
    for key in ("markdown", "ocr_text", "text", "content", "raw_text"):
        if value.get(key):
            return _clean_ocr_text(str(value[key]))
    return ""


def _is_docling_pages_dict(payload: dict) -> bool:
    """True for {"1.jpg": {"markdown": "...", "document": {...}}, ...}."""
    if not payload:
        return False
    # classic flat page record
    if "page_id" in payload and ("ocr_text" in payload or "markdown" in payload):
        return False
    if "pages" in payload or "data" in payload:
        return False

    hits = 0
    for key, val in list(payload.items())[:12]:
        suffix = Path(str(key)).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            hits += 1
            continue
        if isinstance(val, dict) and any(k in val for k in ("markdown", "document", "ocr_text", "text")):
            hits += 1
    return hits >= 1


def _pages_from_docling_dict(payload: dict, *, prefix: str = "") -> list[dict[str, str]]:
    items = sorted(payload.items(), key=lambda kv: _natural_page_key(str(kv[0])))
    pages: list[dict[str, str]] = []
    for name, value in items:
        page_id = f"{prefix}{name}" if prefix else str(name)
        pages.append({"page_id": page_id, "ocr_text": _text_from_page_value(value)})
    return pages


def _pages_from_rapid_txt(text: str, *, prefix: str = "") -> list[dict[str, str]]:
    parts = PAGE_SPLIT_TXT.split(text)
    if len(parts) < 3:
        body = _clean_ocr_text(text)
        return [{"page_id": f"{prefix}page" if prefix else "page", "ocr_text": body}] if body else []
    pages: list[dict[str, str]] = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = _clean_ocr_text(parts[i + 1] if i + 1 < len(parts) else "")
        page_id = f"{prefix}{name}" if prefix else name
        pages.append({"page_id": page_id, "ocr_text": body})
    pages.sort(key=lambda p: _natural_page_key(Path(p["page_id"]).name))
    return pages


def _load_json_file(path: Path, *, prefix: str = "") -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        out: list[dict[str, str]] = []
        for i, obj in enumerate(payload):
            if isinstance(obj, dict) and ("page_id" in obj or "ocr_text" in obj or "markdown" in obj):
                pid = str(obj.get("page_id") or f"page_{i+1}")
                text = obj.get("ocr_text") or obj.get("markdown") or obj.get("text") or ""
                out.append({"page_id": f"{prefix}{pid}", "ocr_text": _clean_ocr_text(str(text))})
            else:
                out.append({"page_id": f"{prefix}page_{i+1}", "ocr_text": _text_from_page_value(obj)})
        return out

    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: JSON must be an object or list")

    if _is_docling_pages_dict(payload):
        return _pages_from_docling_dict(payload, prefix=prefix)

    # wrapper shapes
    if "pages" in payload and isinstance(payload["pages"], dict):
        return _pages_from_docling_dict(payload["pages"], prefix=prefix)
    if "pages" in payload and isinstance(payload["pages"], list):
        out = []
        for i, obj in enumerate(payload["pages"], start=1):
            if not isinstance(obj, dict):
                continue
            pid = str(obj.get("page_id") or f"page_{i}")
            text = obj.get("ocr_text") or obj.get("markdown") or obj.get("text") or ""
            out.append({"page_id": f"{prefix}{pid}", "ocr_text": _clean_ocr_text(str(text))})
        return out

    # single-page docling-ish
    if "markdown" in payload:
        return [{"page_id": f"{prefix}{path.stem}", "ocr_text": _text_from_page_value(payload)}]

    if "page_id" in payload:
        return [
            {
                "page_id": f"{prefix}{payload['page_id']}",
                "ocr_text": _clean_ocr_text(
                    str(payload.get("ocr_text") or payload.get("markdown") or "")
                ),
            }
        ]

    raise SystemExit(
        f"{path}: unrecognized JSON shape. Expected Docling map "
        '{"1.jpg": {"markdown": "...", "document": {...}}}'
    )


def _load_pages(path: Path) -> list[dict[str, str]]:
    pages: list[dict[str, str]] = []

    if path.is_dir():
        # Chart folder layout from pipeline: <dir>/<chart>/<chart>.json or .txt
        json_files = sorted(path.rglob("*.json"))
        txt_files = sorted(path.rglob("*.txt"))
        # Prefer Docling JSON when present
        if json_files:
            for fp in json_files:
                # skip tiny label maps etc.
                try:
                    head = fp.read_text(encoding="utf-8", errors="replace")[:200].lstrip()
                except OSError:
                    continue
                if not head.startswith("{"):
                    continue
                rel = fp.relative_to(path)
                prefix = f"{rel.parent.as_posix()}/" if rel.parent != Path(".") else ""
                chunk = _load_json_file(fp, prefix=prefix)
                print(f"  loaded {len(chunk)} pages from {rel}", flush=True)
                pages.extend(chunk)
            return pages

        if txt_files:
            for fp in txt_files:
                rel = fp.relative_to(path)
                prefix = f"{rel.parent.as_posix()}/" if rel.parent != Path(".") else ""
                chunk = _pages_from_rapid_txt(
                    fp.read_text(encoding="utf-8", errors="replace"), prefix=prefix
                )
                print(f"  loaded {len(chunk)} pages from {rel}", flush=True)
                pages.extend(chunk)
            return pages

        # fallback: loose .txt / .ocr page files
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".ocr"} and p.is_file()
        )
        for fp in files:
            pages.append(
                {
                    "page_id": str(fp.relative_to(path)),
                    "ocr_text": _clean_ocr_text(
                        fp.read_text(encoding="utf-8", errors="replace")
                    ),
                }
            )
        return pages

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        import pandas as pd

        df = pd.read_excel(path)
        cols = {c.lower(): c for c in df.columns}
        if "page_id" not in cols:
            raise SystemExit(f"{path}: Excel must have page_id column")
        text_col = cols.get("ocr_text") or cols.get("markdown") or cols.get("text")
        if not text_col:
            raise SystemExit(f"{path}: Excel must have ocr_text or markdown column")
        for _, row in df.iterrows():
            raw = row[text_col]
            pages.append(
                {
                    "page_id": str(row[cols["page_id"]]),
                    "ocr_text": "" if pd.isna(raw) else _clean_ocr_text(str(raw)),
                }
            )
        return pages

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                text = row.get("ocr_text") or row.get("markdown") or row.get("text") or ""
                pages.append(
                    {
                        "page_id": str(row.get("page_id", "")),
                        "ocr_text": _clean_ocr_text(text),
                    }
                )
        return pages

    if suffix == ".json":
        return _load_json_file(path)

    if suffix == ".txt":
        return _pages_from_rapid_txt(path.read_text(encoding="utf-8", errors="replace"))

    # JSONL
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("ocr_text") or obj.get("markdown") or obj.get("text") or ""
            pages.append(
                {
                    "page_id": str(obj.get("page_id") or obj.get("file") or ""),
                    "ocr_text": _clean_ocr_text(str(text)),
                }
            )
    return pages


def _row_from_result(r: InferenceResult) -> dict:
    row = r.to_dict()
    row["review_required"] = int(bool(row["review_required"]))
    for k in ("p_keep", "p_blank", "p_junk"):
        if row.get(k) is None:
            row[k] = ""
    return {k: row.get(k, "") for k in COLUMNS}


def _output_path_from_input(in_path: Path) -> Path:
    """Excel always lives next to the input you started with."""
    if in_path.is_dir():
        return in_path / "page_flags.xlsx"
    return in_path.with_name(f"{in_path.stem}_page_flags.xlsx")


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
            "Or install openpyxl in the venv."
        ) from exc


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _open_excel_writer(path: Path):
    """Create Excel at start (header only); caller appends + saves per page."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise SystemExit("Excel needs openpyxl: pip install openpyxl") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "page_flags"
    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    wb.save(path)
    return wb, ws


def main() -> int:
    in_path = Path(INPUT_PATH)
    out = _output_path_from_input(in_path)
    model_path = Path(MODEL_PATH)
    config_path = Path(CONFIG_PATH)

    if not in_path.exists():
        raise SystemExit(
            f"Input not found: {in_path}\n"
            f"Edit INPUT_PATH at the top of scripts/infer.py"
        )
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}", flush=True)
    cfg = _load_cfg(config_path)
    model = FlatClassifier.load(str(model_path))
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

    print(f"Excel (from start): {out}", flush=True)
    wb, ws = _open_excel_writer(out)
    print(f"Flagging {total} pages…", flush=True)

    rows: list[dict] = []
    n_keep = n_blank = n_junk = n_review = 0
    for i, page in enumerate(pages, start=1):
        r = service.predict_one(page["page_id"], page.get("ocr_text", ""))
        row = _row_from_result(r)
        rows.append(row)
        ws.append([row.get(c, "") for c in COLUMNS])
        wb.save(out)  # save after every page from the start
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
            f"(conf={r.confidence:.3f}{review}) | saved",
            flush=True,
        )

    # final polish (filter + column widths)
    _write_excel(out, rows)

    print(
        f"Done. Wrote {total} rows → {out}\n"
        f"  KEEP={n_keep}  BLANK={n_blank}  JUNK={n_junk}  review_required={n_review}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
