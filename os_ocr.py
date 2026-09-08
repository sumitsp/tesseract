"""
Convert JPGs to structured DoclingDocuments using Docling's layout/table/reading-order
pipeline with RapidOCR (local model files) as the OCR engine.

For every image, produces:
  - Markdown with real heading hierarchy (# / ##) and proper pipe tables,
    driven by Docling's layout + table-structure (TableFormer) models.
  - A JSON sidecar with the full DoclingDocument dump: provenance (bounding
    boxes/polygons per item), table cell coordinates, and reading order
    (the order items appear in the document body) - everything Docling
    extracted, tied together in one place.

OCR text itself still comes from RapidOCR running on your local model files;
Docling only supplies layout, table structure, hierarchy, and reading order
on top of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, ImageFormatOption
from rapidocr.utils.typings import EngineType

INPUT_DIR = Path(r"\\ADMPDNLP06\AI_Vendor\Run1\Batch2\DEID_PNGs")
OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text")
MODELS_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rapidocr_models")
# OCR starts at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = "65214568_55368341"
JPG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}
DET_MODEL = MODELS_DIR / "PP-OCRv6_det_small.pth"
REC_MODEL = MODELS_DIR / "PP-OCRv6_rec_small.pth"
CLS_MODEL = MODELS_DIR / "ch_ptocr_mobile_v2.0_cls_mobile.pth"
REC_KEYS = MODELS_DIR / "ppocrv6_dict.txt"

# Raw params exactly as rapidocr's own RapidOCR(params={...}) constructor expects.
# Reused as-is if Docling's RapidOcrOptions exposes a passthrough field for it.
RAPIDOCR_PARAMS = {
    "Det.engine_type": EngineType.TORCH,
    "Cls.engine_type": EngineType.TORCH,
    "Rec.engine_type": EngineType.TORCH,
    "Det.model_path": str(DET_MODEL),
    "Cls.model_path": str(CLS_MODEL),
    "Rec.model_path": str(REC_MODEL),
    "Rec.rec_keys_path": str(REC_KEYS),
}


def log(message: str) -> None:
    print(message, flush=True)


def require_model_files() -> None:
    missing = [
        path
        for path in (DET_MODEL, REC_MODEL, CLS_MODEL, REC_KEYS)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        listed = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(
            "Missing RapidOCR model files:\n"
            f"{listed}\n\n"
            f"Put all 4 files in:\n  {MODELS_DIR}"
        )


def build_rapidocr_options() -> RapidOcrOptions:
    """
    Build RapidOcrOptions against whatever fields this installed Docling version
    actually exposes, instead of assuming one API shape. Tries, in order:
      1. A single passthrough dict field (e.g. `rapidocr_params`) that forwards
         straight to rapidocr's own RapidOCR(params={...}) - same shape as
         RAPIDOCR_PARAMS above.
      2. Individual path fields (det_model_path / cls_model_path / rec_model_path
         / rec_keys_path), plus an engine-type field if one exists, so the torch
         (.pth) models actually get used instead of Docling assuming ONNX.
    If neither shape matches, fails fast with the real field names instead of
    letting a mismatched kwarg fail silently or deep inside Docling.
    """
    fields = set(RapidOcrOptions.model_fields.keys())

    passthrough_candidates = [
        name for name in fields if "params" in name.lower() and "rapidocr" in name.lower()
    ] or [name for name in fields if name.lower() == "params"]
    if passthrough_candidates:
        field_name = passthrough_candidates[0]
        log(f"RapidOcrOptions: using passthrough field '{field_name}'")
        return RapidOcrOptions(force_full_page_ocr=True, **{field_name: dict(RAPIDOCR_PARAMS)})

    legacy_fields = {"det_model_path", "cls_model_path", "rec_model_path", "rec_keys_path"}
    if legacy_fields <= fields:
        kwargs: dict[str, Any] = {
            "force_full_page_ocr": True,
            "det_model_path": str(DET_MODEL),
            "cls_model_path": str(CLS_MODEL),
            "rec_model_path": str(REC_MODEL),
            "rec_keys_path": str(REC_KEYS),
        }
        engine_type_fields = [name for name in fields if "engine_type" in name.lower()]
        for name in engine_type_fields:
            kwargs[name] = EngineType.TORCH
        log(
            "RapidOcrOptions: using legacy per-path fields"
            + (f" + engine-type fields {engine_type_fields}" if engine_type_fields else " (no engine-type field found - torch .pth models may not load; check output below)")
        )
        return RapidOcrOptions(**kwargs)

    raise SystemExit(
        "Could not match RapidOcrOptions to a known shape for this Docling version.\n"
        f"Actual fields on RapidOcrOptions: {sorted(fields)}\n"
        "Update build_rapidocr_options() in os_ocr.py to use the correct field name(s)."
    )


def build_converter() -> DocumentConverter:
    require_model_files()
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options = build_rapidocr_options()
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )


def convert_image(converter: DocumentConverter, image_path: Path) -> tuple[str, dict]:
    """Returns (markdown, full DoclingDocument dict with provenance/bbox/table-cell data)."""
    result = converter.convert(str(image_path))
    doc = result.document
    return doc.export_to_markdown(), doc.export_to_dict()


def jpg_sort_key(path: Path) -> tuple:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_chart_folders(input_dir: Path) -> list[Path]:
    try:
        folders = sorted(
            (entry for entry in input_dir.iterdir() if entry.is_dir()),
            key=lambda path: path.name,
        )
    except Exception as exc:
        raise SystemExit(
            "Cannot read the network path. Check the share, VPN, and folder name.\n"
            f"  {input_dir}\n"
            f"  {type(exc).__name__}: {exc}"
        )
    if not START_FROM:
        return folders
    start_index = next(
        (i for i, folder in enumerate(folders) if folder.name == START_FROM),
        None,
    )
    if start_index is None:
        raise SystemExit(f"START_FROM folder not found under input: {START_FROM}")
    log(f"Starting OCR at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def list_jpgs(folder: Path) -> list[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix in JPG_SUFFIXES
    ]
    images.sort(key=jpg_sort_key)
    return images


def process_folder(
    converter: DocumentConverter,
    input_dir: Path,
    output_dir: Path,
    folder: Path,
    images: list[Path],
) -> None:
    out_folder = output_dir / folder.relative_to(input_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{folder.name}.json"
    log(f"  {len(images)} images -> {out_file}")

    pages: dict[str, dict] = {}
    for image_path in images:
        log(f"  {image_path.name}")
        try:
            markdown, doc_dict = convert_image(converter, image_path)
            pages[image_path.name] = {"markdown": markdown or "[no text detected]", "document": doc_dict}
        except Exception as exc:
            pages[image_path.name] = {
                "markdown": f"[ERROR extracting {image_path.name}: {exc}]",
                "document": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        out_file.write_text(json.dumps(pages, indent=2, default=str), encoding="utf-8")
        log(f"  saved {out_file}")

    log(f"Finished {out_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log("=== DOCLING + RAPIDOCR (local models) ===")
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    log(f"INPUT:  {input_dir}")
    log(f"OUTPUT: {output_dir}")
    log(f"START_FROM: {START_FROM or '(first folder)'}")
    if not input_dir.is_dir():
        raise SystemExit(
            "Input folder does not exist or the network share is not reachable:\n"
            f"  {input_dir}\n"
            "If File Explorer shows 'Batch1' with no space, remove the space in the path."
        )
    chart_folders = list_chart_folders(input_dir)
    if not chart_folders:
        raise SystemExit(f"No chart folders found under: {input_dir}")
    log("Loading Docling + RapidOCR from local models...")
    converter = build_converter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in chart_folders:
        images = list_jpgs(folder)
        if not images:
            log(f"Folder: {folder.name} (no JPGs, skipped)")
            continue
        log(f"Folder: {folder}")
        process_folder(converter, input_dir, output_dir, folder, images)


if __name__ == "__main__":
    main()
