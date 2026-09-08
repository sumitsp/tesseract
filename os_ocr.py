"""
Convert JPGs to structured DoclingDocuments using Docling's layout/table/reading-order
pipeline with RapidOCR (local model files) as the OCR engine.

Reads chart folders/images from Azure Blob Storage (same path style as file_counter.py),
downloads each image locally for OCR, then writes results under OUTPUT_DIR.

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
import tempfile
from pathlib import Path
from typing import Any

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, ImageFormatOption
from rapidocr.utils.typings import EngineType

# ============================================================
# AZURE CONFIGURATION (same style as file_counter.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text")
MODELS_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rapidocr_models")
# OCR starts at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = "65214568_55368341"
JPG_SUFFIXES = {".jpg", ".jpeg"}
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


def connect_azure() -> ContainerClient:
    log("Connecting to Azure Blob Storage...")
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        log("Azure connection created.")
        return container_client
    except Exception as exc:
        raise SystemExit(f"ERROR connecting to Azure:\n{exc}") from exc


def list_folder_blobs(container_client: ContainerClient) -> dict[str, list[str]]:
    """Group image blob names by immediate folder under PREFIX."""
    log("=" * 70)
    log("Scanning Azure Blob")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log("=" * 70)

    folder_blobs: dict[str, list[str]] = {}
    try:
        blobs = container_client.list_blobs(name_starts_with=PREFIX)
        for blob in blobs:
            relative_path = blob.name[len(PREFIX) :]
            parts = relative_path.split("/")
            if len(parts) < 2:
                continue
            folder_name = parts[0]
            filename = parts[-1]
            if Path(filename).suffix.lower() not in JPG_SUFFIXES:
                continue
            folder_blobs.setdefault(folder_name, []).append(blob.name)
    except Exception as exc:
        raise SystemExit(f"ERROR while reading blobs:\n{exc}") from exc

    return folder_blobs


def list_chart_folders(folder_blobs: dict[str, list[str]]) -> list[str]:
    folders = sorted(folder_blobs.keys(), key=lambda name: name.lower())
    if not START_FROM:
        return folders
    start_index = next(
        (i for i, folder in enumerate(folders) if folder == START_FROM),
        None,
    )
    if start_index is None:
        raise SystemExit(f"START_FROM folder not found under prefix: {START_FROM}")
    log(f"Starting OCR at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def jpg_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_jpgs(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=jpg_sort_key)
    return images


def download_blob(
    container_client: ContainerClient,
    blob_name: str,
    dest: Path,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = container_client.download_blob(blob_name).readall()
    dest.write_bytes(data)
    return dest


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


def process_folder(
    converter: DocumentConverter,
    container_client: ContainerClient,
    output_dir: Path,
    folder_name: str,
    blob_names: list[str],
) -> None:
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{folder_name}.json"
    log(f"  {len(blob_names)} images -> {out_file}")

    pages: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix=f"os_ocr_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                markdown, doc_dict = convert_image(converter, local_path)
                pages[filename] = {"markdown": markdown or "[no text detected]", "document": doc_dict}
            except Exception as exc:
                pages[filename] = {
                    "markdown": f"[ERROR extracting {filename}: {exc}]",
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
    output_dir = OUTPUT_DIR
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {output_dir}")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / JPG blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    log("Loading Docling + RapidOCR from local models...")
    converter = build_converter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in chart_folders:
        images = list_jpgs(folder_blobs, folder_name)
        if not images:
            log(f"Folder: {folder_name} (no JPGs, skipped)")
            continue
        log(f"Folder: {folder_name}")
        process_folder(converter, container_client, output_dir, folder_name, images)


if __name__ == "__main__":
    main()
