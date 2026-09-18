"""
Extract raw text from JPGs with RapidOCR using local model files.

Reads chart folders/images from Azure Blob Storage (same path style as file_counter.py),
downloads each image locally for OCR, then writes results under OUTPUT_DIR.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from rapidocr import RapidOCR
from rapidocr.utils.typings import EngineType

# ============================================================
# RUN CONFIG — edit these (no .env)
# ============================================================

INPUT_SOURCE = "local"  # "local" or "blob"
LOCAL_INPUT = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\input")
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text")
MODELS_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rapidocr_models")
# OCR starts at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = "52781821_48221457"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DET_MODEL = MODELS_DIR / "PP-OCRv6_det_small.pth"
REC_MODEL = MODELS_DIR / "PP-OCRv6_rec_small.pth"
CLS_MODEL = MODELS_DIR / "ch_ptocr_mobile_v2.0_cls_mobile.pth"
REC_KEYS = MODELS_DIR / "ppocrv6_dict.txt"


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
            if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
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


def list_local_images(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"LOCAL_INPUT does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported image type: {path.suffix}")
        return [path]
    return sorted(
        (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).lower(),
    )


def process_local(ocr: RapidOCR, input_path: Path, output_dir: Path) -> None:
    images = list_local_images(input_path)
    if not images:
        raise SystemExit(f"No supported images found under: {input_path}")
    root = input_path.resolve() if input_path.is_dir() else input_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        relative = image_path.resolve().relative_to(root)
        out_file = output_dir / relative.with_suffix(".txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        log(f"  {image_path} -> {out_file}")
        try:
            text = extract_text(ocr, image_path) or "[no text detected]"
        except Exception as exc:
            text = f"[ERROR extracting {image_path.name}: {exc}]"
        out_file.write_text(text + "\n", encoding="utf-8")


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


def build_ocr() -> RapidOCR:
    require_model_files()
    return RapidOCR(
        params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
            "Det.model_path": str(DET_MODEL),
            "Cls.model_path": str(CLS_MODEL),
            "Rec.model_path": str(REC_MODEL),
            "Rec.rec_keys_path": str(REC_KEYS),
        }
    )


def extract_text(ocr: RapidOCR, image_path: Path) -> str:
    result = ocr(str(image_path))
    if result is None or not getattr(result, "txts", None):
        return ""
    lines = [line.strip() for line in result.txts if line and line.strip()]
    return "\n".join(lines)


def process_folder(
    ocr: RapidOCR,
    container_client: ContainerClient,
    output_dir: Path,
    folder_name: str,
    blob_names: list[str],
) -> None:
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{folder_name}.txt"
    log(f"  {len(blob_names)} images -> {out_file}")
    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"rapid_ocr_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                text = extract_text(ocr, local_path) or "[no text detected]"
            except Exception as exc:
                text = f"[ERROR extracting {filename}: {exc}]"
            parts.append(f"===== {filename} =====\n{text}")
            out_file.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
            log(f"  saved {out_file}")
    log(f"Finished {out_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    source = INPUT_SOURCE.strip().lower()
    if source not in {"local", "blob"}:
        raise SystemExit('INPUT_SOURCE must be "local" or "blob"')
    log("=== RAPID OCR (no Docling, no HuggingFace) ===")
    output_dir = OUTPUT_DIR
    log(
        f"INPUT:  {LOCAL_INPUT if source == 'local' else f'azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}'}"
    )
    log(f"OUTPUT: {output_dir}")

    log("Loading RapidOCR from local models...")
    ocr = build_ocr()
    if source == "local":
        process_local(ocr, LOCAL_INPUT, output_dir)
        return

    log(f"START_FROM: {START_FROM or '(first folder)'}")
    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / images found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in chart_folders:
        images = list_jpgs(folder_blobs, folder_name)
        if not images:
            log(f"Folder: {folder_name} (no JPGs, skipped)")
            continue
        log(f"Folder: {folder_name}")
        process_folder(ocr, container_client, output_dir, folder_name, images)


if __name__ == "__main__":
    main()
