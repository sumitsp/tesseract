"""
Extract raw text from JPGs with RapidOCR using local model files.

Same as rapid_ocr.py, but only processes the folders listed in FOLDERS
(comma-separated names). Does not run later folders automatically.
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
# AZURE CONFIGURATION (same style as file_counter.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text")
MODELS_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rapidocr_models")

# Comma-separated folder names to process (only these; nothing else).
# Example: "52781821_48221457,52743839_44976074,52743997_45500291"
FOLDERS = "52781821_48221457,52743839_44976074"

JPG_SUFFIXES = {".jpg", ".jpeg"}
DET_MODEL = MODELS_DIR / "PP-OCRv6_det_small.pth"
REC_MODEL = MODELS_DIR / "PP-OCRv6_rec_small.pth"
CLS_MODEL = MODELS_DIR / "ch_ptocr_mobile_v2.0_cls_mobile.pth"
REC_KEYS = MODELS_DIR / "ppocrv6_dict.txt"


def log(message: str) -> None:
    print(message, flush=True)


def parse_folder_list(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",")]
    names = [name for name in names if name]
    if not names:
        raise SystemExit("FOLDERS is empty. Add comma-separated folder names.")
    # Keep order as written; drop duplicates while preserving first occurrence.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


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


def list_folder_blobs(
    container_client: ContainerClient,
    selected: list[str],
) -> dict[str, list[str]]:
    """Group image blob names for the selected folders only."""
    log("=" * 70)
    log("Scanning Azure Blob (selected folders only)")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log(f"Folders         : {', '.join(selected)}")
    log("=" * 70)

    selected_set = set(selected)
    folder_blobs: dict[str, list[str]] = {name: [] for name in selected}

    try:
        # List under each folder prefix for less scanning than the whole tree.
        for folder_name in selected:
            folder_prefix = f"{PREFIX}{folder_name}/"
            for blob in container_client.list_blobs(name_starts_with=folder_prefix):
                relative_path = blob.name[len(PREFIX) :]
                parts = relative_path.split("/")
                if len(parts) < 2:
                    continue
                name = parts[0]
                if name not in selected_set:
                    continue
                filename = parts[-1]
                if Path(filename).suffix.lower() not in JPG_SUFFIXES:
                    continue
                folder_blobs[name].append(blob.name)
    except Exception as exc:
        raise SystemExit(f"ERROR while reading blobs:\n{exc}") from exc

    return folder_blobs


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
    log("=== RAPID OCR SELECTED FOLDERS (no Docling) ===")
    output_dir = OUTPUT_DIR
    selected = parse_folder_list(FOLDERS)
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {output_dir}")
    log(f"FOLDERS ({len(selected)}): {', '.join(selected)}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client, selected)

    missing = [name for name in selected if not folder_blobs.get(name)]
    if missing:
        log("WARNING: no JPGs found for:")
        for name in missing:
            log(f"  - {name}")

    log("Loading RapidOCR from local models...")
    ocr = build_ocr()
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in selected:
        images = list_jpgs(folder_blobs, folder_name)
        if not images:
            log(f"Folder: {folder_name} (no JPGs, skipped)")
            continue
        log(f"Folder: {folder_name}")
        process_folder(ocr, container_client, output_dir, folder_name, images)


if __name__ == "__main__":
    main()
