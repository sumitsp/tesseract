"""
Extract raw text from images with Tesseract.
OCR starts at START_FROM, then continues with later folders.

Reads chart folders/images from Azure Blob Storage (same path style as file_counter.py),
downloads each image locally for OCR, then writes results under OUTPUT_DIR.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytesseract
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from PIL import Image

# ============================================================
# AZURE CONFIGURATION (same style as file_counter.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")
# OCR starts at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = "65214568_55368341"
# Set this if auto-detect fails. Leave as-is to search common locations + PATH.
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
CANDIDATE_CMDS = [
    TESSERACT_CMD,
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".gif",
    ".j2k",
    ".jfif",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
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


def image_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_images(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=image_sort_key)
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


def find_tesseract() -> Path:
    for candidate in CANDIDATE_CMDS:
        if candidate.is_file():
            return candidate
    which = shutil.which("tesseract")
    if which:
        return Path(which)
    raise SystemExit(
        "Tesseract is not installed, or tesseract.exe is not on PATH.\n"
        "1. Install: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "2. During setup, tick 'Add to PATH'.\n"
        "3. Close and reopen PowerShell, then run:\n"
        "     tesseract --version\n"
        "4. If that works, run ts_ocr.py again.\n"
        "   If not, set TESSERACT_CMD in ts_ocr.py to the full path of tesseract.exe."
    )


def extract_text(image_path: Path) -> str:
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image)
    return text.strip()


def process_folder(
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
    with tempfile.TemporaryDirectory(prefix=f"ts_ocr_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                text = extract_text(local_path) or "[no text detected]"
            except Exception as exc:
                text = f"[ERROR extracting {filename}: {exc}]"
            parts.append(f"===== {filename} =====\n{text}")
            out_file.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
            log(f"  saved {out_file}")
    log(f"Finished {out_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    tesseract_cmd = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
    log(f"Using Tesseract: {tesseract_cmd}")
    log("=== TESSERACT OCR ===")
    output_dir = OUTPUT_DIR
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {output_dir}")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / image blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    log(f"Will process {len(chart_folders)} chart folder(s)")
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in chart_folders:
        images = list_images(folder_blobs, folder_name)
        if not images:
            log(f"Folder: {folder_name} (no images, skipped)")
            continue
        log(f"Folder: {folder_name}")
        process_folder(container_client, output_dir, folder_name, images)


if __name__ == "__main__":
    main()
