#!/usr/bin/env python3
"""
Classify each page image as Printed or Handwritten → CSV.

Reads chart folders/images from Azure Blob Storage (same path style as
rapid_ocr.py / file_counter.py), downloads each image for classification,
then writes results to CSV.

Usage:
  python classify_hw_printed.py
  python classify_hw_printed.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from hw_printed import classify_image_type, load_model

SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# AZURE CONFIGURATION (same style as rapid_ocr.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "imaging-pipeline"
PREFIX = "Raw_Input/Run1/Batch1/DEID_PNGs/"

# Classification starts at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = "52748416_44709403"

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

CSV_COLUMNS = [
    "chart_name",
    "page_name",
    "page_number",
    "handwritten_or_printed",
    "confidence",
    "method",
]


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
    log(f"Starting at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def page_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_page_blobs(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=page_sort_key)
    return images


def download_blob_bytes(container_client: ContainerClient, blob_name: str) -> bytes:
    return container_client.download_blob(blob_name).readall()


def init_csv(path: Path) -> None:
    """Create CSV with header only (overwrite if exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        f.flush()


def append_csv_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        f.flush()


def process_folder(
    container_client: ContainerClient,
    model,
    folder_name: str,
    blob_names: list[str],
    out_csv: Path,
    per_chart_dir: Path | None,
) -> int:
    log(f"  {len(blob_names)} images")
    chart_csv: Path | None = None
    if per_chart_dir is not None:
        chart_csv = per_chart_dir / f"{folder_name}_hw_printed.csv"
        init_csv(chart_csv)

    count = 0
    for page_number, blob_name in enumerate(blob_names, start=1):
        filename = Path(blob_name).name
        log(f"  {filename}")
        try:
            image_bytes = download_blob_bytes(container_client, blob_name)
            label, conf, method = classify_image_type(image_bytes, model=model)
        except Exception as exc:
            label, conf, method = "ERROR", None, str(exc)
            log(f"    ERROR: {exc}")

        row = {
            "chart_name": folder_name,
            "page_name": filename,
            "page_number": str(page_number),
            "handwritten_or_printed": label,
            "confidence": "" if conf is None else f"{conf:.4f}",
            "method": method,
        }
        append_csv_row(out_csv, row)
        if chart_csv is not None:
            append_csv_row(chart_csv, row)
        count += 1

    if chart_csv is not None:
        log(f"  → {chart_csv}")
    return count


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Classify blob page images as Printed / Handwritten → CSV"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / "image_type_classification.pkl",
        help="Path to image_type_classification.pkl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\sumit.pandey\Desktop\hw_printed.csv"),
        help="Combined CSV path",
    )
    parser.add_argument(
        "--per-chart",
        action="store_true",
        help="Also write one CSV per chart under output/per_chart/",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N folders (0 = all)",
    )
    args = parser.parse_args()

    out_csv = args.out.resolve()
    # Create the CSV immediately so it appears before Azure scan / classification.
    init_csv(out_csv)

    log("=== HW / PRINTED CLASSIFY (Azure Blob) ===")
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {out_csv}")
    log(f"CSV started: {out_csv}")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / image blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    if args.limit and args.limit > 0:
        chart_folders = chart_folders[: args.limit]

    model = load_model(args.model.resolve())
    log(f"folders: {len(chart_folders)}")

    per_chart_dir = (SCRIPT_DIR / "output" / "per_chart") if args.per_chart else None
    if per_chart_dir is not None:
        per_chart_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for folder_name in chart_folders:
        images = list_page_blobs(folder_blobs, folder_name)
        if not images:
            log(f"Folder: {folder_name} (no images, skipped)")
            continue
        log(f"Folder: {folder_name}")
        total_rows += process_folder(
            container_client, model, folder_name, images, out_csv, per_chart_dir
        )

    log(f"done — {total_rows} rows → {out_csv}")


if __name__ == "__main__":
    main()
