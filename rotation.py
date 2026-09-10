"""
Azure blob batch runner for page orientation correction.

Detection is delegated to ``page_orientation.PageOrientationDetector``
(OpenCV only — no OCR).
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import cv2
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from page_orientation import PageOrientationDetector

# ============================================================
# AZURE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rotation_report.csv")
START_FROM = ""
JPG_SUFFIXES = {".jpg", ".jpeg"}

# Tilt is measured always; applied only when |tilt| <= this (detector setting).
MAX_TILT_TO_APPLY = 5.0

CSV_FIELDS = [
    "folder",
    "filename",
    "rotation_deg",
    "tilt_angle_deg",
    "mirrored",
    "rotation_confidence",
    "tilt_confidence",
    "mirror_confidence",
    "overall_confidence",
    "needs_review",
    "tilt_applied",
    "status",
    "error",
    "output_path",
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
    log("=" * 70)
    log("Scanning Azure Blob")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log("=" * 70)

    folder_blobs: dict[str, list[str]] = {}
    try:
        for blob in container_client.list_blobs(name_starts_with=PREFIX):
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
    start_index = next((i for i, f in enumerate(folders) if f == START_FROM), None)
    if start_index is None:
        raise SystemExit(f"START_FROM folder not found under prefix: {START_FROM}")
    log(f"Starting at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def jpg_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_jpgs(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=jpg_sort_key)
    return images


def download_blob(container_client: ContainerClient, blob_name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(container_client.download_blob(blob_name).readall())
    return dest


def process_folder(
    writer: csv.DictWriter,
    csv_file,
    container_client: ContainerClient,
    detector: PageOrientationDetector,
    output_dir: Path,
    folder_name: str,
    blob_names: list[str],
) -> None:
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    log(f"  {len(blob_names)} images -> {out_folder}")

    with tempfile.TemporaryDirectory(prefix=f"rotation_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            row = {
                "folder": folder_name,
                "filename": filename,
                "status": "ok",
                "error": "",
            }
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                image = cv2.imread(str(local_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (unreadable/corrupt image)")

                result = detector.detect_result(image)
                corrected = detector.correct(image, result, expand=True)
                out_path = out_folder / filename
                if not cv2.imwrite(str(out_path), corrected):
                    raise ValueError(f"cv2.imwrite failed: {out_path}")

                tilt_applied = (
                    abs(result.tilt) >= 0.15
                    and abs(result.tilt) <= (detector.max_tilt_to_apply or 1e9)
                )
                row.update(
                    {
                        "rotation_deg": result.rotation,
                        "tilt_angle_deg": result.tilt,
                        "mirrored": "Yes" if result.mirror else "No",
                        "rotation_confidence": result.rotation_confidence,
                        "tilt_confidence": result.tilt_confidence,
                        "mirror_confidence": result.mirror_confidence,
                        "overall_confidence": result.overall_confidence,
                        "needs_review": "Yes" if result.needs_review else "No",
                        "tilt_applied": "Yes" if tilt_applied else "No",
                        "output_path": str(out_path),
                    }
                )
                log(
                    f"    rot={result.rotation} tilt={result.tilt} "
                    f"mirror={row['mirrored']} conf={result.overall_confidence:.2f} "
                    f"review={row['needs_review']} -> {out_path.name}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["output_path"] = ""
                for key in CSV_FIELDS:
                    row.setdefault(key, "")
                log(f"    ERROR: {row['error']}")
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            csv_file.flush()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log("=== PAGE ORIENTATION (OpenCV PageOrientationDetector) ===")
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {OUTPUT_DIR}")
    log(f"CSV:    {CSV_PATH}")
    log(f"Max tilt applied: {MAX_TILT_TO_APPLY}°")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / JPG blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    detector = PageOrientationDetector(
        analysis_max_dimension=1800,
        max_tilt_to_apply=MAX_TILT_TO_APPLY,
        debug=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        for folder_name in chart_folders:
            images = list_jpgs(folder_blobs, folder_name)
            if not images:
                log(f"Folder: {folder_name} (no JPGs, skipped)")
                continue
            log(f"Folder: {folder_name}")
            process_folder(
                writer, csv_file, container_client, detector,
                OUTPUT_DIR, folder_name, images,
            )

    log(f"Done. Report: {CSV_PATH}")


if __name__ == "__main__":
    main()
