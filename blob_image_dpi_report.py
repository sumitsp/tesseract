#!/usr/bin/env python3
"""Inspect Azure Blob images and continuously write dimensions/DPI to Excel."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from openpyxl import Workbook
from PIL import Image

from image_preprocessing.config import RASTER_EXTENSIONS
from image_preprocessing.ingest.image_loader import read_embedded_dpi

# Edit these values, then run: python blob_image_dpi_report.py
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "imaging-pipeline"
PREFIX = "Raw_Input/Run1/Batch1/DEID_PNGs/"
START_FROM = ""
OUTPUT_EXCEL = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\blob_image_dpi_report.xlsx")

HEADERS = [
    "Folder Name",
    "File Name",
    "Blob Path",
    "Width (px)",
    "Height (px)",
    "DPI X",
    "DPI Y",
    "DPI",
    "DPI Source",
    "File Size (bytes)",
    "Status",
    "Error",
]


def connect_container():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    return service.get_container_client(CONTAINER_NAME)


def extract_dpi(image: Image.Image) -> tuple[float | None, float | None, float | None, str]:
    dpi = image.info.get("dpi")
    if isinstance(dpi, tuple) and dpi:
        x = float(dpi[0]) if dpi[0] else None
        y = float(dpi[1]) if len(dpi) > 1 and dpi[1] else x
        valid = [v for v in (x, y) if v is not None and v > 1]
        if valid:
            return x, y, sum(valid) / len(valid), "embedded"
    average, source = read_embedded_dpi(image)
    return average, average, average, source


def save_report(workbook: Workbook) -> None:
    OUTPUT_EXCEL.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT_EXCEL)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    prefix = PREFIX.strip().strip("/") + "/"
    start_from = START_FROM.strip().strip("/")
    container = connect_container()

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Images"
    sheet.append(HEADERS)
    sheet.freeze_panes = "A2"
    save_report(workbook)

    count = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        relative = blob.name[len(prefix) :] if blob.name.startswith(prefix) else blob.name
        parts = relative.split("/")
        if len(parts) < 2:
            continue
        folder = parts[0]
        filename = parts[-1]
        if start_from and folder.lower() < start_from.lower():
            continue
        if Path(filename).suffix.lower() not in RASTER_EXTENSIONS:
            continue

        row = [folder, filename, blob.name, None, None, None, None, None, "unknown",
               getattr(blob, "size", None), "OK", ""]
        try:
            data = container.download_blob(blob.name).readall()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                dpi_x, dpi_y, dpi, source = extract_dpi(image)
            row[3:9] = [width, height, dpi_x, dpi_y, dpi, source]
        except Exception as exc:
            row[10] = "ERROR"
            row[11] = f"{type(exc).__name__}: {exc}"
            logging.exception("Failed: %s", blob.name)

        sheet.append(row)
        count += 1
        save_report(workbook)
        logging.info("%s | %sx%s | DPI=%s", blob.name, row[3], row[4], row[7])

    sheet.auto_filter.ref = sheet.dimensions
    save_report(workbook)
    print(f"Done: {count} image(s)")
    print(f"Excel: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
