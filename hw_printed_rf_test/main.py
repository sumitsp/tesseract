#!/usr/bin/env python3
"""Printed vs handwritten page test pipeline (local or blob).

Edit RUN CONFIG, then from repo root:

    python hw_printed_rf_test/main.py
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

import cv2
from openpyxl import Workbook
from PIL import Image

_PKG_DIR = Path(__file__).resolve().parent
_PARENT = _PKG_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from hw_printed_rf_test.classifier import (  # noqa: E402
    classify_page_image,
    load_classifier,
)
from hw_printed_rf_test.page_classifier import (  # noqa: E402
    classify_page_convnext,
    classify_page_hybrid,
    load_hybrid_classifier,
    load_page_classifier,
)

# =============================================================================
# RUN CONFIG — edit these (no .env)
# =============================================================================

# "hybrid"        = page ConvNeXt + RF upgrade for filled forms (recommended)
# "page_convnext" = page model only
# "rf_regions"    = old prescription RandomForest region aggregator
CLASSIFIER_MODE = "hybrid"

# "local" = file or folder on disk   |   "blob" = Azure chart folders
INPUT_SOURCE = "local"

LOCAL_INPUT = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\input")
OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\hw_printed_rf_test_output")

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "imaging-pipeline"
PREFIX = "Raw_Input/Run1/Batch1/DEID_PNGs/"
START_FROM = ""

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# =============================================================================

HEADERS = [
    "Folder Name",
    "File Name",
    "Document Type",
    "Method",
    "P(Handwritten)",
    "Printed Area",
    "Handwritten Area",
    "Mixed Area",
    "Other Area",
    "Region Count",
    "Region Label Counts",
    "Status",
    "Error",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def connect_container():
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    return client.get_container_client(CONTAINER_NAME)


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


def load_bgr_from_bytes(data: bytes):
    arr = np_from_buffer(data)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    with Image.open(io.BytesIO(data)) as pil:
        pil = pil.convert("RGB")
        rgb = __import__("numpy").array(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def np_from_buffer(data: bytes):
    import numpy as np

    return np.frombuffer(data, dtype=np.uint8)


def save_report(workbook: Workbook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def append_result(sheet, folder: str, filename: str, result) -> None:
    status = "ERROR" if result.error else "OK"
    counts = ""
    if result.region_labels:
        counts = "; ".join(f"{k}={v}" for k, v in sorted(result.region_labels.items()))
    sheet.append(
        [
            folder,
            filename,
            result.document_type,
            result.method,
            result.p_handwritten,
            result.printed_area,
            result.handwritten_area,
            result.mixed_area,
            result.other_area,
            result.region_count,
            counts,
            status,
            result.error or "",
        ]
    )


def classify_one(image, mode: str, model) -> object:
    if mode == "rf_regions":
        return classify_page_image(image, clf=model)
    if mode == "hybrid":
        return classify_page_hybrid(image, bundle=model)
    return classify_page_convnext(image, bundle=model)


def run_local(model, mode: str, output_dir: Path, report_path: Path) -> None:
    images = list_local_images(LOCAL_INPUT)
    if not images:
        raise SystemExit(f"No images under: {LOCAL_INPUT}")
    root = LOCAL_INPUT.resolve() if LOCAL_INPUT.is_dir() else LOCAL_INPUT.resolve().parent

    wb = Workbook()
    sheet = wb.active
    sheet.title = "Pages"
    sheet.append(HEADERS)
    sheet.freeze_panes = "A2"
    save_report(wb, report_path)

    for image_path in images:
        relative = image_path.resolve().relative_to(root)
        folder = relative.parent.name if str(relative.parent) not in {".", ""} else root.name
        log(f"  {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            from hw_printed_rf_test.classifier import PageTypeResult

            result = PageTypeResult(
                document_type="UNCERTAIN",
                method=mode,
                p_handwritten=None,
                error=f"Unreadable: {image_path}",
            )
        else:
            result = classify_one(image, mode, model)
        append_result(sheet, folder, image_path.name, result)
        save_report(wb, report_path)
        log(f"    -> {result.document_type} p_hw={result.p_handwritten}")


def run_blob(model, mode: str, output_dir: Path, report_path: Path) -> None:
    prefix = PREFIX.strip().strip("/") + "/"
    start_from = START_FROM.strip().strip("/")
    container = connect_container()

    wb = Workbook()
    sheet = wb.active
    sheet.title = "Pages"
    sheet.append(HEADERS)
    sheet.freeze_panes = "A2"
    save_report(wb, report_path)

    with tempfile.TemporaryDirectory(prefix="hw_rf_") as tmp:
        tmp_path = Path(tmp)
        for blob in container.list_blobs(name_starts_with=prefix):
            relative = blob.name[len(prefix) :] if blob.name.startswith(prefix) else blob.name
            parts = relative.split("/")
            if len(parts) < 2:
                continue
            folder, filename = parts[0], parts[-1]
            if start_from and folder.lower() < start_from.lower():
                continue
            if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue

            log(f"  {blob.name}")
            try:
                data = container.download_blob(blob.name).readall()
                (tmp_path / filename).write_bytes(data)
                image = load_bgr_from_bytes(data)
                if image is None:
                    raise ValueError("decode failed")
                result = classify_one(image, mode, model)
            except Exception as exc:
                from hw_printed_rf_test.classifier import PageTypeResult

                result = PageTypeResult(
                    document_type="UNCERTAIN",
                    method=mode,
                    p_handwritten=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            append_result(sheet, folder, filename, result)
            save_report(wb, report_path)
            log(f"    -> {result.document_type} p_hw={result.p_handwritten}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    source = INPUT_SOURCE.strip().lower()
    mode = CLASSIFIER_MODE.strip().lower()
    if source not in {"local", "blob"}:
        print('INPUT_SOURCE must be "local" or "blob"', file=sys.stderr)
        return 1
    if mode not in {"page_convnext", "rf_regions", "hybrid"}:
        print(
            'CLASSIFIER_MODE must be "hybrid", "page_convnext", or "rf_regions"',
            file=sys.stderr,
        )
        return 1

    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hw_printed_rf_report.xlsx"

    log(f"CLASSIFIER_MODE={mode}")
    if mode == "rf_regions":
        log("Loading region RandomForest model...")
        model = load_classifier()
    elif mode == "hybrid":
        log("Loading hybrid page ConvNeXt + RF models...")
        model = load_hybrid_classifier()
    else:
        log("Loading page ConvNeXt model...")
        model = load_page_classifier()

    log(f"INPUT_SOURCE={source}")
    log(f"Excel: {report_path}")

    if source == "local":
        run_local(model, mode, output_dir, report_path)
    else:
        if not CONTAINER_NAME or CONTAINER_NAME == "YOUR_CONTAINER_NAME":
            print("ERROR: set CONTAINER_NAME in main.py RUN CONFIG", file=sys.stderr)
            return 1
        run_blob(model, mode, output_dir, report_path)

    log(f"Done. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
