"""Page quality + printed/handwritten pipeline.

For every page: engineering quality score (0-10, Good/Medium/Bad) and document type
(PRINTED / HANDWRITTEN / BLANK / UNCERTAIN) from the ConvNeXt-Tiny page model
plus handwriting ink evidence. Results go to a live CSV report.

Edit RUN CONFIG below, then run:  python main.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quality_hw import hw_printed
from quality_hw.pages import Page, blob_files, connect_container, load_pages, local_files
from quality_hw.quality import analyze_quality
from quality_hw.report import CsvReport

# ============================== RUN CONFIG ==============================
INPUT_SOURCE = "local"  # "local" or "blob"

# local: a file or a folder (searched recursively; the parent folder is the chart folder)
LOCAL_INPUT = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\input")

# blob: PREFIX/<chart folder>/<file>
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"
START_FROM = ""  # chart folder to resume from; appends to the existing report

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\quality_hw_output")
REPORT_NAME = "quality_hw_report.csv"

TARGET_DPI = 400  # PDF render DPI and the DPI quality is scored against
LOW_DPI_THRESHOLD = 150
QUALITY_REVIEW_THRESHOLD = 4.0  # quality below this is Bad -> REVIEW_REQUIRED
# ========================================================================

LOGGER = logging.getLogger("quality_hw")


def process_page(page: Page, bundle: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    h, w = page.image.shape[:2]
    row: dict[str, Any] = {
        "Folder Name": page.folder,
        "File Name": page.file_name,
        "Page": page.page_number,
        "Input DPI": round(page.input_dpi, 1) if page.input_dpi else None,
        "DPI Source": page.dpi_source,
        "Width px": w,
        "Height px": h,
    }
    try:
        q = analyze_quality(
            page.image,
            page.input_dpi,
            target_dpi=TARGET_DPI,
            low_dpi_threshold=LOW_DPI_THRESHOLD,
            quality_review_threshold=QUALITY_REVIEW_THRESHOLD,
        )
        bad = q.quality_score < QUALITY_REVIEW_THRESHOLD
        row.update({
            "Quality Score": round(q.quality_score, 2),
            "Quality": "Bad" if bad else "Good",
            "Quality Warning": "; ".join(dict.fromkeys(page.warnings + q.warnings)),
            "Sharpness": round(q.sharpness_score, 2),
            "Blur": round(q.blur_score, 2),
            "Contrast": round(q.contrast_score, 2),
            "Noise": round(q.noise_score, 2),
            "Brightness": round(q.brightness_score, 2),
            "Clipping": round(q.clipping_score, 2),
            "Text Visibility": round(q.text_visibility_score, 2),
            "Compression": round(q.compression_score, 2),
            "Usable Dimension": round(q.usable_dimension_score, 2),
            "Effective Resolution": round(q.effective_resolution_score, 2),
        })

        dt = hw_printed.classify_page_hybrid(page.image, bundle=bundle)
        doc_type = "BLANK" if dt.method == hw_printed.BLANK_METHOD else dt.document_type
        row.update({
            "Document Type": doc_type,
            "P(Handwritten)": dt.p_handwritten,
            "Visibility": dt.visibility,
            "Handwritten Area %": dt.handwritten_area,
            "Document Type Method": dt.method,
            "Error": dt.error,
        })
        if doc_type == "BLANK":
            # Sharpness/contrast of an empty sheet say nothing about readability.
            row["Quality"] = "N/A"
            bad = False
        review = bad or doc_type == "UNCERTAIN" or dt.visibility == "Not visible" or dt.error
        row["Final Status"] = "REVIEW_REQUIRED" if review else "OK"

        # Handwriting OCRs worse than print, so a clean handwritten page tops out at Medium.
        if doc_type == "HANDWRITTEN" and row["Quality"] == "Good":
            row["Quality"] = "Medium"
            note = "QUALITY_MEDIUM: handwritten page capped from Good"
            row["Quality Warning"] = "; ".join(filter(None, [row["Quality Warning"], note]))
    except Exception as exc:
        LOGGER.exception("Failed on %s/%s page %s", page.folder, page.file_name, page.page_number)
        row.update({"Final Status": "ERROR", "Error": f"{type(exc).__name__}: {exc}"})
    row["Time taken (s)"] = round(time.perf_counter() - started, 3)
    return row


def _error_row(folder: str, file_name: str, exc: Exception) -> dict[str, Any]:
    return {"Folder Name": folder, "File Name": file_name, "Final Status": "ERROR",
            "Error": f"Load failed: {type(exc).__name__}: {exc}"}


def _run(files, read_bytes, report: CsvReport, bundle: dict[str, Any]) -> int:
    n_pages = 0
    for folder, file_name, ref in files:
        try:
            pages = load_pages(read_bytes(ref), folder, file_name, target_dpi=TARGET_DPI)
        except Exception as exc:
            LOGGER.error("Could not load %s/%s: %s", folder, file_name, exc)
            report.add(_error_row(folder, file_name, exc))
            continue
        for page in pages:
            row = process_page(page, bundle)
            report.add(row)
            n_pages += 1
            LOGGER.info("%s/%s p%s  quality=%s  type=%s  %s", folder, file_name, page.page_number,
                        row.get("Quality Score"), row.get("Document Type"), row["Final Status"])
    return n_pages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report_path = OUTPUT_DIR / REPORT_NAME
    source = INPUT_SOURCE.strip().lower()

    LOGGER.info("Loading printed/handwritten model: %s", hw_printed.DEFAULT_PAGE_MODEL.name)
    bundle = hw_printed.load_hybrid_classifier()

    started = time.perf_counter()
    if source == "local":
        files = local_files(LOCAL_INPUT)
        report = CsvReport(report_path)
        n = _run(files, lambda p: Path(p).read_bytes(), report, bundle)
    elif source == "blob":
        container = connect_container(STORAGE_ACCOUNT, CONTAINER_NAME)
        report = CsvReport(report_path, resume=bool(START_FROM))
        files = blob_files(container, PREFIX, START_FROM)
        n = _run(files, lambda name: container.download_blob(name).readall(), report, bundle)
    else:
        raise ValueError(f'INPUT_SOURCE must be "local" or "blob", got {INPUT_SOURCE!r}')

    report.save()
    LOGGER.info("Done: %s page(s) in %.1fs -> %s", n, time.perf_counter() - started, report_path)


if __name__ == "__main__":
    main()
