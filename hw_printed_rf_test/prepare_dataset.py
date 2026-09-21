#!/usr/bin/env python3
"""Build a binary page dataset from training_data/.

Printed  ← every PDF page under training_data/printed/**  (signatures stay PRINTED)
Handwritten ← images under training_data/handwritten/**
Extra HW ← RF region crops (Handwritten_extended) cut from those handwritten pages

Edit RUN CONFIG, then:

    python prepare_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hw_printed_rf_test.classifier import (  # noqa: E402
    detect_and_classify_regions,
    load_classifier,
)

# =============================================================================
# RUN CONFIG
# =============================================================================

TRAINING_DATA = _PKG / "training_data"
PRINTED_ROOT = TRAINING_DATA / "printed"
HANDWRITTEN_ROOT = TRAINING_DATA / "handwritten"

OUT_DIR = _PKG / "prepared_dataset"
PDF_DPI = 120
# Cap printed pages so ~8k EHR pages do not drown ~400 handwritten images.
MAX_PRINTED_PAGES = 2500
# Save RF handwritten region crops as extra HANDWRITTEN samples.
EXTRACT_RF_HW_CROPS = False
MIN_CROP_AREA = 400
SEED = 42
# Resume-friendly: skip PDF page files that already exist.
SKIP_EXISTING_PDF_PAGES = True
MAX_RF_HW_CROPS = 1500

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def _rel_key(path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def iter_pdfs(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
        and not p.name.startswith("._")
    )


def iter_images(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not p.name.startswith("._")
    )


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(len(doc)):
            out = out_dir / f"{_rel_key(pdf_path)}_p{i + 1:03d}.png"
            if SKIP_EXISTING_PDF_PAGES and out.is_file() and out.stat().st_size > 0:
                written.append(out)
                continue
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            # Write via bytes to avoid Windows file-lock issues with pix.save().
            tmp = out.with_suffix(".png.tmp")
            try:
                tmp.write_bytes(pix.tobytes("png"))
                if out.exists():
                    try:
                        out.unlink()
                    except OSError:
                        pass
                tmp.replace(out)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            if out.is_file() and out.stat().st_size > 0:
                written.append(out)
    finally:
        doc.close()
    return written


def copy_image_as_png(src: Path, out_dir: Path, prefix: str) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return None
    out = out_dir / f"{prefix}_{_rel_key(src)}.png"
    cv2.imwrite(str(out), image)
    return out


def extract_rf_hw_crops(image_path: Path, clf, out_dir: Path) -> list[Path]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return []
    regions = detect_and_classify_regions(image, clf)
    # Work on the same resized canvas the detector used.
    hgt, wdt = image.shape[:2]
    h_bw = hgt / float(wdt) if wdt else 1.0
    dim = (576, max(1, int(576 * h_bw)))
    resized = cv2.resize(image, dim)

    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (label, area, (x, y, w, h)) in enumerate(regions):
        if label != "Handwritten_extended" or area < MIN_CROP_AREA:
            continue
        crop = resized[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        out = out_dir / f"{_rel_key(image_path)}_rf{idx:03d}.png"
        cv2.imwrite(str(out), crop)
        saved.append(out)
    return saved


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "source", "origin"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    random.seed(SEED)
    printed_out = OUT_DIR / "printed"
    hw_out = OUT_DIR / "handwritten"
    rf_out = OUT_DIR / "handwritten_rf_crops"
    for d in (printed_out, hw_out, rf_out):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # --- PRINTED from PDFs ---
    pdfs = iter_pdfs(PRINTED_ROOT)
    log(f"Found {len(pdfs)} printed PDFs under {PRINTED_ROOT}")
    printed_paths: list[Path] = []
    for i, pdf in enumerate(pdfs, start=1):
        log(f"[{i}/{len(pdfs)}] render {pdf.relative_to(PRINTED_ROOT)}")
        try:
            pages = render_pdf_pages(pdf, printed_out, PDF_DPI)
        except Exception as exc:
            log(f"  SKIP PDF error: {exc}")
            continue
        printed_paths.extend(pages)

    if len(printed_paths) > MAX_PRINTED_PAGES:
        printed_paths = random.sample(printed_paths, MAX_PRINTED_PAGES)
        log(f"Subsampled printed pages to {MAX_PRINTED_PAGES}")

    for p in printed_paths:
        rows.append(
            {
                "path": str(p.relative_to(OUT_DIR)),
                "label": "Printed",
                "source": "pdf",
                "origin": str(p.name),
            }
        )

    # --- HANDWRITTEN full pages ---
    hw_images = iter_images(HANDWRITTEN_ROOT)
    log(f"Found {len(hw_images)} handwritten images under {HANDWRITTEN_ROOT}")
    for src in hw_images:
        out = copy_image_as_png(src, hw_out, "page")
        if out is None:
            log(f"  SKIP unreadable {src}")
            continue
        rows.append(
            {
                "path": str(out.relative_to(OUT_DIR)),
                "label": "Handwritten",
                "source": "handwritten_folder",
                "origin": str(src.relative_to(HANDWRITTEN_ROOT)),
            }
        )

    # --- EXTRA HANDWRITTEN from RF region crops on HW pages ---
    if EXTRACT_RF_HW_CROPS and hw_images:
        log("Loading RF model for extra handwritten crops...")
        clf = load_classifier()
        crop_count = 0
        for src in hw_images:
            crops = extract_rf_hw_crops(src, clf, rf_out)
            for crop in crops:
                rows.append(
                    {
                        "path": str(crop.relative_to(OUT_DIR)),
                        "label": "Handwritten",
                        "source": "rf_hw_crop",
                        "origin": str(src.relative_to(HANDWRITTEN_ROOT)),
                    }
                )
                crop_count += 1
        log(f"Saved {crop_count} RF handwritten crops")

    manifest = OUT_DIR / "manifest.csv"
    write_manifest(rows, manifest)

    n_print = sum(1 for r in rows if r["label"] == "Printed")
    n_hw = sum(1 for r in rows if r["label"] == "Handwritten")
    log(f"Done. Printed={n_print} Handwritten={n_hw}")
    log(f"Manifest: {manifest}")
    log("Extracted OCR text under printed/*/ocr was not required for image training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
