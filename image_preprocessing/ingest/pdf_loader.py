"""Load PDF pages. Vector pages render at target DPI; raster pages keep source DPI."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from image_preprocessing.config import PDF_EXTENSIONS
from image_preprocessing.results import LoadedPage

LOGGER = logging.getLogger(__name__)


def _estimate_embedded_image_dpi(page) -> tuple[float | None, float]:
    """Median DPI of embedded images and the fraction of page area they cover."""
    try:
        images = page.get_images(full=True)
    except Exception:
        return None, 0.0
    if not images:
        return None, 0.0

    page_area = float(abs(page.rect.width * page.rect.height)) or 1.0
    dpis: list[float] = []
    covered = 0.0
    seen_xrefs: set[int] = set()
    for img in images:
        xref = int(img[0])
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            meta = page.parent.extract_image(xref)
            pix_w = float(meta["width"])
            pix_h = float(meta["height"])
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            inch_w = float(rect.width) / 72.0
            inch_h = float(rect.height) / 72.0
            if inch_w < 0.2 or inch_h < 0.2:
                continue
            dpi = min(pix_w / inch_w, pix_h / inch_h)
            if dpi > 1.0:
                dpis.append(float(dpi))
            covered += float(abs(rect.width * rect.height))
    if not dpis:
        return None, 0.0
    return float(np.median(np.array(dpis, dtype=np.float64))), min(1.0, covered / page_area)


def _choose_render_dpi(
    estimated_dpi: float | None,
    coverage: float,
    target_dpi: int,
    low_dpi_threshold: int,
) -> tuple[int, float | None, str, str, list[str]]:
    """Return (render_dpi, reported_input_dpi, dpi_source, source_kind, warnings)."""
    warnings: list[str] = []
    if estimated_dpi is None or coverage < 0.80:
        # Vector / mixed: render at the pipeline target unless the page is
        # clearly a full-page photograph already.
        return (
            int(target_dpi),
            float(target_dpi),
            "pdf_vector_render",
            "pdf_vector" if estimated_dpi is None else "pdf_mixed",
            warnings,
        )

    src = float(estimated_dpi)
    if src > target_dpi:
        return int(target_dpi), src, "pdf_raster_estimate", "pdf_raster", warnings
    if src >= low_dpi_threshold:
        return max(72, int(round(src))), src, "pdf_raster_estimate", "pdf_raster", warnings
    warnings.append("DPI_WARNING: Input DPI below 150")
    warnings.append("LOW_DPI")
    return max(72, int(round(src))), src, "pdf_raster_estimate", "pdf_raster", warnings


def _pixmap_to_bgr(pix) -> np.ndarray:
    import cv2

    samples = pix.samples
    arr = np.frombuffer(samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported pixmap channel count: {pix.n}")


def load_pdf_pages(
    path: Path,
    *,
    document_index: int,
    target_dpi: int,
    low_dpi_threshold: int,
    document_name: str | None = None,
) -> list[LoadedPage]:
    import fitz

    path = path.resolve()
    document_name = document_name or path.name
    pages: list[LoadedPage] = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            estimated_dpi, coverage = _estimate_embedded_image_dpi(page)
            render_dpi, input_dpi, dpi_source, source_kind, warnings = _choose_render_dpi(
                estimated_dpi, coverage, target_dpi, low_dpi_threshold
            )
            zoom = render_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False, annots=True)
            image = _pixmap_to_bgr(pix)
            pages.append(
                LoadedPage(
                    document_name=document_name,
                    document_index=document_index,
                    page_number=idx,
                    input_file=str(path),
                    input_path=path,
                    input_format="pdf",
                    image=image,
                    input_dpi=input_dpi,
                    dpi_source=dpi_source,
                    source_kind=source_kind,
                    warnings=warnings,
                    extras={
                        "pdf_page_width_pt": float(page.rect.width),
                        "pdf_page_height_pt": float(page.rect.height),
                        "pdf_image_coverage": coverage,
                        "pdf_estimated_raster_dpi": estimated_dpi,
                        "pdf_render_dpi": render_dpi,
                    },
                )
            )
    LOGGER.debug("Loaded %s PDF page(s) from %s", len(pages), path)
    return pages


def is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS
