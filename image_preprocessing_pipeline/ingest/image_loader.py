"""Load raster document pages (JPG, PNG, TIFF, and other common formats)."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

from image_preprocessing.config import RASTER_EXTENSIONS
from image_preprocessing.results import LoadedPage
from image_preprocessing.utils.image_utils import pil_to_bgr

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger(__name__)


def read_embedded_dpi(image: Image.Image) -> tuple[float | None, str]:
    """Return (dpi, source) from file metadata. Never invents a DPI value."""
    info = image.info or {}
    dpi = info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) >= 1:
        vals = [float(v) for v in dpi if v]
        positive = [v for v in vals if v > 1.0]
        if positive:
            return float(sum(positive) / len(positive)), "embedded"

    jfif = info.get("jfif_density")
    jfif_unit = info.get("jfif_unit")
    if jfif and jfif_unit == 1:
        vals = [float(v) for v in (jfif if isinstance(jfif, tuple) else (jfif,))]
        positive = [v for v in vals if v > 1.0]
        if positive:
            return float(sum(positive) / len(positive)), "embedded"

    # TIFF x/y resolution tags.
    try:
        x_res = image.tag_v2.get(282) if hasattr(image, "tag_v2") else None
        y_res = image.tag_v2.get(283) if hasattr(image, "tag_v2") else None
        unit = image.tag_v2.get(296) if hasattr(image, "tag_v2") else None
        if x_res and y_res:
            xr = float(x_res[0]) / float(x_res[1]) if isinstance(x_res, tuple) else float(x_res)
            yr = float(y_res[0]) / float(y_res[1]) if isinstance(y_res, tuple) else float(y_res)
            dpi = (xr + yr) / 2.0
            if unit == 3:  # pixels/cm
                dpi *= 2.54
            if dpi > 1.0:
                return float(dpi), "embedded"
    except Exception:
        pass
    return None, "unknown"


def _frame_count(image: Image.Image) -> int:
    n = getattr(image, "n_frames", 1)
    try:
        return max(1, int(n))
    except (TypeError, ValueError):
        return 1


def load_raster_pages(
    path: Path,
    *,
    document_index: int,
    document_name: str | None = None,
) -> list[LoadedPage]:
    """Load every page/frame of a raster file.

    Important for multi-page TIFF
    -----------------------------
    Do **not** materialise ``list(ImageSequence.Iterator(im))`` and copy later.
    Pillow reuses one Image object and only ``seek()`` changes the active
    frame, so that pattern silently turns every page into the **last** frame.

    Correct pattern: ``seek(i)`` then ``copy()`` immediately inside the loop.
    """
    path = path.resolve()
    document_name = document_name or path.name
    pages: list[LoadedPage] = []
    with Image.open(path) as master:
        n_frames = _frame_count(master)
        for idx in range(n_frames):
            # Seek + independent copy before reading pixels / converting.
            master.seek(idx)
            frame = master.copy()
            frame.load()  # force decode of this frame now
            frame = ImageOps.exif_transpose(frame) or frame
            if frame.mode not in {"RGB", "L"}:
                frame = frame.convert("RGB")
            dpi, dpi_source = read_embedded_dpi(frame)
            # Own contiguous array — never share buffers across pages.
            image = pil_to_bgr(frame).copy()
            warnings: list[str] = []
            if dpi is None:
                warnings.append("DPI_UNKNOWN: Raster DPI metadata unavailable")
            pages.append(
                LoadedPage(
                    document_name=document_name,
                    document_index=document_index,
                    page_number=idx + 1,
                    input_file=str(path),
                    input_path=path,
                    input_format=path.suffix.lower().lstrip(".") or "unknown",
                    image=image,
                    input_dpi=dpi,
                    dpi_source=dpi_source,
                    source_kind="raster",
                    warnings=warnings,
                    extras={"frame_count": n_frames},
                )
            )
    if n_frames > 1:
        LOGGER.info("Loaded %s multi-page raster frames from %s", n_frames, path)
    else:
        LOGGER.debug("Loaded %s raster page(s) from %s", len(pages), path)
    return pages


def is_raster_file(path: Path) -> bool:
    return path.suffix.lower() in RASTER_EXTENSIONS
