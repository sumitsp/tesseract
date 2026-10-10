"""Load document pages from local files or Azure Blob chart folders.

Rasters (JPG, PNG, TIFF, ...) keep their embedded DPI; multi-page TIFFs yield one
page per frame. PDF pages render at the target DPI (or the embedded scan DPI).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger(__name__)

RASTER_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".jfif",
     ".pbm", ".pgm", ".ppm", ".pnm"}
)
PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class Page:
    folder: str
    file_name: str
    page_number: int
    image: np.ndarray  # BGR
    input_dpi: float | None
    dpi_source: str
    warnings: list[str] = field(default_factory=list)


def read_embedded_dpi(image: Image.Image) -> tuple[float | None, str]:
    """Return (dpi, source) from file metadata. Never invents a DPI value."""
    info = image.info or {}
    dpi = info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) >= 1:
        positive = [float(v) for v in dpi if v and float(v) > 1.0]
        if positive:
            return float(sum(positive) / len(positive)), "embedded"

    jfif = info.get("jfif_density")
    if jfif and info.get("jfif_unit") == 1:
        vals = [float(v) for v in (jfif if isinstance(jfif, tuple) else (jfif,))]
        positive = [v for v in vals if v > 1.0]
        if positive:
            return float(sum(positive) / len(positive)), "embedded"

    try:
        tags = image.tag_v2 if hasattr(image, "tag_v2") else {}
        x_res, y_res, unit = tags.get(282), tags.get(283), tags.get(296)
        if x_res and y_res:
            xr = float(x_res[0]) / float(x_res[1]) if isinstance(x_res, tuple) else float(x_res)
            yr = float(y_res[0]) / float(y_res[1]) if isinstance(y_res, tuple) else float(y_res)
            dpi_val = (xr + yr) / 2.0
            if unit == 3:  # pixels/cm
                dpi_val *= 2.54
            if dpi_val > 1.0:
                return float(dpi_val), "embedded"
    except Exception:
        pass
    return None, "unknown"


def _raster_pages(data: bytes, folder: str, file_name: str) -> list[Page]:
    pages: list[Page] = []
    with Image.open(io.BytesIO(data)) as master:
        n_frames = max(1, int(getattr(master, "n_frames", 1) or 1))
        for idx in range(n_frames):
            # seek + copy inside the loop; Pillow reuses one object across frames.
            master.seek(idx)
            frame = master.copy()
            frame.load()
            frame = ImageOps.exif_transpose(frame) or frame
            dpi, dpi_source = read_embedded_dpi(frame)
            rgb = np.array(frame.convert("RGB"))
            warnings = [] if dpi is not None else ["DPI_UNKNOWN: Raster DPI metadata unavailable"]
            pages.append(Page(folder, file_name, idx + 1, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              dpi, dpi_source, warnings))
    return pages


def _embedded_image_dpi(page) -> tuple[float | None, float]:
    """Median DPI of embedded images and the fraction of page area they cover."""
    try:
        images = page.get_images(full=True)
    except Exception:
        return None, 0.0
    page_area = float(abs(page.rect.width * page.rect.height)) or 1.0
    dpis: list[float] = []
    covered = 0.0
    for xref in {int(img[0]) for img in images}:
        try:
            meta = page.parent.extract_image(xref)
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            inch_w, inch_h = float(rect.width) / 72.0, float(rect.height) / 72.0
            if inch_w < 0.2 or inch_h < 0.2:
                continue
            dpi = min(float(meta["width"]) / inch_w, float(meta["height"]) / inch_h)
            if dpi > 1.0:
                dpis.append(dpi)
            covered += float(abs(rect.width * rect.height))
    if not dpis:
        return None, 0.0
    return float(np.median(dpis)), min(1.0, covered / page_area)


def _pdf_pages(data: bytes, folder: str, file_name: str, target_dpi: int) -> list[Page]:
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            scan_dpi, coverage = _embedded_image_dpi(page)
            if scan_dpi is None or coverage < 0.80:
                render_dpi, input_dpi, source = target_dpi, float(target_dpi), "pdf_vector_render"
            else:
                render_dpi = min(target_dpi, max(72, int(round(scan_dpi))))
                input_dpi, source = scan_dpi, "pdf_raster_estimate"
            zoom = render_dpi / 72.0
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, annots=True)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            code = {1: cv2.COLOR_GRAY2BGR, 3: cv2.COLOR_RGB2BGR, 4: cv2.COLOR_RGBA2BGR}[pix.n]
            pages.append(Page(folder, file_name, idx, cv2.cvtColor(arr, code), input_dpi, source))
    return pages


def load_pages(data: bytes, folder: str, file_name: str, *, target_dpi: int) -> list[Page]:
    suffix = Path(file_name).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return _pdf_pages(data, folder, file_name, target_dpi)
    if suffix in RASTER_EXTENSIONS:
        return _raster_pages(data, folder, file_name)
    raise ValueError(f"Unsupported file type: {suffix}")


def _natural_key(name: str) -> tuple:
    stem = Path(name).stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem.lower())


def local_files(input_path: Path) -> list[tuple[str, str, Path]]:
    """(folder, file_name, path) for every supported file under input_path."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LOCAL_INPUT does not exist: {path}")
    if path.is_file():
        files = [path]
    else:
        files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort(key=lambda p: (str(p.parent).lower(), _natural_key(p.name)))
    return [(p.parent.name, p.name, p) for p in files]


def connect_container(storage_account: str, container_name: str) -> Any:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    account_url = f"https://{storage_account}.blob.core.windows.net"
    client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
    return client.get_container_client(container_name)


def blob_files(container: Any, prefix: str, start_from: str = "") -> Iterator[tuple[str, str, str]]:
    """(folder, file_name, blob_name) per chart folder under prefix, folders in name order."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    by_folder: dict[str, list[str]] = {}
    for blob in container.list_blobs(name_starts_with=prefix):
        parts = blob.name[len(prefix):].split("/")
        if len(parts) < 2 or Path(parts[-1]).suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        by_folder.setdefault(parts[0], []).append(blob.name)
    folders = sorted(by_folder, key=str.lower)
    if start_from:
        if start_from not in folders:
            raise FileNotFoundError(f"START_FROM folder not found under prefix: {start_from}")
        LOGGER.info("Starting at %s; skipping %s earlier folder(s)", start_from, folders.index(start_from))
        folders = folders[folders.index(start_from):]
    for folder in folders:
        for name in sorted(by_folder[folder], key=lambda n: _natural_key(Path(n).name)):
            yield folder, Path(name).name, name
