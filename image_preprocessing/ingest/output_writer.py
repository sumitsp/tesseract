"""Output directory layout and corrected-page naming."""

from __future__ import annotations

from pathlib import Path

from image_preprocessing.results import LoadedPage
from image_preprocessing.utils.image_utils import save_png


class OutputLayout:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.corrected_pages = self.root / "corrected_pages"
        self.report = self.root / "report"
        self.debug = self.root / "debug"

    def create(self) -> None:
        self.corrected_pages.mkdir(parents=True, exist_ok=True)
        self.report.mkdir(parents=True, exist_ok=True)
        self.debug.mkdir(parents=True, exist_ok=True)

    def debug_page_dir(self, page: LoadedPage) -> Path:
        name = f"{_safe_stem(page.document_name)}_page_{page.page_number:03d}"
        path = self.debug / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def corrected_page_filename(page: LoadedPage) -> str:
    stem = _safe_stem(page.document_name)
    multi_page = (
        page.input_format.lower() in {"pdf", "tif", "tiff"}
        or int(page.extras.get("frame_count", 1) or 1) > 1
        or page.page_number > 1
    )
    if multi_page:
        return f"{stem}_page_{page.page_number:03d}.png"
    return f"{stem}_corrected.png"


def write_corrected_page(
    layout: OutputLayout,
    page: LoadedPage,
    image,
    output_dpi: float | None,
) -> Path:
    filename = corrected_page_filename(page)
    dest = layout.corrected_pages / filename
    save_png(dest, image, dpi=output_dpi)
    return dest


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    return cleaned.strip("_") or "document"
