"""Live CSV report: one row per page, flushed after every page."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = [
    "Folder Name",
    "File Name",
    "Page",
    "Input DPI",
    "DPI Source",
    "Width px",
    "Height px",
    "Quality Score",
    "Quality",
    "Quality Warning",
    "Sharpness",
    "Blur",
    "Contrast",
    "Noise",
    "Brightness",
    "Clipping",
    "Text Visibility",
    "Compression",
    "Usable Dimension",
    "Effective Resolution",
    "Document Type",
    "P(Handwritten)",
    "Visibility",
    "Handwritten Area %",
    "Document Type Method",
    "Final Status",
    "Error",
    "Time taken (s)",
]


class CsvReport:
    def __init__(self, path: Path, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: list[dict[str, Any]] = []
        if not (resume and self.path.is_file()):
            # utf-8-sig so Excel opens the file with the right encoding.
            with self.path.open("w", newline="", encoding="utf-8-sig") as fh:
                csv.writer(fh).writerow(COLUMNS)

    def add(self, row: dict[str, Any]) -> None:
        self._pending.append(row)
        self.save()

    def save(self) -> None:
        if not self._pending:
            return
        try:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
                writer.writerows(self._pending)
        except PermissionError:
            # Excel locks the file while it is open; rows are kept and written on the next page.
            return
        self._pending.clear()
