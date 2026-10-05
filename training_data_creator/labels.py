"""Local CSV of labels. One row per folder + image; a new label replaces that row.

Each page has a type, a visibility, and the share of the page inside the drawn box.
"""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

COLUMNS = [
    "folder name",
    "image name",
    "value",
    "visibility",
    "handwritten percent",
    "box left",
    "box top",
    "box width",
    "box height",
]
TYPES = {"Handwritten", "Printed", "Form", "Visual", "Blank", "Uncertain"}
VISIBILITY = {"Visible", "Not visible"}


def format_percent(value: float) -> str:
    number = float(value)
    if number < 0 or number > 100:
        raise ValueError(f"handwritten percent must be from 0 to 100, got {value!r}")
    if number == 0:
        return "0"
    text = f"{number:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def format_fraction(value: float) -> str:
    number = float(value)
    if number < 0 or number > 1:
        raise ValueError(f"box fraction must be from 0 to 1, got {value!r}")
    return f"{number:.4f}".rstrip("0").rstrip(".") or "0"


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, str]] = []
        self._index: dict[tuple[str, str], int] = {}
        self._load()

    def _blank(self, folder: str, image_name: str) -> dict[str, str]:
        return {
            "folder name": folder,
            "image name": image_name,
            "value": "",
            "visibility": "",
            "handwritten percent": "",
            "box left": "",
            "box top": "",
            "box width": "",
            "box height": "",
        }

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                folder = (row.get("folder name") or "").strip()
                name = (row.get("image name") or "").strip()
                value = (row.get("value") or "").strip()
                visibility = (row.get("visibility") or "").strip()
                if not folder or not name:
                    continue
                if value and value not in TYPES:
                    continue
                if visibility and visibility not in VISIBILITY:
                    continue
                record = self._blank(folder, name)
                record["value"] = value
                record["visibility"] = visibility
                percent = (row.get("handwritten percent") or "").strip()
                try:
                    record["handwritten percent"] = format_percent(percent) if percent else ""
                except ValueError:
                    record["handwritten percent"] = ""
                for column in ("box left", "box top", "box width", "box height"):
                    raw = (row.get(column) or "").strip()
                    try:
                        record[column] = format_fraction(raw) if raw else ""
                    except ValueError:
                        record[column] = ""
                if record["handwritten percent"] in {"", "0"} and not record["box width"]:
                    record["handwritten percent"] = ""
                    for column in ("box left", "box top", "box width", "box height"):
                        record[column] = ""
                key = (folder, name)
                if key in self._index:
                    self._rows[self._index[key]] = record
                else:
                    self._index[key] = len(self._rows)
                    self._rows.append(record)

    def get(self, folder: str, image_name: str) -> dict[str, str] | None:
        with self._lock:
            pos = self._index.get((folder, image_name))
            return None if pos is None else dict(self._rows[pos])

    def count(self) -> int:
        with self._lock:
            return len(self._rows)

    def has_row(self, folder: str, image_name: str) -> bool:
        return self.get(folder, image_name) is not None

    def replace(
        self,
        folder: str,
        image_name: str,
        *,
        value: str | None,
        visibility: str | None,
        handwritten_percent: float | None,
        box: dict[str, float] | None,
    ) -> dict[str, str]:
        """Write the whole mark for one page. Any field may be empty."""
        if value is not None and value not in TYPES:
            raise ValueError(f"value must be one of {sorted(TYPES)}, got {value!r}")
        if visibility is not None and visibility not in VISIBILITY:
            raise ValueError(f"visibility must be Visible or Not visible, got {visibility!r}")
        with self._lock:
            key = (folder, image_name)
            pos = self._index.get(key)
            created = pos is None
            if created:
                self._index[key] = len(self._rows)
                self._rows.append(self._blank(folder, image_name))
                pos = self._index[key]
            previous = dict(self._rows[pos])
            row = self._rows[pos]
            row["value"] = value or ""
            row["visibility"] = visibility or ""
            if handwritten_percent is None:
                row["handwritten percent"] = ""
                box = None
            else:
                row["handwritten percent"] = format_percent(handwritten_percent)
                if row["handwritten percent"] == "0":
                    box = None
            if box is None:
                for column in ("box left", "box top", "box width", "box height"):
                    row[column] = ""
            else:
                row["box left"] = format_fraction(box["left"])
                row["box top"] = format_fraction(box["top"])
                row["box width"] = format_fraction(box["width"])
                row["box height"] = format_fraction(box["height"])
            try:
                self._write()
            except Exception:
                if created:
                    self._rows.pop()
                    del self._index[key]
                else:
                    self._rows[pos] = previous
                raise
            return dict(row)

    def update(
        self,
        folder: str,
        image_name: str,
        *,
        value: str | None = None,
        visibility: str | None = None,
        handwritten_percent: float | None = None,
        box: dict[str, float] | None = None,
        clear_box: bool = False,
    ) -> dict[str, str]:
        if value is not None and value not in TYPES:
            raise ValueError(f"value must be one of {sorted(TYPES)}, got {value!r}")
        if visibility is not None and visibility not in VISIBILITY:
            raise ValueError(f"visibility must be Visible or Not visible, got {visibility!r}")
        if value is None and visibility is None and handwritten_percent is None and not clear_box:
            raise ValueError("nothing to save")
        with self._lock:
            key = (folder, image_name)
            pos = self._index.get(key)
            created = pos is None
            if created:
                self._index[key] = len(self._rows)
                self._rows.append(self._blank(folder, image_name))
                pos = self._index[key]
            previous = dict(self._rows[pos])
            row = self._rows[pos]
            if value is not None:
                row["value"] = value
            if visibility is not None:
                row["visibility"] = visibility
            if clear_box:
                row["handwritten percent"] = ""
                for column in ("box left", "box top", "box width", "box height"):
                    row[column] = ""
            elif handwritten_percent is not None:
                row["handwritten percent"] = format_percent(handwritten_percent)
                if box is None or row["handwritten percent"] == "0":
                    for column in ("box left", "box top", "box width", "box height"):
                        row[column] = ""
                else:
                    row["box left"] = format_fraction(box["left"])
                    row["box top"] = format_fraction(box["top"])
                    row["box width"] = format_fraction(box["width"])
                    row["box height"] = format_fraction(box["height"])
            try:
                self._write()
            except Exception:
                if created:
                    self._rows.pop()
                    del self._index[key]
                else:
                    self._rows[pos] = previous
                raise
            return dict(row)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        os.replace(tmp, self.path)
