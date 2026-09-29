"""Local CSV of labels. One row per folder + image; a new label replaces that row."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

COLUMNS = ["folder name", "image name", "value"]
VALUES = {"Handwritten", "Printed"}


class LabelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[dict[str, str]] = []
        self._index: dict[tuple[str, str], int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                folder = (row.get("folder name") or "").strip()
                name = (row.get("image name") or "").strip()
                value = (row.get("value") or "").strip()
                if not folder or not name or value not in VALUES:
                    continue
                key = (folder, name)
                if key in self._index:
                    self._rows[self._index[key]]["value"] = value
                else:
                    self._index[key] = len(self._rows)
                    self._rows.append({"folder name": folder, "image name": name, "value": value})

    def get(self, folder: str, image_name: str) -> str | None:
        with self._lock:
            pos = self._index.get((folder, image_name))
            return None if pos is None else self._rows[pos]["value"]

    def count(self) -> int:
        with self._lock:
            return len(self._rows)

    def upsert(self, folder: str, image_name: str, value: str) -> None:
        if value not in VALUES:
            raise ValueError(f"value must be Handwritten or Printed, got {value!r}")
        with self._lock:
            key = (folder, image_name)
            pos = self._index.get(key)
            if pos is None:
                self._index[key] = len(self._rows)
                self._rows.append({"folder name": folder, "image name": image_name, "value": value})
                try:
                    self._write()
                except Exception:
                    self._rows.pop()
                    del self._index[key]
                    raise
            else:
                old = self._rows[pos]["value"]
                self._rows[pos]["value"] = value
                try:
                    self._write()
                except Exception:
                    self._rows[pos]["value"] = old
                    raise

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        os.replace(tmp, self.path)
