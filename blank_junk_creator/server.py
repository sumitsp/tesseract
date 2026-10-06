"""Localhost labeling server. Images come from a local folder or blob; the CSV stays on this machine."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from blank_junk_creator.catalog import ImageItem
from blank_junk_creator.labels import SUBTYPES, VALUES, LabelStore

LOGGER = logging.getLogger("blank_junk_creator")
_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
ReadImage = Callable[[ImageItem], tuple[bytes, str]]

SUBTYPE_NAMES = {
    "JUNK_INVOICE": "Invoice",
    "JUNK_COVER_PAGE": "Cover page",
    "JUNK_RECORD_REQUEST": "Record request",
    "JUNK_INSTRUCTIONS": "Instructions",
    "JUNK_LETTER_FAX": "Letter / fax",
    "JUNK_OTHERS": "Others",
}


def _summary(row: dict[str, str] | None) -> str:
    if not row or not row.get("value"):
        return ""
    if row["value"] == "JUNK" and row.get("subtype"):
        return "JUNK / " + SUBTYPE_NAMES.get(row["subtype"], row["subtype"])
    return row["value"]


class App:
    def __init__(self, store: LabelStore, read_image: ReadImage | None = None) -> None:
        self.store = store
        self.read_image = read_image
        self.items: list[ImageItem] = []
        self.ready = False
        self.error: str | None = None
        self._by_key: dict[tuple[str, str], int] = {}
        self._folders: list[dict] = []

    def set_items(self, items: list[ImageItem]) -> None:
        self.items = list(items)
        self._by_key = {(item.folder, item.image_name): i for i, item in enumerate(self.items)}
        folders: list[dict] = []
        for i, item in enumerate(self.items):
            if not folders or folders[-1]["name"] != item.folder:
                folders.append({"name": item.folder, "start": i, "count": 1})
            else:
                folders[-1]["count"] += 1
        self._folders = folders
        self.ready = True

    def fail(self, message: str) -> None:
        self.error = message
        self.ready = True

    def _resume_index(self) -> int:
        for i, item in enumerate(self.items):
            if not self.store.has_row(item.folder, item.image_name):
                return i
        return max(0, len(self.items) - 1)

    def meta(self) -> dict:
        if not self.ready:
            return {"ready": False, "error": None, "csv_path": str(self.store.path)}
        if self.error:
            return {"ready": True, "error": self.error, "csv_path": str(self.store.path)}
        return {
            "ready": True,
            "error": None,
            "csv_path": str(self.store.path),
            "total": len(self.items),
            "labeled": sum(1 for item in self.items if self.store.has_row(item.folder, item.image_name)),
            "folders": self._folders,
            "resume_index": self._resume_index(),
        }

    def _label_view(self, folder: str, image_name: str) -> dict:
        row = self.store.get(folder, image_name)
        return {
            "saved": row is not None,
            "value": (row or {}).get("value") or None,
            "subtype": (row or {}).get("subtype") or None,
            "summary": _summary(row),
        }

    def item(self, index: int) -> dict:
        self._check_index(index)
        item = self.items[index]
        folder_pos = next(i for i, folder in enumerate(self._folders) if folder["name"] == item.folder)
        folder = self._folders[folder_pos]
        return {
            "index": index,
            "total": len(self.items),
            "folder": item.folder,
            "image_name": item.image_name,
            "folder_number": folder_pos + 1,
            "folder_count": len(self._folders),
            "image_in_folder": index - folder["start"] + 1,
            "images_in_folder": folder["count"],
            **self._label_view(item.folder, item.image_name),
        }

    def folder_images(self, name: str) -> list[dict]:
        rows = []
        for i, item in enumerate(self.items):
            if item.folder != name:
                continue
            view = self._label_view(item.folder, item.image_name)
            rows.append({
                "image_name": item.image_name,
                "index": i,
                "summary": view["summary"] or ("—" if view["saved"] else ""),
            })
        return rows

    def label(self, folder: str, image_name: str, body: dict) -> dict:
        if (folder, image_name) not in self._by_key:
            raise KeyError(f"{folder}/{image_name} is not in the blob list")
        value = body.get("value") or None
        subtype = body.get("subtype") or None
        row = self.store.replace(
            folder,
            image_name,
            value=None if value is None else str(value),
            subtype=None if subtype is None else str(subtype),
        )
        LOGGER.info("%s / %s -> %s", folder, image_name, _summary(row) or "(empty)")
        return {"ok": True, "labeled": sum(1 for item in self.items if self.store.has_row(item.folder, item.image_name)), **self._label_view(folder, image_name)}

    def image(self, index: int) -> tuple[bytes, str]:
        self._check_index(index)
        if self.read_image is None:
            raise RuntimeError("No image reader")
        return self.read_image(self.items[index])

    def _check_index(self, index: int) -> None:
        if not self.items or index < 0 or index >= len(self.items):
            raise IndexError(f"image {index} is out of range")


def _handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._bytes(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
                elif parsed.path == "/api/meta":
                    self._json(200, app.meta())
                elif parsed.path == "/api/item":
                    self._json(200, app.item(_index(parsed.query)))
                elif parsed.path == "/api/folder":
                    name = (parse_qs(parsed.query).get("name") or [""])[0]
                    images = app.folder_images(name)
                    if not images:
                        self._json(404, {"error": f"No folder named {name}"})
                    else:
                        self._json(200, {"folder": name, "images": images})
                elif parsed.path == "/image":
                    data, media = app.image(_index(parsed.query))
                    self._bytes(200, data, media, cache=False)
                else:
                    self._json(404, {"error": "not found"})
            except IndexError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                LOGGER.exception("GET %s failed", parsed.path)
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/label":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(length, 1_000_000)) or b"{}")
                value = body.get("value")
                subtype = body.get("subtype")
                if value is not None and str(value) not in VALUES:
                    self._json(400, {"error": "value must be KEEP, BLANK, or JUNK"})
                    return
                if subtype is not None and str(subtype) not in SUBTYPES:
                    self._json(400, {"error": "unknown junk subtype"})
                    return
                self._json(200, app.label(str(body.get("folder") or ""), str(body.get("image_name") or ""), body))
            except PermissionError:
                self._json(409, {"error": "The CSV is open in another program. Close it, then press the key again."})
            except (KeyError, ValueError) as exc:
                status = 404 if isinstance(exc, KeyError) else 400
                self._json(status, {"error": str(exc)})
            except Exception as exc:
                LOGGER.exception("label failed")
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _json(self, status: int, payload: dict) -> None:
            self._bytes(status, json.dumps(payload).encode(), "application/json")

        def _bytes(self, status: int, data: bytes, media: str, *, cache: bool = True) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(data)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _index(query: str) -> int:
    raw = (parse_qs(query).get("index") or ["-1"])[0]
    return int(raw)


def serve(app: App, host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), _handler(app))
    LOGGER.info("Open http://%s:%s", host, port)
    LOGGER.info("CSV: %s", app.store.path)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    finally:
        httpd.server_close()
