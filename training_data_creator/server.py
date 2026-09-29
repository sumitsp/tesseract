"""Localhost labeling server. Images come from blob; the CSV stays on this machine."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from training_data_creator.catalog import ImageItem
from training_data_creator.labels import VALUES, LabelStore

LOGGER = logging.getLogger("training_data_creator")
_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
ReadImage = Callable[[ImageItem], tuple[bytes, str]]


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
            if self.store.get(item.folder, item.image_name) is None:
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
            "labeled": self._labeled_in_catalog(),
            "folders": self._folders,
            "resume_index": self._resume_index(),
        }

    def _labeled_in_catalog(self) -> int:
        return sum(1 for item in self.items if self.store.get(item.folder, item.image_name))

    def item(self, index: int) -> dict:
        self._check_index(index)
        item = self.items[index]
        folder_pos = next(i for i, f in enumerate(self._folders) if f["name"] == item.folder)
        folder = self._folders[folder_pos]
        return {
            "index": index,
            "total": len(self.items),
            "folder": item.folder,
            "image_name": item.image_name,
            "value": self.store.get(item.folder, item.image_name),
            "folder_number": folder_pos + 1,
            "folder_count": len(self._folders),
            "image_in_folder": index - folder["start"] + 1,
            "images_in_folder": folder["count"],
        }

    def folder_images(self, name: str) -> list[dict]:
        return [
            {
                "image_name": item.image_name,
                "index": i,
                "value": self.store.get(item.folder, item.image_name),
            }
            for i, item in enumerate(self.items)
            if item.folder == name
        ]

    def label(self, folder: str, image_name: str, value: str) -> dict:
        if (folder, image_name) not in self._by_key:
            raise KeyError(f"{folder}/{image_name} is not in the blob list")
        self.store.upsert(folder, image_name, value)
        LOGGER.info("%s / %s -> %s", folder, image_name, value)
        return {"ok": True, "value": value, "labeled": self._labeled_in_catalog()}

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
                value = str(body.get("value") or "")
                if value not in VALUES:
                    self._json(400, {"error": "value must be Handwritten or Printed"})
                    return
                self._json(200, app.label(str(body.get("folder") or ""), str(body.get("image_name") or ""), value))
            except PermissionError:
                self._json(409, {"error": "The CSV is open in another program. Close it, then press the key again."})
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
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
