"""Chart images from a local folder or a blob prefix: folder 1 image 1, then image 2, then folder 2."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("blank_junk_creator")

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".jfif", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
)
_PASSTHROUGH = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class ImageItem:
    folder: str
    image_name: str
    blob_name: str


def _natural_key(name: str) -> tuple:
    stem = Path(name).stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem.lower())


def connect_container(storage_account: str, container_name: str) -> Any:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    account_url = f"https://{storage_account}.blob.core.windows.net"
    client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
    return client.get_container_client(container_name)


def apply_start_from(items: list[ImageItem], start_from: str) -> list[ImageItem]:
    """Keep the named folder and every folder after it. Empty means the full list."""
    name = (start_from or "").strip()
    if not name:
        return items
    folders: list[str] = []
    for item in items:
        if not folders or folders[-1] != item.folder:
            folders.append(item.folder)
    exact = [folder for folder in folders if folder == name]
    folded = [folder for folder in folders if folder.lower() == name.lower()]
    match = exact[0] if exact else (folded[0] if len(folded) == 1 else "")
    if not match:
        if len(folded) > 1:
            raise ValueError(f"START_FROM {name!r} matches more than one folder")
        raise ValueError(f"START_FROM folder not found: {name}")
    LOGGER.info("Starting at folder %s", match)
    return [item for item in items if folders.index(item.folder) >= folders.index(match)]


def list_images(container: Any, prefix: str, start_from: str = "") -> list[ImageItem]:
    """Images under prefix/<folder>/<file>, folders then files in name order."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    by_folder: dict[str, list[tuple[str, str]]] = {}
    seen = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        seen += 1
        if seen % 2000 == 0:
            LOGGER.info("Scanned %s blobs...", seen)
        parts = blob.name[len(prefix):].split("/")
        if len(parts) < 2 or Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        folder = parts[0]
        image_name = "/".join(parts[1:])
        by_folder.setdefault(folder, []).append((image_name, blob.name))
    items: list[ImageItem] = []
    for folder in sorted(by_folder, key=str.lower):
        files = sorted(by_folder[folder], key=lambda pair: _natural_key(Path(pair[0]).name))
        items.extend(ImageItem(folder, name, blob_name) for name, blob_name in files)
    LOGGER.info("%s images in %s folders (scanned %s blobs)", len(items), len(by_folder), seen)
    kept = apply_start_from(items, start_from)
    if kept is not items:
        LOGGER.info("%s images from folder %s onward", len(kept), (start_from or "").strip())
    return kept


def list_local_images(input_path: Path, start_from: str = "") -> list[ImageItem]:
    """Images under a folder. The parent folder name is the chart folder."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LOCAL_INPUT does not exist: {path}")
    if path.is_file():
        files = [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    else:
        files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: (str(p.parent).lower(), _natural_key(p.name)))
    items = [ImageItem(p.parent.name, p.name, str(p)) for p in files]
    LOGGER.info("%s images in %s folders under %s", len(items), len({item.folder for item in items}), path)
    kept = apply_start_from(items, start_from)
    if kept is not items:
        LOGGER.info("%s images from folder %s onward", len(kept), (start_from or "").strip())
    return kept


def _as_display(data: bytes, suffix: str) -> tuple[bytes, str]:
    """JPEG/PNG/GIF/WEBP pass through; TIFF and BMP become PNG."""
    media = _PASSTHROUGH.get(suffix)
    if media is not None:
        return data, media
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)
        frame = im.convert("RGB")
        out = io.BytesIO()
        frame.save(out, format="PNG")
    return out.getvalue(), "image/png"


def read_local_image(item: ImageItem) -> tuple[bytes, str]:
    data = Path(item.blob_name).read_bytes()
    return _as_display(data, Path(item.image_name).suffix.lower())


def read_blob_image(container: Any, item: ImageItem) -> tuple[bytes, str]:
    data = container.download_blob(item.blob_name).readall()
    return _as_display(data, Path(item.image_name).suffix.lower())
