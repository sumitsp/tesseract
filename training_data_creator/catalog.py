"""Chart images in a blob prefix: folder 1 image 1, folder 1 image 2, then folder 2."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("training_data_creator")

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


def list_images(container: Any, prefix: str) -> list[ImageItem]:
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
    return items


def read_blob_image(container: Any, item: ImageItem) -> tuple[bytes, str]:
    """Return display bytes. JPEG/PNG/GIF/WEBP pass through; TIFF and BMP become PNG."""
    data = container.download_blob(item.blob_name).readall()
    suffix = Path(item.image_name).suffix.lower()
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
