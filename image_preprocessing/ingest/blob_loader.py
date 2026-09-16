"""Azure Blob Storage input — one chart folder at a time, like rotation.py."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from image_preprocessing.config import RASTER_EXTENSIONS, BlobSettings
from image_preprocessing.ingest.image_loader import read_embedded_dpi
from image_preprocessing.results import LoadedPage
from image_preprocessing.utils.image_utils import pil_to_bgr

LOGGER = logging.getLogger(__name__)


def connect_container_client(settings: BlobSettings) -> Any:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    account_url = f"https://{settings.storage_account}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    client = BlobServiceClient(account_url=account_url, credential=credential)
    return client.get_container_client(settings.container_name)


def list_folder_blobs(
    container_client: Any,
    settings: BlobSettings,
) -> dict[str, list[str]]:
    """Map chart folder name -> blob paths under ``prefix/<folder>/``."""
    prefix = settings.prefix_normalized
    folder_blobs: dict[str, list[str]] = {}
    for blob in container_client.list_blobs(name_starts_with=prefix):
        relative = blob.name[len(prefix) :] if blob.name.startswith(prefix) else blob.name
        parts = relative.split("/")
        if len(parts) < 2:
            continue
        folder_name = parts[0]
        filename = parts[-1]
        if Path(filename).suffix.lower() not in RASTER_EXTENSIONS:
            continue
        folder_blobs.setdefault(folder_name, []).append(blob.name)
    return folder_blobs


def list_chart_folders(
    folder_blobs: dict[str, list[str]],
    *,
    start_from: str = "",
) -> list[str]:
    folders = sorted(folder_blobs.keys(), key=lambda name: name.lower())
    if not start_from:
        return folders
    idx = next((i for i, f in enumerate(folders) if f == start_from), None)
    if idx is None:
        raise FileNotFoundError(f"START_FROM folder not found under prefix: {start_from}")
    LOGGER.info("Starting at %s; skipping %s earlier folder(s)", start_from, idx)
    return folders[idx:]


def _blob_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_images_in_folder(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=_blob_sort_key)
    return images


def download_blob_bytes(container_client: Any, blob_name: str) -> bytes:
    return container_client.download_blob(blob_name).readall()


def load_raster_pages_from_bytes(
    data: bytes,
    *,
    document_index: int,
    document_name: str,
    input_file: str,
    blob_name: str,
    chart_folder: str,
) -> list[LoadedPage]:
    pages: list[LoadedPage] = []
    with Image.open(io.BytesIO(data)) as master:
        n_frames = max(1, getattr(master, "n_frames", 1) or 1)
        for idx in range(n_frames):
            master.seek(idx)
            frame = master.copy()
            frame.load()
            frame = ImageOps.exif_transpose(frame) or frame
            if frame.mode not in {"RGB", "L"}:
                frame = frame.convert("RGB")
            dpi, dpi_source = read_embedded_dpi(frame)
            image = pil_to_bgr(frame).copy()
            warnings: list[str] = []
            if dpi is None:
                warnings.append("DPI_UNKNOWN: Raster DPI metadata unavailable")
            pages.append(
                LoadedPage(
                    document_name=document_name,
                    document_index=document_index,
                    page_number=idx + 1,
                    input_file=input_file,
                    input_path=Path(input_file),
                    input_format=Path(document_name).suffix.lower().lstrip(".") or "unknown",
                    image=image,
                    input_dpi=dpi,
                    dpi_source=dpi_source,
                    source_kind="raster",
                    warnings=warnings,
                    extras={
                        "frame_count": n_frames,
                        "blob_name": blob_name,
                        "chart_folder": chart_folder,
                    },
                )
            )
    return pages
