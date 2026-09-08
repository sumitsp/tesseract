"""
Detect and correct skew (tilt), 90/180/270 rotation, and horizontal mirroring
in scanned JPGs using OpenCV + Tesseract OSD, then write:
  - one CSV row per image with the measured/applied correction values
  - a corrected copy of every image under OUTPUT_DIR, same folder structure as input

Reads chart folders/images from Azure Blob Storage (same path style as file_counter.py),
downloads each image locally for processing, then writes corrected copies under OUTPUT_DIR.

Pipeline per image (each step is applied to the full-resolution image; detection
itself runs on a downscaled copy for speed):
  1. Tilt   - Hough line transform on text-line edges -> small-angle deskew.
  2. Rotate - Tesseract OSD on the deskewed image -> 0/90/180/270 correction.
  3. Mirror - OCR word-count/confidence compared normal vs. horizontally flipped
              on the now-upright image; whichever scores better is kept.

CSV "*_deg" columns are the degrees rotated CLOCKWISE that were actually applied
to correct the image (not the raw measured tilt of the original).
"""
from __future__ import annotations

import csv
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from pytesseract import Output

# ============================================================
# AZURE CONFIGURATION (same style as file_counter.py)
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rotation_report.csv")
# Optional override. Leave as None to auto-detect common install paths + PATH.
# Only set this if auto-detect fails on that machine, e.g.:
#   TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD: str | None = None
# OCR/rotation start at this folder, then continues with later folders.
# Use only the folder name, not the full path.
START_FROM = ""
JPG_SUFFIXES = {".jpg", ".jpeg"}

DETECT_MAX_DIM = 1600       # downscale target (longest side, px) used only for detection
SKEW_MIN_DEG = 0.1          # ignore measured tilt smaller than this (noise)
SKEW_MAX_DEG = 30.0         # ignore measured tilt larger than this (Hough noise, not real skew)
MIN_MIRROR_WORDS = 3        # need at least this many OCR'd words on one side to trust the mirror check

CSV_FIELDS = [
    "folder", "filename",
    "skew_angle_deg", "rotation_deg", "rotation_confidence",
    "mirrored", "mirror_words_normal", "mirror_conf_normal",
    "mirror_words_flipped", "mirror_conf_flipped",
    "status", "error", "output_path",
]


def log(message: str) -> None:
    print(message, flush=True)


def connect_azure() -> ContainerClient:
    log("Connecting to Azure Blob Storage...")
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        log("Azure connection created.")
        return container_client
    except Exception as exc:
        raise SystemExit(f"ERROR connecting to Azure:\n{exc}") from exc


def list_folder_blobs(container_client: ContainerClient) -> dict[str, list[str]]:
    """Group image blob names by immediate folder under PREFIX."""
    log("=" * 70)
    log("Scanning Azure Blob")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log("=" * 70)

    folder_blobs: dict[str, list[str]] = {}
    try:
        blobs = container_client.list_blobs(name_starts_with=PREFIX)
        for blob in blobs:
            relative_path = blob.name[len(PREFIX) :]
            parts = relative_path.split("/")
            if len(parts) < 2:
                continue
            folder_name = parts[0]
            filename = parts[-1]
            if Path(filename).suffix.lower() not in JPG_SUFFIXES:
                continue
            folder_blobs.setdefault(folder_name, []).append(blob.name)
    except Exception as exc:
        raise SystemExit(f"ERROR while reading blobs:\n{exc}") from exc

    return folder_blobs


def list_chart_folders(folder_blobs: dict[str, list[str]]) -> list[str]:
    folders = sorted(folder_blobs.keys(), key=lambda name: name.lower())
    if not START_FROM:
        return folders
    start_index = next(
        (i for i, folder in enumerate(folders) if folder == START_FROM),
        None,
    )
    if start_index is None:
        raise SystemExit(f"START_FROM folder not found under prefix: {START_FROM}")
    log(f"Starting at {START_FROM}; skipping {start_index} earlier folder(s)")
    return folders[start_index:]


def jpg_sort_key(blob_name: str) -> tuple:
    stem = Path(blob_name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_jpgs(folder_blobs: dict[str, list[str]], folder_name: str) -> list[str]:
    images = list(folder_blobs.get(folder_name, []))
    images.sort(key=jpg_sort_key)
    return images


def download_blob(
    container_client: ContainerClient,
    blob_name: str,
    dest: Path,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = container_client.download_blob(blob_name).readall()
    dest.write_bytes(data)
    return dest


def find_tesseract() -> Path:
    """Locate tesseract.exe on this machine (override, common paths, then PATH)."""
    candidates: list[Path] = []
    if TESSERACT_CMD:
        candidates.append(Path(TESSERACT_CMD))
    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    which = shutil.which("tesseract")
    if which:
        return Path(which)
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise SystemExit(
        "Tesseract OCR engine is not installed on this machine "
        "(pytesseract alone is not enough).\n"
        "Install the Windows binary, then reopen PowerShell:\n"
        "  https://github.com/UB-Mannheim/tesseract/wiki\n"
        "During setup, tick 'Add to PATH'. Verify with:\n"
        "  tesseract --version\n"
        "If installed elsewhere, set TESSERACT_CMD in rotation.py to that tesseract.exe.\n"
        f"Checked:\n{tried}"
    )


def require_tesseract() -> None:
    cmd = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = str(cmd)
    log(f"Using Tesseract: {cmd}")

    # First launch on a machine can spuriously fail (Defender/SmartScreen).
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            version = pytesseract.get_tesseract_version()
            log(f"Tesseract version: {version}")
            return
        except Exception as exc:
            last_exc = exc
            log(f"  Tesseract check attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            time.sleep(2)

    raise SystemExit(
        "Found tesseract.exe but it is not runnable.\n"
        f"  {cmd}\n"
        f"  {type(last_exc).__name__}: {last_exc}"
    )


def downscale(image: np.ndarray, max_dim: int = DETECT_MAX_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def rotate_bound_cw(image: np.ndarray, angle_cw_deg: float) -> np.ndarray:
    """Rotate `image` by `angle_cw_deg` degrees clockwise, expanding the canvas
    (white fill) so nothing is cropped."""
    if abs(angle_cw_deg) < 1e-6:
        return image
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D((cx, cy), -angle_cw_deg, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - cx
    matrix[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def measure_skew_deg(gray_small: np.ndarray) -> float:
    """Median tilt of near-horizontal text-line edges, as degrees CLOCKWISE
    needed to correct it (0.0 if nothing usable is found)."""
    edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
    min_len = max(30, gray_small.shape[1] // 4)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=150, minLineLength=min_len, maxLineGap=20
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= 30:
            angles.append(angle)
        elif abs(angle) >= 150:
            angles.append(angle - 180 if angle > 0 else angle + 180)
    if not angles:
        return 0.0
    measured_tilt = float(np.median(angles))  # image-space, clockwise-positive tilt of the text
    correction = -measured_tilt               # rotate clockwise by this amount to undo it
    if abs(correction) < SKEW_MIN_DEG or abs(correction) > SKEW_MAX_DEG:
        return 0.0
    return correction


def measure_rotation(gray: np.ndarray) -> tuple[int, float]:
    """Returns (degrees clockwise to apply, orientation confidence) from Tesseract OSD."""
    try:
        osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
    except pytesseract.TesseractError:
        return 0, 0.0
    return int(osd.get("rotate", 0)) % 360, float(osd.get("orientation_conf", 0.0))


def ocr_quality(gray: np.ndarray) -> tuple[int, float]:
    """Returns (word_count, mean_confidence) from a plain OCR pass."""
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    word_count = 0
    conf_sum = 0.0
    for text, conf in zip(data["text"], data["conf"]):
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val >= 0 and text.strip():
            word_count += 1
            conf_sum += conf_val
    mean_conf = conf_sum / word_count if word_count else 0.0
    return word_count, mean_conf


def correct_image(image: np.ndarray) -> dict:
    """Runs the full tilt -> rotation -> mirror pipeline on a full-resolution
    BGR image. Returns a dict of measured values plus the corrected image."""
    gray_small = cv2.cvtColor(downscale(image), cv2.COLOR_BGR2GRAY)
    skew_deg = measure_skew_deg(gray_small)
    deskewed = rotate_bound_cw(image, skew_deg)

    rotation_deg, rotation_conf = measure_rotation(cv2.cvtColor(downscale(deskewed), cv2.COLOR_BGR2GRAY))
    if rotation_deg == 90:
        upright = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_deg == 180:
        upright = cv2.rotate(deskewed, cv2.ROTATE_180)
    elif rotation_deg == 270:
        upright = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        upright = deskewed

    upright_small = cv2.cvtColor(downscale(upright), cv2.COLOR_BGR2GRAY)
    words_normal, conf_normal = ocr_quality(upright_small)
    flipped_small = cv2.flip(upright_small, 1)
    words_flipped, conf_flipped = ocr_quality(flipped_small)

    mirrored = False
    if max(words_normal, words_flipped) >= MIN_MIRROR_WORDS:
        mirrored = (words_flipped, conf_flipped) > (words_normal, conf_normal)
    final = cv2.flip(upright, 1) if mirrored else upright

    return {
        "skew_angle_deg": round(skew_deg, 3),
        "rotation_deg": rotation_deg,
        "rotation_confidence": round(rotation_conf, 3),
        "mirrored": mirrored,
        "mirror_words_normal": words_normal,
        "mirror_conf_normal": round(conf_normal, 2),
        "mirror_words_flipped": words_flipped,
        "mirror_conf_flipped": round(conf_flipped, 2),
        "image": final,
    }


def process_folder(
    writer: csv.DictWriter,
    csv_file,
    container_client: ContainerClient,
    output_dir: Path,
    folder_name: str,
    blob_names: list[str],
) -> None:
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)
    log(f"  {len(blob_names)} images -> {out_folder}")

    with tempfile.TemporaryDirectory(prefix=f"rotation_{folder_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for blob_name in blob_names:
            filename = Path(blob_name).name
            log(f"  {filename}")
            row = {"folder": folder_name, "filename": filename, "status": "ok", "error": ""}
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                image = cv2.imread(str(local_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (unreadable/corrupt image)")
                result = correct_image(image)
                out_path = out_folder / filename
                # Always write a copy — including images that needed no correction
                # (skew=0, rotation=0, mirrored=False).
                changed = (
                    abs(result["skew_angle_deg"]) > 0
                    or result["rotation_deg"] != 0
                    or result["mirrored"]
                )
                if not cv2.imwrite(str(out_path), result.pop("image")):
                    raise ValueError(f"cv2.imwrite failed: {out_path}")
                row.update(result)
                row["output_path"] = str(out_path)
                log(
                    f"    wrote {'corrected' if changed else 'unchanged (copied)'} "
                    f"skew={row['skew_angle_deg']} rot={row['rotation_deg']} "
                    f"mirror={row['mirrored']} -> {out_path.name}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["output_path"] = ""
                log(f"    ERROR: {row['error']}")
            writer.writerow(row)
            csv_file.flush()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log("=== TILT / ROTATION / MIRROR CORRECTION (OpenCV + Tesseract OSD) ===")
    output_dir = OUTPUT_DIR
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {output_dir}")
    log(f"CSV:    {CSV_PATH}")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / JPG blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    require_tesseract()
    output_dir.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not CSV_PATH.is_file()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            csv_file.flush()
        for folder_name in chart_folders:
            images = list_jpgs(folder_blobs, folder_name)
            if not images:
                log(f"Folder: {folder_name} (no JPGs, skipped)")
                continue
            log(f"Folder: {folder_name}")
            process_folder(writer, csv_file, container_client, output_dir, folder_name, images)

    log(f"Done. Report: {CSV_PATH}")


if __name__ == "__main__":
    main()
