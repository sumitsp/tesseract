"""
Detect and correct page rotation (0/90/180/270) and horizontal mirroring
using OpenCV only (no Tesseract).

Also measures tilt (skew). Tilt is corrected only when |tilt| <= 5 degrees.
If |tilt| > 5 degrees, tilt is reported but NOT corrected (rotation + mirror
still applied).

Writes:
  - CSV with tilt_angle_deg, rotation_deg, mirrored (Yes/No)
  - corrected images under OUTPUT_DIR
"""
from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

# ============================================================
# AZURE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"

OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rotation_report.csv")
START_FROM = ""
JPG_SUFFIXES = {".jpg", ".jpeg"}

DETECT_MAX_DIM = 1200
TILT_CORRECT_MAX_DEG = 5.0   # correct tilt only if |tilt| <= this; else leave tilt alone
TILT_MIN_DEG = 0.5           # ignore tiny noise below this
ROTATION_MARGIN = 1.15       # best orientation score must beat 2nd-best by this factor
MIRROR_MARGIN = 1.20         # flipped margin score must clearly beat normal to flip

CSV_FIELDS = [
    "folder",
    "filename",
    "tilt_angle_deg",
    "tilt_corrected",
    "rotation_deg",
    "mirrored",
    "status",
    "error",
    "output_path",
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
    log("=" * 70)
    log("Scanning Azure Blob")
    log("=" * 70)
    log(f"Storage Account : {STORAGE_ACCOUNT}")
    log(f"Container       : {CONTAINER_NAME}")
    log(f"Prefix          : {PREFIX}")
    log("=" * 70)

    folder_blobs: dict[str, list[str]] = {}
    try:
        for blob in container_client.list_blobs(name_starts_with=PREFIX):
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
    start_index = next((i for i, f in enumerate(folders) if f == START_FROM), None)
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


def download_blob(container_client: ContainerClient, blob_name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(container_client.download_blob(blob_name).readall())
    return dest


def downscale(image: np.ndarray, max_dim: int = DETECT_MAX_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binary_ink(gray: np.ndarray) -> np.ndarray:
    """Dark text -> white ink mask on black background."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return thr


def rotate_bound_cw(image: np.ndarray, angle_cw_deg: float) -> np.ndarray:
    if abs(angle_cw_deg) < 1e-6:
        return image
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), -angle_cw_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - cx
    matrix[1, 2] += (new_h / 2.0) - cy
    fill = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
    )


def apply_rotation_cw(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


# ---------------------------------------------------------------------------
# TILT (skew) — conservative Hough on near-horizontal edges only
# ---------------------------------------------------------------------------

def measure_tilt_deg(gray_small: np.ndarray) -> float:
    """
    Measured page tilt in degrees (clockwise-positive of the content).
    Returns 0.0 if not enough consistent near-horizontal lines.
    """
    edges = cv2.Canny(gray_small, 60, 160, apertureSize=3)
    min_len = max(40, gray_small.shape[1] // 5)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120,
        minLineLength=min_len, maxLineGap=12,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = int(x2) - int(x1)
        dy = int(y2) - int(y1)
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        # Keep only near-horizontal strokes (true text-line tilt), not random edges.
        if abs(angle) <= 15:
            angles.append(angle)
        elif abs(angle) >= 165:
            angles.append(angle - 180 if angle > 0 else angle + 180)

    if len(angles) < 8:
        return 0.0

    median = float(np.median(angles))
    # Require agreement: most lines near the median.
    close = [a for a in angles if abs(a - median) <= 2.0]
    if len(close) < max(8, int(0.55 * len(angles))):
        return 0.0

    if abs(median) < TILT_MIN_DEG:
        return 0.0
    return round(median, 2)


# ---------------------------------------------------------------------------
# ROTATION 0/90/180/270 — projection profile scoring (OpenCV only)
# ---------------------------------------------------------------------------

def _orientation_score(gray: np.ndarray) -> float:
    """
    Higher = more likely upright text (strong horizontal text-line structure).
    Uses row-projection variance of ink + slight preference for content on top half.
    """
    ink = binary_ink(gray)
    # Prefer page with clear horizontal bands of text.
    row_proj = ink.sum(axis=1).astype(np.float64)
    if row_proj.sum() < 1:
        return 0.0
    row_proj /= row_proj.sum()
    row_var = float(np.var(row_proj))

    col_proj = ink.sum(axis=0).astype(np.float64)
    col_proj /= col_proj.sum() + 1e-9
    col_var = float(np.var(col_proj))

    # Upright text: row variance usually dominates column variance.
    structure = row_var / (col_var + 1e-12)

    h = ink.shape[0]
    top = ink[: h // 2].sum()
    bottom = ink[h // 2 :].sum()
    # Mild bias: headers / titles often put more ink in upper half when upright.
    top_bias = 1.0 + 0.05 * ((top - bottom) / (top + bottom + 1e-9))

    return structure * top_bias


def measure_rotation_deg(gray_small: np.ndarray) -> int:
    """
    Best clockwise rotation in {0,90,180,270}.
    Stays at 0 unless another orientation clearly wins (avoids random flips).
    """
    candidates = {
        0: gray_small,
        90: cv2.rotate(gray_small, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(gray_small, cv2.ROTATE_180),
        270: cv2.rotate(gray_small, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    scores = {deg: _orientation_score(img) for deg, img in candidates.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_deg, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_deg == 0:
        return 0
    if best_score < second_score * ROTATION_MARGIN:
        return 0
    # Extra caution on 180: only accept if clearly better than 0.
    if best_deg == 180 and best_score < scores[0] * (ROTATION_MARGIN + 0.1):
        return 0
    return best_deg


# ---------------------------------------------------------------------------
# MIRROR — left-margin whitespace heuristic (OpenCV only, conservative)
# ---------------------------------------------------------------------------

def _left_margin_score(gray: np.ndarray) -> float:
    """
    Larger score => more empty left margin (typical LTR document layout).
    """
    ink = binary_ink(gray)
    h, w = ink.shape
    band = max(8, w // 12)
    left = ink[:, :band].mean()
    right = ink[:, w - band :].mean()
    # Prefer less ink on the left (wider blank left margin).
    return float((right + 1.0) / (left + 1.0))


def measure_mirror(gray_small: np.ndarray) -> bool:
    """
    Return True only if horizontally flipped clearly looks more like a normal
    LTR page (wider left margin). Default False when unsure.
    """
    normal = _left_margin_score(gray_small)
    flipped = _left_margin_score(cv2.flip(gray_small, 1))
    if flipped < normal * MIRROR_MARGIN:
        return False
    # Also require absolute gap so tiny noise doesn't flip pages.
    if (flipped - normal) < 0.25:
        return False
    return True


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def correct_image(image: np.ndarray) -> dict:
    gray_small = to_gray(downscale(image))

    tilt_angle = measure_tilt_deg(gray_small)
    # Correct tilt ONLY when |tilt| <= 5°. Larger tilt is reported but not applied.
    tilt_corrected = abs(tilt_angle) <= TILT_CORRECT_MAX_DEG and abs(tilt_angle) >= TILT_MIN_DEG
    working = rotate_bound_cw(image, -tilt_angle) if tilt_corrected else image

    gray_for_rot = to_gray(downscale(working))
    rotation_deg = measure_rotation_deg(gray_for_rot)
    working = apply_rotation_cw(working, rotation_deg)

    gray_for_mirror = to_gray(downscale(working))
    mirrored = measure_mirror(gray_for_mirror)
    if mirrored:
        working = cv2.flip(working, 1)

    return {
        "tilt_angle_deg": tilt_angle,
        "tilt_corrected": "Yes" if tilt_corrected else "No",
        "rotation_deg": rotation_deg,
        "mirrored": "Yes" if mirrored else "No",
        "image": working,
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
            row = {
                "folder": folder_name,
                "filename": filename,
                "status": "ok",
                "error": "",
            }
            try:
                local_path = download_blob(container_client, blob_name, tmp_path / filename)
                image = cv2.imread(str(local_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (unreadable/corrupt image)")

                result = correct_image(image)
                out_path = out_folder / filename
                if not cv2.imwrite(str(out_path), result.pop("image")):
                    raise ValueError(f"cv2.imwrite failed: {out_path}")

                row.update(result)
                row["output_path"] = str(out_path)
                log(
                    f"    tilt={row['tilt_angle_deg']} (corrected={row['tilt_corrected']}) "
                    f"rot={row['rotation_deg']} mirror={row['mirrored']} -> {out_path.name}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["output_path"] = ""
                row.setdefault("tilt_angle_deg", "")
                row.setdefault("tilt_corrected", "")
                row.setdefault("rotation_deg", "")
                row.setdefault("mirrored", "")
                log(f"    ERROR: {row['error']}")
            writer.writerow(row)
            csv_file.flush()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    log("=== TILT / ROTATION / MIRROR (OpenCV only) ===")
    log(f"INPUT:  azure://{STORAGE_ACCOUNT}/{CONTAINER_NAME}/{PREFIX}")
    log(f"OUTPUT: {OUTPUT_DIR}")
    log(f"CSV:    {CSV_PATH}")
    log(f"Tilt correction only if |tilt| <= {TILT_CORRECT_MAX_DEG}°")
    log(f"START_FROM: {START_FROM or '(first folder)'}")

    container_client = connect_azure()
    folder_blobs = list_folder_blobs(container_client)
    chart_folders = list_chart_folders(folder_blobs)
    if not chart_folders:
        raise SystemExit(
            "No chart folders / JPG blobs found.\n"
            "Check CONTAINER_NAME, PREFIX, and blob folder structure."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fresh report file (new columns). Rename old CSV if you need to keep it.
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        for folder_name in chart_folders:
            images = list_jpgs(folder_blobs, folder_name)
            if not images:
                log(f"Folder: {folder_name} (no JPGs, skipped)")
                continue
            log(f"Folder: {folder_name}")
            process_folder(writer, csv_file, container_client, OUTPUT_DIR, folder_name, images)

    log(f"Done. Report: {CSV_PATH}")


if __name__ == "__main__":
    main()
