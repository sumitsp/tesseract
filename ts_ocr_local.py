"""
Extract raw text from images with Tesseract.

Reads chart folders from a local INPUT_DIR (no Azure, no START_FROM).
Writes one .txt per chart folder under OUTPUT_DIR.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytesseract
from PIL import Image

# ============================================================
# LOCAL PATHS
# ============================================================

INPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\extracted_text_tesseract")

# Set this if auto-detect fails. Leave as-is to search common locations + PATH.
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
CANDIDATE_CMDS = [
    TESSERACT_CMD,
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
IMAGE_SUFFIXES = {
    ".bmp",
    ".dib",
    ".gif",
    ".j2k",
    ".jfif",
    ".jp2",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pbm",
    ".pgm",
    ".png",
    ".pnm",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


def log(message: str) -> None:
    print(message, flush=True)


def find_tesseract() -> Path:
    for candidate in CANDIDATE_CMDS:
        if candidate.is_file():
            return candidate
    which = shutil.which("tesseract")
    if which:
        return Path(which)
    raise SystemExit(
        "Tesseract is not installed, or tesseract.exe is not on PATH.\n"
        "1. Install: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "2. During setup, tick 'Add to PATH'.\n"
        "3. Close and reopen PowerShell, then run:\n"
        "     tesseract --version\n"
        "4. If that works, run ts_ocr_local.py again.\n"
        "   If not, set TESSERACT_CMD in ts_ocr_local.py to the full path of tesseract.exe."
    )


def image_sort_key(path: Path) -> tuple:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def list_chart_folders(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"INPUT_DIR does not exist or is not a folder:\n  {input_dir}")
    return sorted(
        (entry for entry in input_dir.iterdir() if entry.is_dir()),
        key=lambda path: path.name.lower(),
    )


def list_images(folder: Path) -> list[Path]:
    images = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    images.sort(key=image_sort_key)
    return images


def extract_text(image_path: Path) -> str:
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image)
    return text.strip()


def process_folder(
    input_dir: Path,
    output_dir: Path,
    folder: Path,
    images: list[Path],
) -> None:
    out_folder = output_dir / folder.relative_to(input_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"{folder.name}.txt"
    log(f"  {len(images)} images -> {out_file}")
    parts: list[str] = []
    for image_path in images:
        log(f"  {image_path.name}")
        try:
            text = extract_text(image_path) or "[no text detected]"
        except Exception as exc:
            text = f"[ERROR extracting {image_path.name}: {exc}]"
        parts.append(f"===== {image_path.name} =====\n{text}")
        out_file.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        log(f"  saved {out_file}")
    log(f"Finished {out_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    tesseract_cmd = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
    log(f"Using Tesseract: {tesseract_cmd}")
    log("=== TESSERACT OCR (LOCAL) ===")
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    log(f"INPUT:  {input_dir}")
    log(f"OUTPUT: {output_dir}")

    chart_folders = list_chart_folders(input_dir)
    if not chart_folders:
        raise SystemExit(f"No chart folders found under: {input_dir}")

    log(f"Will process {len(chart_folders)} chart folder(s)")
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in chart_folders:
        images = list_images(folder)
        if not images:
            log(f"Folder: {folder.name} (no images, skipped)")
            continue
        log(f"Folder: {folder}")
        process_folder(input_dir, output_dir, folder, images)


if __name__ == "__main__":
    main()
