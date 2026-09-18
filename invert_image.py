#!/usr/bin/env python3
"""Convert document images between light and dark backgrounds."""

from __future__ import annotations

from pathlib import Path

import cv2

# =============================================================================
# RUN CONFIG — edit these
# =============================================================================

INPUT_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\input")
OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\inverted_output")

# "invert"         = always swap every pixel
# "white_bg"       = produce black text on a white background
# "black_bg"       = produce white text on a black background
PROCESS = "invert"

# Used only when INPUT_PATH is a folder.
RECURSIVE = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# =============================================================================


def has_white_background(image) -> bool:
    """Determine background polarity from the median grayscale intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.medianBlur(gray, 5).mean()) >= 127.5


def transform(image, process: str):
    process = process.strip().lower()
    if process == "invert":
        return cv2.bitwise_not(image)
    if process == "white_bg":
        return image if has_white_background(image) else cv2.bitwise_not(image)
    if process == "black_bg":
        return cv2.bitwise_not(image) if has_white_background(image) else image
    raise ValueError('PROCESS must be "invert", "white_bg", or "black_bg"')


def discover_images(input_path: Path) -> tuple[Path, list[Path]]:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"INPUT_PATH does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return path.parent, [path]

    iterator = path.rglob("*") if RECURSIVE else path.glob("*")
    images = sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).lower(),
    )
    return path, images


def main() -> None:
    root, images = discover_images(INPUT_PATH)
    if not images:
        raise SystemExit(f"No supported images found under: {INPUT_PATH}")

    output_root = OUTPUT_DIR.expanduser().resolve()
    for input_file in images:
        relative = input_file.relative_to(root)
        output_file = output_root / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(input_file), cv2.IMREAD_COLOR)
        if image is None:
            print(f"SKIPPED unreadable image: {input_file}")
            continue

        output = transform(image, PROCESS)
        if not cv2.imwrite(str(output_file), output):
            print(f"ERROR writing: {output_file}")
            continue
        print(f"{input_file} -> {output_file}")


if __name__ == "__main__":
    main()
