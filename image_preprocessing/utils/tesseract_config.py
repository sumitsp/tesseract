"""Locate the Tesseract binary for pytesseract (Windows-friendly)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _candidates() -> list[Path]:
    env = os.environ.get("TESSERACT_CMD", "").strip()
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    which = shutil.which("tesseract")
    if which:
        paths.append(Path(which))
    for base in (
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    ):
        paths.append(base / "tesseract.exe")
    return paths


def configure_tesseract() -> Path | None:
    """Set ``pytesseract.pytesseract.tesseract_cmd`` if we can find a binary."""
    try:
        import pytesseract
    except ImportError:
        return None

    for path in _candidates():
        if path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return path
    return None


def tesseract_help_message() -> str:
    return (
        "Tesseract OCR was not found.\n"
        "1) Install: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "2) Either add the install folder to PATH, or set before running:\n"
        '   set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n'
        "3) Verify: tesseract --list-langs   (must include osd)\n"
        "4) Run with venv python:\n"
        "   .\\venv\\Scripts\\python.exe main.py --input ... --output ..."
    )
