#!/usr/bin/env python3
"""Preprocess document pages before OCR.

Examples
--------
    python main.py --input /documents/input.pdf --output /documents/output
    python main.py --input /documents/input_folder --output /documents/output
    python main.py --input /documents/input_folder --output /documents/output --debug
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_PARENT = _PKG_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from image_preprocessing.config import (  # noqa: E402
    ENABLE_DEBUG,
    LOW_DPI_WARNING_THRESHOLD,
    MAX_SKEW_ANGLE,
    MIRROR_CONFIDENCE_THRESHOLD,
    OSD_MIN_ORIENTATION_CONFIDENCE,
    QUALITY_REVIEW_THRESHOLD,
    ROTATION_RESIDUAL_CONFIDENCE_THRESHOLD,
    SKEW_CONFIDENCE_THRESHOLD,
    SUPPORTED_EXTENSIONS,
    TARGET_DPI,
    PipelineConfig,
    default_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Production document-page preprocessor for OCR. "
            "Reads PDF / JPG / PNG / TIFF (and other common document images), "
            "processes each page sequentially, writes corrected PNG pages and "
            "an Excel report. Uncertain detections leave the page unchanged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipeline per page (sequential, not a joint optimiser):\n"
            "  1. Quality / DPI analysis\n"
            "  2. Standardize DPI (downsample to 400 only; never upscale)\n"
            "  3. Printed vs handwritten (existing ConvNeXt classifier)\n"
            "  4. Residual angle (classical) + Tesseract OSD for the quadrant\n"
            "  5. Mirror detection (after rotation only; OCR-confirmed)\n"
            "  6. Fine tilt / skew (after rotation + mirror)\n"
            "  7. Validate; reject corrections that make alignment worse\n"
            "  8. Save corrected page\n"
            "  9. Append one Excel row\n\n"
            "Sign convention: rotation_angle and tilt_angle are the clockwise\n"
            "offset of content from upright, in degrees. Correction rotates the\n"
            "image counter-clockwise by that amount (OpenCV positive angle).\n\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
        ),
    )
    parser.add_argument("--input", required=True, help="File or directory of documents")
    parser.add_argument("--output", required=True, help="Output directory (created if needed)")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=ENABLE_DEBUG,
        help="Write intermediate debug images under output/debug/",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When input is a directory, do not recurse into subfolders",
    )
    parser.add_argument("--target-dpi", type=int, default=TARGET_DPI)
    parser.add_argument(
        "--low-dpi-threshold",
        type=int,
        default=LOW_DPI_WARNING_THRESHOLD,
        help="DPI below this emits LOW_DPI (pages are never upscaled)",
    )
    parser.add_argument("--max-skew-angle", type=float, default=MAX_SKEW_ANGLE)
    parser.add_argument(
        "--osd-min-confidence",
        type=float,
        default=OSD_MIN_ORIENTATION_CONFIDENCE,
        help="Legacy; coarse rotation always uses OSD min confidence 1.0 (model-repo)",
    )
    parser.add_argument(
        "--rotation-residual-confidence-threshold",
        type=float,
        default=ROTATION_RESIDUAL_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--mirror-confidence-threshold",
        type=float,
        default=MIRROR_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--skew-confidence-threshold",
        type=float,
        default=SKEW_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=QUALITY_REVIEW_THRESHOLD,
        help="Pages below this quality score are marked REVIEW_REQUIRED",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to handwritten_printed_convnext_tiny.pth",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = default_config()
    cfg.enable_debug = bool(args.debug)
    cfg.recursive = not bool(args.no_recursive)
    cfg.target_dpi = int(args.target_dpi)
    cfg.low_dpi_warning_threshold = int(args.low_dpi_threshold)
    cfg.max_skew_angle = float(args.max_skew_angle)
    cfg.osd_min_orientation_confidence = float(args.osd_min_confidence)
    cfg.rotation_residual_confidence_threshold = float(
        args.rotation_residual_confidence_threshold
    )
    cfg.mirror_confidence_threshold = float(args.mirror_confidence_threshold)
    cfg.skew_confidence_threshold = float(args.skew_confidence_threshold)
    cfg.quality_review_threshold = float(args.quality_threshold)
    cfg.classifier_model_path = args.model
    return cfg


def _check_tesseract_runtime() -> bool:
    try:
        import pytesseract
    except ImportError:
        print(
            f"ERROR: pytesseract is not installed for this Python:\n  {sys.executable}\n"
            "Install and run with the project venv, e.g.:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            "  .\\venv\\Scripts\\python.exe main.py --input ... --output ...",
            file=sys.stderr,
        )
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        print(
            "ERROR: Tesseract OCR binary not found on PATH "
            f"(pytesseract error: {exc}).\n"
            "Install Tesseract for Windows and add it to PATH, then run:\n"
            "  tesseract --list-langs   (must include 'osd')",
            file=sys.stderr,
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output / "preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    cfg = config_from_args(args)
    if not _check_tesseract_runtime():
        return 1

    from image_preprocessing.pipeline import run_pipeline

    report = run_pipeline(Path(args.input), output, cfg)
    print(f"Report: {report}")
    print(f"Corrected pages: {output / 'corrected_pages'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
