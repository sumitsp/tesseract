#!/usr/bin/env python3
"""Preprocess document pages before OCR.

Edit RUN CONFIG below, then:

    python main.py
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
    BlobSettings,
    PipelineConfig,
    default_config,
)

# =============================================================================
# RUN CONFIG — edit these (no .env)
# =============================================================================

# "local" = file or folder on disk   |   "blob" = Azure chart folders
INPUT_SOURCE = "local"

# Local input (used when INPUT_SOURCE == "local")
LOCAL_INPUT = Path(r"C:\Users\sumit.pandey\Downloads\Pg4.tif")

# Where corrected PNGs + Excel report are written
OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\image_preprocessing\output")

# Azure Blob (used when INPUT_SOURCE == "blob")
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"
START_FROM = ""

# Optional: full path to handwritten_printed_convnext_tiny.pth (None = default)
CLASSIFIER_MODEL: Path | None = None

# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Document-page preprocessor. Paths and blob settings are in main.py RUN CONFIG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=ENABLE_DEBUG,
        help="Write intermediate debug images under output/debug/",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Local folder only: do not recurse into subfolders",
    )
    parser.add_argument("--target-dpi", type=int, default=TARGET_DPI)
    parser.add_argument(
        "--low-dpi-threshold",
        type=int,
        default=LOW_DPI_WARNING_THRESHOLD,
    )
    parser.add_argument("--max-skew-angle", type=float, default=MAX_SKEW_ANGLE)
    parser.add_argument(
        "--osd-min-confidence",
        type=float,
        default=OSD_MIN_ORIENTATION_CONFIDENCE,
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
    cfg.classifier_model_path = CLASSIFIER_MODEL
    source = INPUT_SOURCE.strip().lower()
    if source == "blob":
        cfg.blob = BlobSettings(
            storage_account=STORAGE_ACCOUNT.strip(),
            container_name=CONTAINER_NAME.strip(),
            prefix=PREFIX.strip(),
            start_from=START_FROM.strip(),
        )
    elif source != "local":
        raise ValueError(f'INPUT_SOURCE must be "local" or "blob", got: {INPUT_SOURCE!r}')
    return cfg


def _check_tesseract_runtime() -> bool:
    from image_preprocessing.utils.tesseract_config import (
        configure_tesseract,
        tesseract_help_message,
    )

    try:
        import pytesseract
    except ImportError:
        print(
            f"ERROR: pytesseract is not installed for this Python:\n  {sys.executable}\n"
            "  .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return False
    found = configure_tesseract()
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        print(f"ERROR: Tesseract OCR binary not found ({exc}).\n", file=sys.stderr)
        if found:
            print(f"  Tried: {found}", file=sys.stderr)
        print(tesseract_help_message(), file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output = Path(OUTPUT_DIR).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output / "preprocessing.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    cfg = config_from_args(args)
    if not _check_tesseract_runtime():
        return 1

    from image_preprocessing.pipeline import run_blob_pipeline, run_pipeline

    source = INPUT_SOURCE.strip().lower()
    if source == "blob":
        if not cfg.blob or not cfg.blob.storage_account or not cfg.blob.container_name:
            print("ERROR: Set STORAGE_ACCOUNT and CONTAINER_NAME in main.py RUN CONFIG.", file=sys.stderr)
            return 1
        report = run_blob_pipeline(output, cfg)
        print(f"Report: {report}")
        print(f"Corrected pages: {output} (one subfolder per chart)")
    else:
        input_path = Path(LOCAL_INPUT).expanduser().resolve()
        if not input_path.exists():
            print(f"ERROR: LOCAL_INPUT does not exist: {input_path}", file=sys.stderr)
            return 1
        report = run_pipeline(input_path, output, cfg)
        print(f"Report: {report}")
        print(f"Corrected pages: {output / 'corrected_pages'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
