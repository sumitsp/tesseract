from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def env_path(key: str, default: str = "") -> Path:
    raw = env(key, default)
    if not raw:
        raise SystemExit(f"Missing required path in .env: {key}")
    return Path(raw).expanduser()


def env_path_optional(key: str, default: str = "") -> Path | None:
    raw = env(key, default)
    if not raw:
        return None
    return Path(raw).expanduser()


STORAGE_ACCOUNT = env("AZURE_STORAGE_ACCOUNT", "azsadve2aipoc")
CONTAINER_NAME = env("AZURE_CONTAINER_NAME", "imaging-pipeline")
PREFIX = env("AZURE_BLOB_PREFIX", "Raw_Input/Run1/Batch1/DEID_PNGs/")
if PREFIX and not PREFIX.endswith("/"):
    PREFIX += "/"

START_FROM = env("START_FROM", "")

DOCLING_FORMAT_AND_RAPID_OUTPUT_DIR = env_path(
    "DOCLING_FORMAT_AND_RAPID_OUTPUT_DIR",
    str(Path.home() / "Desktop" / "Imaging" / "docling_format_and_rapid"),
)
RAPID_OUTPUT_DIR = env_path(
    "RAPID_OUTPUT_DIR",
    str(Path.home() / "Desktop" / "Imaging" / "rapid"),
)
TESSERACT_OUTPUT_DIR = env_path(
    "TESSERACT_OUTPUT_DIR",
    str(Path.home() / "Desktop" / "Imaging" / "tesseract"),
)
ROTATION_CORRECTED_OUTPUT_DIR = env_path(
    "ROTATION_CORRECTED_OUTPUT_DIR",
    str(Path.home() / "Desktop" / "Imaging" / "rotation_corrected"),
)
ROTATION_CORRECTED_CSV_PATH = env_path(
    "ROTATION_CORRECTED_CSV_PATH",
    str(Path.home() / "Desktop" / "Imaging" / "rotation_corrected_report.csv"),
)

RAPID_MODELS_DIR = env_path(
    "RAPID_MODELS_DIR",
    str(Path.home() / "Desktop" / "Imaging" / "rapidocr_models"),
)
CSV_PATH_HW_PRINTED = env_path(
    "CSV_PATH_HW_PRINTED",
    str(Path.home() / "Desktop" / "hw_printed.csv"),
)
HW_MODEL_PATH = env_path(
    "HW_MODEL_PATH",
    str(ROOT / "image_type_classification.pkl"),
)
if not HW_MODEL_PATH.is_absolute():
    HW_MODEL_PATH = (ROOT / HW_MODEL_PATH).resolve()
TESSERACT_CMD = env_path_optional("TESSERACT_CMD", "")
