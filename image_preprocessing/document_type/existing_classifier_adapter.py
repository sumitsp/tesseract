"""Adapter around the existing ConvNeXt handwritten-vs-printed classifier.

This file does not reimplement classification. It converts a pipeline page
image into PNG bytes and calls ``hw_printed.classify_image_type``, which is
the source of truth (ConvNeXt-Tiny, 0=Printed, 1=Handwritten).

The existing classifier must see original page pixels. It letterboxes and
resizes internally. Do not binarize or otherwise preprocess for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_preprocessing.document_type import hw_printed
from image_preprocessing.utils.image_utils import encode_png_bytes, to_gray

# Below this fraction of dark pixels, the page is treated as blank for HW/printed.
# The ConvNeXt model was not trained on empty sheets and tends to say "Printed".
MIN_INK_RATIO_FOR_CLASSIFICATION = 0.004

LOGGER = logging.getLogger(__name__)

_LABEL_MAP = {
    "Printed": "PRINTED",
    "Handwritten": "HANDWRITTEN",
    "Uncertain": "UNCERTAIN",
}


@dataclass
class DocumentTypeResult:
    document_type: str  # PRINTED | HANDWRITTEN | UNCERTAIN | ERROR
    confidence: float | None
    method: str | None
    p_handwritten: float | None = None
    raw_label: str | None = None
    error: str | None = None


def default_model_path() -> Path:
    newer = (
        Path(__file__).resolve().parents[2]
        / "hw_printed_rf_test"
        / "models"
        / "page_printed_handwritten_convnext_tiny.pth"
    )
    if newer.is_file() and newer.stat().st_size > 1_000_000:
        return newer
    return Path(hw_printed.DEFAULT_MODEL_PATH)


def load_classifier(model_path: Path | None = None) -> Any:
    path = Path(model_path) if model_path is not None else default_model_path()
    LOGGER.info("Loading handwritten/printed classifier from %s", path)
    if path.name == "page_printed_handwritten_convnext_tiny.pth":
        from hw_printed_rf_test.page_classifier import load_hybrid_classifier

        return load_hybrid_classifier(path)
    return hw_printed.load_model(path)


def _ink_ratio(image: np.ndarray) -> float:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = float(np.mean(thr == 0))
    if thr.mean() < 127:
        ink = 1.0 - ink
    return ink


def classify_page(image: np.ndarray, model: Any | None = None) -> DocumentTypeResult:
    """Classify one page. Never raises — errors become document_type=ERROR."""
    try:
        if _ink_ratio(image) < MIN_INK_RATIO_FOR_CLASSIFICATION:
            return DocumentTypeResult(
                document_type="UNCERTAIN",
                confidence=0.0,
                method="blank_page",
                p_handwritten=None,
                raw_label="Uncertain",
            )
        if isinstance(model, dict) and "model" in model:
            from hw_printed_rf_test.page_classifier import classify_page_hybrid

            hybrid = classify_page_hybrid(image, bundle=model)
            return DocumentTypeResult(
                document_type=hybrid.document_type,
                confidence=hybrid.p_handwritten,
                method=hybrid.method,
                p_handwritten=hybrid.p_handwritten,
                raw_label=hybrid.document_type,
                error=hybrid.error,
            )
        image_bytes = encode_png_bytes(image)
        label, confidence, method = hw_printed.classify_image_type(image_bytes, model=model)
        mapped = _LABEL_MAP.get(label, label.upper() if label else "UNCERTAIN")
        p_hw = None
        if mapped == "HANDWRITTEN" and confidence is not None:
            p_hw = float(confidence)
        elif mapped == "PRINTED" and confidence is not None:
            p_hw = float(1.0 - confidence)
        return DocumentTypeResult(
            document_type=mapped,
            confidence=None if confidence is None else float(confidence),
            method=method,
            p_handwritten=p_hw,
            raw_label=label,
        )
    except Exception as exc:
        LOGGER.exception("Handwritten/printed classifier failed")
        return DocumentTypeResult(
            document_type="ERROR",
            confidence=None,
            method=None,
            error=str(exc),
        )
