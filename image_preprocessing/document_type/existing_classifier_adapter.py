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

import numpy as np

from image_preprocessing.document_type import hw_printed
from image_preprocessing.utils.image_utils import encode_png_bytes

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
    return Path(hw_printed.DEFAULT_MODEL_PATH)


def load_classifier(model_path: Path | None = None) -> Any:
    path = Path(model_path) if model_path is not None else default_model_path()
    LOGGER.info("Loading handwritten/printed classifier from %s", path)
    return hw_printed.load_model(path)


def classify_page(image: np.ndarray, model: Any | None = None) -> DocumentTypeResult:
    """Classify one page. Never raises — errors become document_type=ERROR."""
    try:
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
