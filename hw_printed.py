#!/usr/bin/env python3
"""Handwritten vs Printed classification (ported from autocoder OCR).

Uses RandomForest features from Tesseract word confidences:
  [avg_conf, std_conf, text_len, space_ratio]

Preprocess matches autocoder /ocr flow before classify:
  crop top 5% + bottom 2.5% → Sauvola adaptive binarization → features → .pkl

Model: models/image_type_classification.pkl
  classes: 0 = Handwritten, 1 = Printed
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer models/ subfolder; fall back to script-dir copy.
_MODEL_CANDIDATES = (
    SCRIPT_DIR / "models" / "image_type_classification.pkl",
    SCRIPT_DIR / "image_type_classification.pkl",
)
DEFAULT_MODEL_PATH = next(
    (p for p in _MODEL_CANDIDATES if p.is_file()),
    _MODEL_CANDIDATES[0],
)

_rf_model: Any = None
_model_load_attempted = False


def load_model(model_path: Path | None = None) -> Any:
    """Load joblib RandomForest; returns None if missing / unloadable."""
    global _rf_model, _model_load_attempted
    if _model_load_attempted and model_path is None:
        return _rf_model
    _model_load_attempted = True
    path = (model_path or DEFAULT_MODEL_PATH).resolve()
    if not path.is_file():
        print(f"Model not found: {path} — using heuristic fallback")
        _rf_model = None
        return None
    import joblib
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Trying to unpickle estimator.*version.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", module="sklearn.base")
        _rf_model = joblib.load(path)
    print(f"Loaded model: {path} ({type(_rf_model).__name__})")
    return _rf_model


def crop_top_margin(image: Image.Image, ratio: float = 0.05) -> Image.Image:
    if ratio <= 0:
        return image
    w, h = image.size
    cut = int(h * ratio)
    if cut <= 0 or cut >= h:
        return image
    return image.crop((0, cut, w, h))


def crop_bottom_margin(image: Image.Image, ratio: float = 0.025) -> Image.Image:
    if ratio <= 0:
        return image
    w, h = image.size
    cut = int(h * ratio)
    if cut <= 0 or cut >= h:
        return image
    return image.crop((0, 0, w, h - cut))


def apply_adaptive_binarization(
    image: Image.Image, window: int = 25, k: float = 0.2
) -> Image.Image:
    """Sauvola adaptive thresholding (same as autocoder ocr.py)."""
    gray = image.convert("L")
    img = np.asarray(gray, dtype=np.float64)

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    area = w * w
    r_dyn = 128.0

    padded = np.pad(img, half, mode="reflect")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral_sq = np.cumsum(np.cumsum(padded * padded, axis=0), axis=1)
    integral_sq = np.pad(
        integral_sq, ((1, 0), (1, 0)), mode="constant", constant_values=0
    )

    sums = (
        integral[w:, w:]
        - integral[:-w, w:]
        - integral[w:, :-w]
        + integral[:-w, :-w]
    )
    sums_sq = (
        integral_sq[w:, w:]
        - integral_sq[:-w, w:]
        - integral_sq[w:, :-w]
        + integral_sq[:-w, :-w]
    )

    mean = sums / area
    variance = np.maximum((sums_sq / area) - (mean**2), 0)
    std = np.sqrt(variance)
    threshold = mean * (1 + k * ((std / r_dyn) - 1))
    binary = (img > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def preprocess_for_classify(image: Image.Image) -> bytes:
    """Match autocoder page prep before classify_image_type."""
    img = crop_top_margin(image, 0.05)
    img = crop_bottom_margin(img, 0.025)
    bin_img = apply_adaptive_binarization(img)
    buf = io.BytesIO()
    bin_img.save(buf, format="PNG")
    return buf.getvalue()


def extract_ocr_features(image_bytes: bytes) -> list[float]:
    """Same feature vector used to train image_type_classification.pkl."""
    import pytesseract

    try:
        img = Image.open(io.BytesIO(image_bytes))
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in ocr_data["conf"] if str(c) != "-1"]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        std_conf = float(np.std(confs)) if confs else 0.0
        text = " ".join(ocr_data["text"]).strip()
        text_len = float(len(text))
        space_ratio = text.count(" ") / (len(text) + 1)
        return [avg_conf, std_conf, text_len, space_ratio]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def classify_image_type(
    image_bytes: bytes,
    model: Any | None = None,
    *,
    already_preprocessed: bool = False,
) -> tuple[str, float | None, str]:
    """
    Returns (label, confidence, method).
    label: \"Handwritten\" | \"Printed\"
    method: \"ml\" | \"heuristic\"
    """
    if not already_preprocessed:
        try:
            image_bytes = preprocess_for_classify(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            pass

    rf = model if model is not None else _rf_model
    if rf is None:
        rf = load_model()

    if rf is None:
        try:
            import pytesseract

            img = Image.open(io.BytesIO(image_bytes))
            ocr_data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )
            confs = [int(c) for c in ocr_data["conf"] if str(c) != "-1"]
            avg_conf = float(np.mean(confs)) if confs else 0.0
        except Exception:
            avg_conf = 0.0
        label = "Handwritten" if avg_conf < 50 else "Printed"
        conf = max(0.0, min(1.0, abs(avg_conf - 50) / 50.0))
        return label, round(conf, 4), "heuristic"

    feats = np.array(extract_ocr_features(image_bytes), dtype=float).reshape(1, -1)
    pred = int(rf.predict(feats)[0])
    label = "Handwritten" if pred == 0 else "Printed"
    conf: float | None = None
    if hasattr(rf, "predict_proba"):
        try:
            proba = rf.predict_proba(feats)[0]
            conf = float(max(proba))
        except Exception:
            conf = None
    return label, round(conf, 4) if conf is not None else None, "ml"


def classify_image_path(path: Path, model: Any | None = None) -> tuple[str, float | None, str]:
    img = Image.open(path)
    prepped = preprocess_for_classify(img)
    return classify_image_type(prepped, model=model, already_preprocessed=True)
