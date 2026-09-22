"""Page-level Printed vs Handwritten ConvNeXt classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import convnext_tiny

from hw_printed_rf_test.classifier import PageTypeResult

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_PAGE_MODEL = MODELS_DIR / "page_printed_handwritten_convnext_tiny.pth"
METHOD_NAME = "page_convnext_tiny"
HYBRID_METHOD = "page_convnext_plus_ink"

INDEX_TO_CLASS = {0: "Printed", 1: "Handwritten"}

# Page model alone is printed-heavy on filled forms.
DEFAULT_DECISION_THRESHOLD = 0.28
# Ink heuristic upgrade (filled forms have taller irregular strokes; EHR usually does not).
# Logos/graphics make tall blobs too, so require vertical spread across the page.
INK_SCORE_THRESHOLD = 0.48
INK_TALL_COMPONENT_MIN = 8
INK_MIN_Y_SPAN_FRAC = 0.22
INK_MIN_Y_STD = 55.0
# ConvNeXt was not trained on empty sheets and often predicts HANDWRITTEN.
BLANK_INK_RATIO_MAX = 0.004
BLANK_METHOD = "blank_page"


def letterbox_rgb(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def page_ink_ratio(image_bgr: np.ndarray) -> float:
    """Fraction of dark (ink-like) pixels after Otsu; used for blank detection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = float(np.mean(thr == 0))
    if thr.mean() < 127:
        ink = 1.0 - ink
    return ink


def handwriting_ink_evidence(image_bgr: np.ndarray) -> dict[str, float | int]:
    """Fast cue for filled-form / freehand ink vs clean typed EHR.

    Header logos and decorative graphics often look like a few tall irregular
    blobs clustered in one band. Real filled-form handwriting spreads down the
    page, so tall components must also show vertical dispersion.
    """
    h0, w0 = image_bgr.shape[:2]
    scale = 1000.0 / float(max(h0, w0))
    image = image_bgr
    if scale < 1.0:
        image = cv2.resize(
            image_bgr,
            (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
    )
    ink = cv2.subtract(bw, cv2.bitwise_or(horiz, vert))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _n, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    h, w = gray.shape
    page = float(h * w)
    good = 0
    tall = 0
    wide = 0
    area_sum = 0
    heights: list[int] = []
    tall_ys: list[float] = []
    for i in range(1, stats.shape[0]):
        _x, y, ww, hh, area = stats[i]
        if area < 20 or area > 0.03 * page:
            continue
        ar = ww / float(hh + 1e-6)
        if ar < 0.08 or ar > 10:
            continue
        fill = area / float(ww * hh + 1e-6)
        # Dense / logo-sized blobs (facility marks, seals) — not pen strokes.
        if area > 0.008 * page:
            continue
        if ww > 0.22 * w and hh > 0.06 * h:
            continue
        if fill > 0.72:
            continue
        # Header-band graphics: large-ish components only in the top strip.
        if y < 0.16 * h and area > 0.0035 * page and ww >= 40:
            continue
        good += 1
        area_sum += int(area)
        heights.append(int(hh))
        if hh >= 18 and ww >= 25:
            tall += 1
            tall_ys.append(float(y) + 0.5 * float(hh))
        if ar >= 1.8 and hh <= 40:
            wide += 1
    hstd = float(np.std(heights)) if len(heights) > 3 else 0.0
    ink_ratio = area_sum / page
    if len(tall_ys) >= 2:
        y_std = float(np.std(tall_ys))
        y_span_frac = float((max(tall_ys) - min(tall_ys)) / float(h))
    else:
        y_std = 0.0
        y_span_frac = 0.0
    # Logos cluster in one band (low y_span); handwriting spans the form.
    dispersed = y_span_frac >= INK_MIN_Y_SPAN_FRAC and y_std >= INK_MIN_Y_STD
    score = (
        0.40 * min(tall / 20.0, 1.0)
        + 0.20 * min(hstd / 8.0, 1.0)
        + 0.15 * min(wide / 30.0, 1.0)
        + 0.10 * min(ink_ratio / 0.05, 1.0)
        + 0.15 * min(y_span_frac / 0.45, 1.0)
    )
    if not dispersed:
        score = min(score, 0.35)
    return {
        "score": float(score),
        "tall": int(tall),
        "wide": int(wide),
        "good": int(good),
        "height_std": float(hstd),
        "ink_ratio": float(ink_ratio),
        "y_span_frac": float(y_span_frac),
        "y_std": float(y_std),
        "dispersed": int(1 if dispersed else 0),
    }


def _build_model() -> nn.Module:
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model


def load_page_classifier(model_path: Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path is not None else DEFAULT_PAGE_MODEL
    if not path.is_file():
        raise FileNotFoundError(
            f"Page model not found: {path}. Run git lfs pull or train_page_classifier.py"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = _build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "device": device,
        "image_size": int(ckpt.get("image_size") or 224),
        "mean": tuple(ckpt.get("normalize_mean") or (0.485, 0.456, 0.406)),
        "std": tuple(ckpt.get("normalize_std") or (0.229, 0.224, 0.225)),
        "threshold": float(DEFAULT_DECISION_THRESHOLD),
        "uncertain_min_confidence": float(ckpt.get("uncertain_min_confidence") or 0.50),
        "uncertain_margin": float(ckpt.get("uncertain_margin") or 0.06),
    }


def load_hybrid_classifier(model_path: Path | None = None) -> dict[str, Any]:
    return load_page_classifier(model_path)


def _preprocess(image_bgr: np.ndarray, bundle: dict[str, Any]) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tf = T.Compose(
        [
            T.Lambda(letterbox_rgb),
            T.Resize(
                (bundle["image_size"], bundle["image_size"]),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
            T.Normalize(mean=bundle["mean"], std=bundle["std"]),
        ]
    )
    return tf(pil).unsqueeze(0)


def classify_page_convnext(
    image_bgr: np.ndarray,
    bundle: dict[str, Any] | None = None,
) -> PageTypeResult:
    try:
        ink_r = page_ink_ratio(image_bgr)
        if ink_r < BLANK_INK_RATIO_MAX:
            return PageTypeResult(
                document_type="UNCERTAIN",
                method=BLANK_METHOD,
                p_handwritten=0.0,
                region_labels={"blank": 1, "ink_ratio": ink_r},
            )
        model_bundle = bundle if bundle is not None else load_page_classifier()
        tensor = _preprocess(image_bgr, model_bundle).to(model_bundle["device"])
        with torch.inference_mode():
            logits = model_bundle["model"](tensor)
            proba = torch.softmax(logits, dim=-1)[0]
        p_hw = float(proba[1].item())
        t = float(model_bundle["threshold"])
        margin = float(model_bundle["uncertain_margin"])
        min_conf = float(model_bundle["uncertain_min_confidence"])

        if p_hw >= t + margin:
            label, conf = "HANDWRITTEN", p_hw
        elif p_hw <= t - margin:
            label, conf = "PRINTED", 1.0 - p_hw
        else:
            if p_hw >= t:
                label, conf = "HANDWRITTEN", p_hw
            else:
                label, conf = "PRINTED", 1.0 - p_hw
            if conf < min_conf:
                label = "UNCERTAIN"

        return PageTypeResult(
            document_type=label,
            method=METHOD_NAME,
            p_handwritten=round(p_hw, 4),
        )
    except Exception as exc:
        return PageTypeResult(
            document_type="UNCERTAIN",
            method=METHOD_NAME,
            p_handwritten=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def classify_page_hybrid(
    image_bgr: np.ndarray,
    bundle: dict[str, Any] | None = None,
) -> PageTypeResult:
    """Page ConvNeXt first; ink evidence upgrades filled forms to HANDWRITTEN."""
    model_bundle = bundle if bundle is not None else load_hybrid_classifier()
    page = classify_page_convnext(image_bgr, bundle=model_bundle)
    if page.error or page.document_type == "HANDWRITTEN" or page.method == BLANK_METHOD:
        return page

    ink = handwriting_ink_evidence(image_bgr)
    score = float(ink["score"])
    tall = int(ink["tall"])
    dispersed = bool(ink.get("dispersed"))
    # Only upgrade when ink is both strong and spread down the page (not a logo).
    if dispersed and (score >= INK_SCORE_THRESHOLD or tall >= INK_TALL_COMPONENT_MIN):
        return PageTypeResult(
            document_type="HANDWRITTEN",
            method=HYBRID_METHOD,
            p_handwritten=round(max(float(page.p_handwritten or 0.0), score), 4),
            region_count=int(ink["good"]),
            region_labels={
                "ink_score": score,
                "tall_components": tall,
                "wide_components": int(ink["wide"]),
                "height_std": float(ink["height_std"]),
                "y_span_frac": float(ink["y_span_frac"]),
                "y_std": float(ink["y_std"]),
            },
        )
    page.region_labels = {
        "ink_score": score,
        "tall_components": tall,
        "wide_components": int(ink["wide"]),
        "height_std": float(ink["height_std"]),
        "y_span_frac": float(ink["y_span_frac"]),
        "y_std": float(ink["y_std"]),
    }
    return page
