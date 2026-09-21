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

from hw_printed_rf_test.classifier import (
    PageTypeResult,
    classify_page_image,
    load_classifier,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_PAGE_MODEL = MODELS_DIR / "page_printed_handwritten_convnext_tiny.pth"
METHOD_NAME = "page_convnext_tiny"
HYBRID_METHOD = "page_convnext_plus_rf"

INDEX_TO_CLASS = {0: "Printed", 1: "Handwritten"}

# Bias away from the printed-heavy training skew.
DEFAULT_DECISION_THRESHOLD = 0.32
# If page model is unsure-printed but RF finds real handwriting coverage, upgrade.
RF_UPGRADE_MIN_PAGE_P_HW = 0.12
RF_UPGRADE_MIN_HW_AREA_RATIO = 0.18
RF_UPGRADE_MIN_HW_REGIONS = 4


def letterbox_rgb(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


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
    image_size = int(ckpt.get("image_size") or 224)
    mean = tuple(ckpt.get("normalize_mean") or (0.485, 0.456, 0.406))
    std = tuple(ckpt.get("normalize_std") or (0.229, 0.224, 0.225))
    threshold = float(DEFAULT_DECISION_THRESHOLD)
    return {
        "model": model,
        "device": device,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "uncertain_min_confidence": float(ckpt.get("uncertain_min_confidence") or 0.50),
        "uncertain_margin": float(ckpt.get("uncertain_margin") or 0.06),
        "rf_model": None,
    }


def load_hybrid_classifier(model_path: Path | None = None) -> dict[str, Any]:
    bundle = load_page_classifier(model_path)
    bundle["rf_model"] = load_classifier()
    return bundle


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
    """Page ConvNeXt first; RF can upgrade filled forms to HANDWRITTEN."""
    model_bundle = bundle if bundle is not None else load_hybrid_classifier()
    page = classify_page_convnext(image_bgr, bundle=model_bundle)
    if page.error:
        return page
    if page.document_type == "HANDWRITTEN":
        return page

    p_hw = float(page.p_handwritten or 0.0)
    # Pure EHR pages usually sit near ~0 after printed-heavy training.
    # Filled forms often land in a low-but-nonzero band.
    if p_hw < RF_UPGRADE_MIN_PAGE_P_HW:
        return page

    rf = classify_page_image(image_bgr, clf=model_bundle.get("rf_model"))
    content = rf.printed_area + rf.handwritten_area + rf.mixed_area
    hw_ratio = (rf.handwritten_area / content) if content else 0.0
    hw_regions = 0
    if rf.region_labels:
        hw_regions = int(rf.region_labels.get("Handwritten_extended", 0))

    if (
        hw_ratio >= RF_UPGRADE_MIN_HW_AREA_RATIO
        and hw_regions >= RF_UPGRADE_MIN_HW_REGIONS
    ):
        return PageTypeResult(
            document_type="HANDWRITTEN",
            method=HYBRID_METHOD,
            p_handwritten=round(max(p_hw, hw_ratio), 4),
            printed_area=rf.printed_area,
            handwritten_area=rf.handwritten_area,
            mixed_area=rf.mixed_area,
            other_area=rf.other_area,
            region_count=rf.region_count,
            region_labels=rf.region_labels,
        )

    page.printed_area = rf.printed_area
    page.handwritten_area = rf.handwritten_area
    page.mixed_area = rf.mixed_area
    page.other_area = rf.other_area
    page.region_count = rf.region_count
    page.region_labels = rf.region_labels
    return page
