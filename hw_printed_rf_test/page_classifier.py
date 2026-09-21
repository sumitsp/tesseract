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

INDEX_TO_CLASS = {0: "Printed", 1: "Handwritten"}


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
    threshold = float(ckpt.get("decision_threshold") or 0.5)
    return {
        "model": model,
        "device": device,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "uncertain_min_confidence": float(ckpt.get("uncertain_min_confidence") or 0.55),
        "uncertain_margin": float(ckpt.get("uncertain_margin") or 0.08),
    }


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
