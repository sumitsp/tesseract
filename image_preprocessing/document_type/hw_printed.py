#!/usr/bin/env python3
"""Production handwritten-vs-printed page classifier.

Canonical class mapping (used everywhere — training, evaluation, inference):

    0 = Printed
    1 = Handwritten

This replaces the previous Tesseract-OCR + RandomForest pipeline.
The model is ConvNeXt-Tiny fine-tuned on page pixels.

API (compatible with the Azure CSV pipeline):

    model = load_model(path)
    label, confidence, method = classify_image_type(image_bytes, model=model)
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger(__name__)

INDEX_TO_CLASS: dict[int, str] = {0: "Printed", 1: "Handwritten"}
CLASS_TO_INDEX: dict[str, int] = {"Printed": 0, "Handwritten": 1}

# isPrinted.json stores 1 if the page IS printed and 0 if it is handwritten.
# Production RandomForest used the same convention. The Colab notebook comment
# ("0 = printed, 1 = handwritten") was inverted and must not be used.
ISPRINTED_JSON_TO_NAME: dict[int, str] = {0: "Handwritten", 1: "Printed"}

METHOD_NAME = "convnext_tiny"
ARCHITECTURE = "convnext_tiny"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL_RGB = (255, 255, 255)

# Probability of class 1 (Handwritten) at or above this value → Handwritten.
DEFAULT_DECISION_THRESHOLD = 0.50
# Predicted-class probability below this value → Uncertain.
DEFAULT_UNCERTAIN_MIN_CONFIDENCE = 0.55
# |P(Handwritten) - threshold| below this → Uncertain.
DEFAULT_UNCERTAIN_MARGIN = 0.08

SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_CANDIDATES = (
    SCRIPT_DIR / "models" / "handwritten_printed_convnext_tiny.pth",
    SCRIPT_DIR / "models" / "best_model.pth",
)
DEFAULT_MODEL_PATH = next(
    (p for p in _MODEL_CANDIDATES if p.is_file()),
    _MODEL_CANDIDATES[0],
)

_bundle: "ClassifierBundle | None" = None
_load_attempted = False


@dataclass
class ClassifierBundle:
    """Loaded inference artifact. Passed as `model=` to classify_image_type()."""

    torch_model: Any
    device: Any
    image_size: int = DEFAULT_IMAGE_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD
    uncertain_min_confidence: float = DEFAULT_UNCERTAIN_MIN_CONFIDENCE
    uncertain_margin: float = DEFAULT_UNCERTAIN_MARGIN
    classes: dict[int, str] = field(default_factory=lambda: dict(INDEX_TO_CLASS))
    architecture: str = ARCHITECTURE
    metadata: dict[str, Any] = field(default_factory=dict)

    def eval(self) -> "ClassifierBundle":
        self.torch_model.eval()
        return self


def detect_device(prefer: str | None = None):
    """Return a torch.device. Never requires CUDA."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int = 2, pretrained: bool = False):
    """Build ConvNeXt-Tiny with a 2-class head."""
    import torch.nn as nn
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_later_stages(model, n_stages: int = 2) -> None:
    """Unfreeze the last `n_stages` feature blocks of ConvNeXt."""
    n_stages = max(1, min(n_stages, len(model.features)))
    for parameter in model.features[-n_stages:].parameters():
        parameter.requires_grad = True


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty image bytes")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes: {exc}") from exc
    image = ImageOps.exif_transpose(image) or image
    return image.convert("RGB")


def letterbox_rgb(
    image: Image.Image,
    fill: tuple[int, int, int] = PAD_FILL_RGB,
) -> Image.Image:
    """Pad to square, preserving aspect ratio. Does not stretch handwriting."""
    image = image.convert("RGB")
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError(f"Invalid image size: {image.size}")
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def preprocess_for_model(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
):
    """Deterministic inference transform: RGB → letterbox → resize → normalize."""
    import torch
    from torchvision import transforms as T

    transform = T.Compose(
        [
            T.Lambda(letterbox_rgb),
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
            T.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )
    tensor = transform(image)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Preprocess did not return a tensor")
    return tensor


def _bundle_from_checkpoint(checkpoint: dict[str, Any], device) -> ClassifierBundle:
    import torch

    metadata = checkpoint.get("metadata") or {}
    image_size = int(checkpoint.get("image_size") or metadata.get("image_size") or DEFAULT_IMAGE_SIZE)
    mean = tuple(checkpoint.get("normalize_mean") or metadata.get("normalize_mean") or IMAGENET_MEAN)
    std = tuple(checkpoint.get("normalize_std") or metadata.get("normalize_std") or IMAGENET_STD)
    threshold = float(
        checkpoint.get("decision_threshold")
        if checkpoint.get("decision_threshold") is not None
        else metadata.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    )
    uncertain = float(
        checkpoint.get("uncertain_min_confidence")
        if checkpoint.get("uncertain_min_confidence") is not None
        else metadata.get("uncertain_min_confidence", DEFAULT_UNCERTAIN_MIN_CONFIDENCE)
    )
    margin = float(
        checkpoint.get("uncertain_margin")
        if checkpoint.get("uncertain_margin") is not None
        else metadata.get("uncertain_margin", DEFAULT_UNCERTAIN_MARGIN)
    )
    raw_classes = checkpoint.get("classes") or metadata.get("classes") or INDEX_TO_CLASS
    classes = {int(k): str(v) for k, v in dict(raw_classes).items()}
    if classes.get(0) != "Printed" or classes.get(1) != "Handwritten":
        raise ValueError(
            f"Checkpoint class mapping is not canonical 0=Printed, 1=Handwritten: {classes}"
        )

    torch_model = build_model(num_classes=2, pretrained=False)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint is missing model_state_dict")
    torch_model.load_state_dict(state)
    torch_model.to(device)
    torch_model.eval()
    return ClassifierBundle(
        torch_model=torch_model,
        device=device,
        image_size=image_size,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        decision_threshold=threshold,
        uncertain_min_confidence=uncertain,
        uncertain_margin=margin,
        classes=classes,
        architecture=str(checkpoint.get("architecture") or metadata.get("architecture") or ARCHITECTURE),
        metadata=metadata,
    )


def load_model(model_path: Path | str | None = None, device=None) -> ClassifierBundle:
    """Load the ConvNeXt checkpoint once and reuse it.

    `model` in classify_image_type() is this bundle, not a raw sklearn pickle.
    """
    global _bundle, _load_attempted
    import torch

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    path = path.resolve()
    if _bundle is not None and model_path is None:
        return _bundle
    _load_attempted = True
    if path.suffix.lower() in {".pkl", ".joblib"}:
        raise ValueError(
            f"Refusing to load OCR RandomForest pickle {path}. "
            "Use handwritten_printed_convnext_tiny.pth from this project."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"ConvNeXt model not found: {path}. Train it with "
            "training/train_handwriting_classifier.py"
        )
    resolved_device = device if device is not None else detect_device()
    LOGGER.info("Loading classifier from %s on %s", path, resolved_device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unrecognized checkpoint format: {path}")
    bundle = _bundle_from_checkpoint(checkpoint, resolved_device)
    _bundle = bundle
    LOGGER.info(
        "Loaded %s (threshold=%.3f, uncertain_margin=%.3f, min_conf=%.3f)",
        bundle.architecture,
        bundle.decision_threshold,
        bundle.uncertain_margin,
        bundle.uncertain_min_confidence,
    )
    return bundle


def probabilities_from_logits(logits):
    import torch

    return torch.softmax(logits, dim=-1)


def decide_label(
    p_handwritten: float,
    decision_threshold: float,
    uncertain_min_confidence: float,
    uncertain_margin: float = 0.08,
) -> tuple[str, float]:
    """Map P(Handwritten) to (label, confidence).

    confidence is the probability of the decided class (Printed or Handwritten).
    Uncertain is used only near the decision boundary, or when that class
    probability is below the saved floor.
    """
    if p_handwritten >= decision_threshold:
        label = "Handwritten"
        confidence = float(p_handwritten)
    else:
        label = "Printed"
        confidence = float(1.0 - p_handwritten)
    near_boundary = abs(p_handwritten - decision_threshold) < uncertain_margin
    if near_boundary or confidence < uncertain_min_confidence:
        return "Uncertain", round(confidence, 4)
    return label, round(confidence, 4)


def classify_tensor_batch(batch, bundle: ClassifierBundle):
    """Run a preprocessed NCHW tensor batch. Returns P(Handwritten) per row."""
    import torch

    bundle.torch_model.eval()
    with torch.inference_mode():
        batch = batch.to(bundle.device, non_blocking=False)
        logits = bundle.torch_model(batch)
        proba = probabilities_from_logits(logits)
    return proba.detach().cpu()


def classify_image_type(
    image_bytes: bytes,
    model: Any | None = None,
    *,
    already_preprocessed: bool = False,
) -> tuple[str, float, str]:
    """Classify one document page.

    Returns (label, confidence, method) where label is
    Printed | Handwritten | Uncertain.

    `already_preprocessed` is accepted for API compatibility with the old
    OCR pipeline and is ignored — this classifier must see the original pixels.
    """
    del already_preprocessed  # OCR binarization is intentionally not applied.
    bundle = model if isinstance(model, ClassifierBundle) else None
    if bundle is None:
        bundle = load_model()

    image = decode_image(image_bytes)
    tensor = preprocess_for_model(
        image,
        image_size=bundle.image_size,
        mean=bundle.mean,
        std=bundle.std,
    ).unsqueeze(0)
    proba = classify_tensor_batch(tensor, bundle)[0]
    p_handwritten = float(proba[CLASS_TO_INDEX["Handwritten"]])
    label, confidence = decide_label(
        p_handwritten,
        bundle.decision_threshold,
        bundle.uncertain_min_confidence,
        bundle.uncertain_margin,
    )
    return label, confidence, METHOD_NAME


def classify_image_path(path: Path | str, model: Any | None = None) -> tuple[str, float, str]:
    return classify_image_type(Path(path).read_bytes(), model=model)


def classify_image_type_batch(
    image_bytes_list: Sequence[bytes],
    model: Any | None = None,
    batch_size: int = 16,
) -> list[tuple[str, float, str]]:
    """Batched inference for production throughput."""
    import torch

    bundle = model if isinstance(model, ClassifierBundle) else load_model()
    results: list[tuple[str, float, str]] = []
    tensors = []
    for image_bytes in image_bytes_list:
        image = decode_image(image_bytes)
        tensors.append(
            preprocess_for_model(
                image,
                image_size=bundle.image_size,
                mean=bundle.mean,
                std=bundle.std,
            )
        )
    stacked = torch.stack(tensors, dim=0)
    probabilities = []
    for start in range(0, len(stacked), batch_size):
        chunk = stacked[start : start + batch_size]
        probabilities.append(classify_tensor_batch(chunk, bundle))
    proba = torch.cat(probabilities, dim=0)
    for row in proba:
        p_handwritten = float(row[CLASS_TO_INDEX["Handwritten"]])
        label, confidence = decide_label(
            p_handwritten,
            bundle.decision_threshold,
            bundle.uncertain_min_confidence,
            bundle.uncertain_margin,
        )
        results.append((label, confidence, METHOD_NAME))
    return results


def load_metadata(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
