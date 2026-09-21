#!/usr/bin/env python3
"""Train binary Printed vs Handwritten ConvNeXt-Tiny on prepared_dataset/.

Requires prepare_dataset.py first.

    python train_page_classifier.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

ImageFile.LOAD_TRUNCATED_IMAGES = True

_PKG = Path(__file__).resolve().parent

# =============================================================================
# RUN CONFIG
# =============================================================================

PREPARED_DIR = _PKG / "prepared_dataset"
MANIFEST = PREPARED_DIR / "manifest.csv"
OUT_MODEL = _PKG / "models" / "page_printed_handwritten_convnext_tiny.pth"
OUT_META = _PKG / "models" / "page_printed_handwritten_metadata.json"

IMAGE_SIZE = 224
EPOCHS = 6
LR = 3e-4
VAL_FRACTION = 0.15
SEED = 42
NUM_WORKERS = 0
DEVICE = None  # None = auto
BATCH_SIZE = 8

LABEL_TO_INDEX = {"Printed": 0, "Handwritten": 1}
INDEX_TO_LABEL = {0: "Printed", 1: "Handwritten"}

# =============================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device():
    if DEVICE:
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox_rgb(image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


@dataclass
class Sample:
    path: Path
    label: int
    source: str


class PageDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.samples = samples
        aug = []
        if train:
            aug = [
                T.RandomHorizontalFlip(p=0.1),
                T.RandomRotation(degrees=3),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        self.tf = T.Compose(
            [
                T.Lambda(letterbox_rgb),
                T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
                *aug,
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Skip unreadable/truncated files by probing a few alternatives.
        for offset in range(len(self.samples)):
            sample = self.samples[(idx + offset) % len(self.samples)]
            try:
                with Image.open(sample.path) as im:
                    im = ImageOps.exif_transpose(im) or im
                    im = im.convert("RGB")
                    im.load()
                    tensor = self.tf(im)
                return tensor, sample.label
            except Exception:
                continue
        raise RuntimeError(f"Could not load any readable image near index {idx}")


def load_manifest(path: Path, root: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row["label"]
            if label_name not in LABEL_TO_INDEX:
                continue
            p = root / row["path"]
            if not p.is_file():
                continue
            samples.append(
                Sample(path=p, label=LABEL_TO_INDEX[label_name], source=row.get("source", ""))
            )
    return samples


def split_train_val(samples: list[Sample], val_fraction: float, seed: int):
    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for s in samples:
        by_label[s.label].append(s)
    train, val = [], []
    rng = random.Random(seed)
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction)) if len(items) > 5 else max(0, len(items) // 5)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_model() -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model


def class_weights(samples: list[Sample]) -> torch.Tensor:
    counts = [0, 0]
    for s in samples:
        counts[s.label] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class = {0: [0, 0], 1: [0, 0]}  # correct, total
    for batch, labels in loader:
        batch = batch.to(device)
        labels = labels.to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        for y, p in zip(labels.tolist(), pred.tolist()):
            per_class[y][1] += 1
            if y == p:
                per_class[y][0] += 1
    acc = correct / total if total else 0.0
    class_acc = {
        INDEX_TO_LABEL[k]: (v[0] / v[1] if v[1] else 0.0) for k, v in per_class.items()
    }
    return {"acc": acc, "class_acc": class_acc, "n": total}


def main() -> int:
    if not MANIFEST.is_file():
        print(f"Missing {MANIFEST}. Run prepare_dataset.py first.", file=sys.stderr)
        return 1

    set_seed(SEED)
    device = detect_device()
    samples = load_manifest(MANIFEST, PREPARED_DIR)
    if len(samples) < 10:
        print(f"Too few samples: {len(samples)}", file=sys.stderr)
        return 1

    train_s, val_s = split_train_val(samples, VAL_FRACTION, SEED)
    log(f"Device={device} train={len(train_s)} val={len(val_s)}")
    n0 = sum(1 for s in train_s if s.label == 0)
    n1 = sum(1 for s in train_s if s.label == 1)
    log(f"Train class counts Printed={n0} Handwritten={n1}")

    train_loader = DataLoader(
        PageDataset(train_s, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        PageDataset(val_s, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model().to(device)
    weights = class_weights(train_s).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = -1.0
    history = []
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            running += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        train_loss = running / max(1, seen)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        log(
            f"Epoch {epoch}/{EPOCHS} loss={train_loss:.4f} "
            f"val_acc={metrics['acc']:.4f} class_acc={metrics['class_acc']}"
        )
        if metrics["acc"] >= best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "convnext_tiny",
                    "classes": INDEX_TO_LABEL,
                    "image_size": IMAGE_SIZE,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                    "decision_threshold": 0.5,
                    "uncertain_min_confidence": 0.55,
                    "uncertain_margin": 0.08,
                    "label_rule": {
                        "Printed": "Full typed EHR/print, including signature-only pages",
                        "Handwritten": "Full handwritten pages AND filled forms with handwriting",
                    },
                    "train_counts": {"Printed": n0, "Handwritten": n1},
                    "best_val_acc": best_acc,
                },
                OUT_MODEL,
            )
            log(f"  saved best -> {OUT_MODEL}")

    meta = {
        "model_path": str(OUT_MODEL),
        "best_val_acc": best_acc,
        "history": history,
        "manifest": str(MANIFEST),
        "classes": INDEX_TO_LABEL,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {OUT_META}")
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
