"""Train the page model on manifest.csv and save it for quality_hw.

Each page has a type (Printed, Handwritten, Form, Visual, Blank, Uncertain).
Visibility and the handwritten-area percentage are learned only on the rows
where the labeler filled them in. An empty area is left empty; it is not
treated as zero.

Empty scanner pages are still marked BLANK from the image before this model
runs. Edit the paths in make_manifest.py, run that first, then:

    python quality_hw/train.py
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
from torchvision.models import ConvNeXt_Tiny_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quality_hw.hw_printed import PAGE_TAGS, VISIBILITY_TAGS, PageTagModel
from quality_hw.make_manifest import IMAGES_DIR, MANIFEST_PATH

# ============================== RUN CONFIG ==============================
EPOCHS = 6
BATCH_SIZE = 8
LR = 3e-4
VAL_FRACTION = 0.15
SEED = 42
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Windows hangs if this is raised
DEVICE = None  # None = cuda if available, else cpu
# ========================================================================

_PKG = Path(__file__).resolve().parent
OUT_MODEL = _PKG / "models" / "page_printed_handwritten_convnext_tiny.pth"
OUT_META = _PKG / "models" / "page_printed_handwritten_metadata.json"

LABEL_TO_INDEX = {name: i for i, name in enumerate(PAGE_TAGS)}
VIS_TO_INDEX = {name: i for i, name in enumerate(VISIBILITY_TAGS)}
INDEX_TO_LABEL = {i: name for name, i in LABEL_TO_INDEX.items()}
IGNORE = -1
IGNORE_AREA = -1.0


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
    visibility: int
    area: float


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
        for offset in range(len(self.samples)):
            sample = self.samples[(idx + offset) % len(self.samples)]
            try:
                with Image.open(sample.path) as im:
                    im = ImageOps.exif_transpose(im) or im
                    im = im.convert("RGB")
                    im.load()
                    return self.tf(im), sample.label, sample.visibility, sample.area
            except Exception:
                continue
        raise RuntimeError(f"Could not load any readable image near index {idx}")


def load_manifest(path: Path, root: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            label_name = row.get("label") or ""
            if label_name not in LABEL_TO_INDEX:
                continue
            image = root / row["path"]
            if not image.is_file():
                continue
            visibility = VIS_TO_INDEX.get((row.get("visibility") or "").strip(), IGNORE)
            raw_area = (row.get("handwritten_percent") or "").strip()
            area = float(raw_area) / 100.0 if raw_area else IGNORE_AREA
            samples.append(Sample(image, LABEL_TO_INDEX[label_name], visibility, area))
    return samples


def split_train_val(samples: list[Sample], val_fraction: float, seed: int):
    by_label: dict[int, list[Sample]] = {i: [] for i in INDEX_TO_LABEL}
    for sample in samples:
        by_label[sample.label].append(sample)
    train, val = [], []
    rng = random.Random(seed)
    for items in by_label.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction)) if len(items) > 5 else max(0, len(items) // 5)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_model() -> nn.Module:
    return PageTagModel(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)


def class_weights(samples: list[Sample]) -> torch.Tensor:
    n = len(LABEL_TO_INDEX)
    counts = [0] * n
    for sample in samples:
        counts[sample.label] += 1
    total = sum(counts)
    present = sum(1 for c in counts if c)
    weights = [total / (present * c) if c else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class = {i: [0, 0] for i in INDEX_TO_LABEL}
    for batch, labels, _visibility, _area in loader:
        batch = batch.to(device)
        labels = labels.to(device)
        type_logits, _vis_logits, _area_hat = model(batch)
        pred = type_logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        for y, p in zip(labels.tolist(), pred.tolist()):
            per_class[y][1] += 1
            if y == p:
                per_class[y][0] += 1
    class_acc = {INDEX_TO_LABEL[k]: (v[0] / v[1] if v[1] else 0.0) for k, v in per_class.items()}
    return {"acc": correct / total if total else 0.0, "class_acc": class_acc, "n": total}


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"Missing {MANIFEST_PATH}. Run: python quality_hw/make_manifest.py", file=sys.stderr)
        return 1

    set_seed(SEED)
    device = detect_device()
    samples = load_manifest(MANIFEST_PATH, IMAGES_DIR)
    if len(samples) < 10:
        print(f"Too few readable images: {len(samples)}", file=sys.stderr)
        return 1

    train_s, val_s = split_train_val(samples, VAL_FRACTION, SEED)
    counts = {name: sum(1 for s in train_s if s.label == i) for name, i in LABEL_TO_INDEX.items()}
    log(f"Device={device} train={len(train_s)} val={len(val_s)}")
    log("Train class counts " + " ".join(f"{name}={n}" for name, n in counts.items()))

    train_loader = DataLoader(PageDataset(train_s, train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(PageDataset(val_s, train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(device)
    type_loss = nn.CrossEntropyLoss(weight=class_weights(train_s).to(device))
    vis_loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = -1.0
    history = []
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch, labels, visibility, area in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            visibility = visibility.to(device)
            area = area.to(device)
            optim.zero_grad(set_to_none=True)
            type_logits, vis_logits, area_hat = model(batch)
            loss = type_loss(type_logits, labels)
            vis_mask = visibility >= 0
            if bool(vis_mask.any()):
                loss = loss + vis_loss(vis_logits[vis_mask], visibility[vis_mask])
            area_mask = area >= 0
            if bool(area_mask.any()):
                loss = loss + nn.functional.l1_loss(area_hat[area_mask], area[area_mask])
            loss.backward()
            optim.step()
            running += float(loss.item()) * int(labels.numel())
            seen += int(labels.numel())
        train_loss = running / max(1, seen)
        metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        log(f"Epoch {epoch}/{EPOCHS} loss={train_loss:.4f} val_acc={metrics['acc']:.4f} class_acc={metrics['class_acc']}")
        if metrics["acc"] >= best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "convnext_tiny",
                    "task": "page_tags",
                    "classes": INDEX_TO_LABEL,
                    "visibility_classes": {i: name for i, name in enumerate(VISIBILITY_TAGS)},
                    "image_size": IMAGE_SIZE,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                    "labeled_retrain": True,
                    "decision_threshold": 0.5,
                    "uncertain_min_confidence": 0.55,
                    "uncertain_margin": 0.08,
                    "train_counts": counts,
                    "best_val_acc": best_acc,
                },
                OUT_MODEL,
            )
            log(f"  saved best -> {OUT_MODEL}")

    OUT_META.write_text(json.dumps({
        "model_path": str(OUT_MODEL),
        "best_val_acc": best_acc,
        "history": history,
        "manifest": str(MANIFEST_PATH),
        "classes": INDEX_TO_LABEL,
        "labeled_retrain": True,
    }, indent=2), encoding="utf-8")
    log(f"Wrote {OUT_META}")
    log("Done. quality_hw/main.py will load this file. Empty pages are still marked BLANK before the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
