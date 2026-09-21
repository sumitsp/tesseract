"""Region RandomForest → page-wide printed / handwritten / mixed label."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from joblib import load

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "data.joblib"

REGION_TO_PAGE = {
    "Printed_extended": "PRINTED",
    "Handwritten_extended": "HANDWRITTEN",
    "Mixed_extended": "MIXED",
    "Other_extended": "OTHER",
}

METHOD_NAME = "prescription_rf_regions"


@dataclass
class PageTypeResult:
    document_type: str  # PRINTED | HANDWRITTEN | MIXED | UNCERTAIN
    method: str
    p_handwritten: float | None
    printed_area: int = 0
    handwritten_area: int = 0
    mixed_area: int = 0
    other_area: int = 0
    region_count: int = 0
    region_labels: dict[str, int] | None = None
    error: str | None = None


def ensure_model(model_path: Path | None = None) -> Path:
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if path.is_file() and path.stat().st_size > 0:
        return path
    try:
        from hw_printed_rf_test.train_model import train
    except ImportError:
        from train_model import train  # type: ignore

    return train()


def load_classifier(model_path: Path | None = None) -> Any:
    path = ensure_model(model_path)
    return load(path)


def _region_features(bgr_crop: np.ndarray) -> list[float]:
    """Same 5 features as the original prescription checker.py."""
    rows = int(bgr_crop.shape[0])
    cols = int(bgr_crop.shape[1])
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    myavg = 0.0
    for xx in range(cols):
        mycnt = 0
        for yy in range(rows):
            if bw[yy, xx] == 0:
                mycnt += 1
        myavg += (mycnt * 1.0) / rows
    myavg = myavg / cols if cols else 0.0

    # Original checker used a buggy slice; keep equivalent neighbor-change rate.
    change = 0.0
    for xx in range(rows):
        mycnt = 0
        for yy in range(cols - 1):
            if bw[xx, yy] != bw[xx, yy + 1]:
                mycnt += 1
        change += (mycnt * 1.0) / cols if cols else 0.0
    change = change / rows if rows else 0.0

    return [float(rows), float(cols), float(rows / cols) if cols else 0.0, float(myavg), float(change)]


def detect_and_classify_regions(
    image_bgr: np.ndarray,
    clf: Any,
) -> list[tuple[str, int, tuple[int, int, int, int]]]:
    """Return list of (region_label, area, (x, y, w, h))."""
    hgt, wdt = image_bgr.shape[:2]
    h_bw = hgt / float(wdt) if wdt else 1.0
    dim = (576, max(1, int(576 * h_bw)))
    img = cv2.resize(image_bgr, dim)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, ...] = 1
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
    gray = cv2.subtract(gray, opened)

    kernel = np.ones((5, 5), np.uint8)
    _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.dilate(gray, kernel, iterations=1)
    contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    results: list[tuple[str, int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue
        if w * h < 64:
            continue
        crop = img[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        feats = [_region_features(crop)]
        label = str(clf.predict(feats)[0])
        results.append((label, int(w * h), (int(x), int(y), int(w), int(h))))
    return results


def aggregate_page_label(
    regions: list[tuple[str, int, tuple[int, int, int, int]]],
) -> PageTypeResult:
    printed = handwritten = mixed = other = 0
    counts = {
        "Printed_extended": 0,
        "Handwritten_extended": 0,
        "Mixed_extended": 0,
        "Other_extended": 0,
    }
    for label, area, _ in regions:
        counts[label] = counts.get(label, 0) + 1
        if label == "Printed_extended":
            printed += area
        elif label == "Handwritten_extended":
            handwritten += area
        elif label == "Mixed_extended":
            mixed += area
        else:
            other += area

    content = printed + handwritten + mixed
    if content <= 0:
        return PageTypeResult(
            document_type="UNCERTAIN",
            method=METHOD_NAME,
            p_handwritten=None,
            printed_area=printed,
            handwritten_area=handwritten,
            mixed_area=mixed,
            other_area=other,
            region_count=len(regions),
            region_labels=counts,
        )

    p_hw = handwritten / float(content)
    print_r = printed / float(content)
    hw_r = handwritten / float(content)
    mixed_r = mixed / float(content)

    if mixed_r >= 0.15 or (print_r >= 0.15 and hw_r >= 0.15):
        page = "MIXED"
    elif hw_r >= 0.55:
        page = "HANDWRITTEN"
    elif print_r >= 0.55:
        page = "PRINTED"
    elif hw_r >= print_r and hw_r >= mixed_r:
        page = "HANDWRITTEN" if hw_r >= 0.40 else "UNCERTAIN"
    elif print_r >= hw_r:
        page = "PRINTED" if print_r >= 0.40 else "UNCERTAIN"
    else:
        page = "UNCERTAIN"

    return PageTypeResult(
        document_type=page,
        method=METHOD_NAME,
        p_handwritten=round(p_hw, 4),
        printed_area=printed,
        handwritten_area=handwritten,
        mixed_area=mixed,
        other_area=other,
        region_count=len(regions),
        region_labels=counts,
    )


def classify_page_image(image_bgr: np.ndarray, clf: Any | None = None) -> PageTypeResult:
    try:
        model = clf if clf is not None else load_classifier()
        regions = detect_and_classify_regions(image_bgr, model)
        return aggregate_page_label(regions)
    except Exception as exc:
        return PageTypeResult(
            document_type="UNCERTAIN",
            method=METHOD_NAME,
            p_handwritten=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def classify_image_path(path: Path | str, clf: Any | None = None) -> PageTypeResult:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return PageTypeResult(
            document_type="UNCERTAIN",
            method=METHOD_NAME,
            p_handwritten=None,
            error=f"Unreadable image: {path}",
        )
    return classify_page_image(image, clf=clf)
