"""Robust preprocessing for orientation analysis (OpenCV + NumPy only)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessBundle:
    """Analysis-resolution images/masks used by detectors."""

    gray: np.ndarray
    clahe: np.ndarray
    ink: np.ndarray           # text-like ink, white=ink on black
    ink_clean: np.ndarray     # border/noise cleaned
    scale: float              # analysis / original linear scale
    analysis_size: tuple[int, int]  # (w, h)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_for_analysis(gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return gray, 1.0
    scale = max_dim / float(longest)
    out = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out, scale


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by large-kernel background estimate to flatten uneven lighting."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 20) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg = np.maximum(bg, 1)
    norm = (gray.astype(np.float32) / bg.astype(np.float32)) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _ink_from_adaptive(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )


def _ink_from_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Decide polarity: ink should be minority (typically).
    inv = 255 - thr
    if cv2.countNonZero(inv) < cv2.countNonZero(thr):
        return inv
    return thr if cv2.countNonZero(thr) < cv2.countNonZero(inv) else inv


def combine_ink_masks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Union with light opening to suppress speckles without killing thin strokes."""
    combined = cv2.bitwise_or(a, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)


def remove_page_borders(ink: np.ndarray, margin_frac: float = 0.02) -> np.ndarray:
    """Zero out a thin outer border band (page edges / scanner frames)."""
    h, w = ink.shape
    m_y = max(2, int(h * margin_frac))
    m_x = max(2, int(w * margin_frac))
    out = ink.copy()
    out[:m_y, :] = 0
    out[-m_y:, :] = 0
    out[:, :m_x] = 0
    out[:, -m_x:] = 0
    return out


def remove_tiny_noise(ink: np.ndarray, min_area: int = 12) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = np.zeros_like(ink)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def preprocess(image: np.ndarray, analysis_max_dimension: int = 1800) -> PreprocessBundle:
    gray_full = to_gray(image)
    gray, scale = resize_for_analysis(gray_full, analysis_max_dimension)
    norm = normalize_illumination(gray)
    clahe = apply_clahe(norm)
    ink_a = _ink_from_adaptive(clahe)
    ink_b = _ink_from_otsu(clahe)
    ink = combine_ink_masks(ink_a, ink_b)
    ink_borderless = remove_page_borders(ink)
    ink_clean = remove_tiny_noise(ink_borderless, min_area=10)
    h, w = gray.shape
    return PreprocessBundle(
        gray=gray,
        clahe=clahe,
        ink=ink,
        ink_clean=ink_clean,
        scale=scale,
        analysis_size=(w, h),
    )
