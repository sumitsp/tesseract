"""Engineering image-quality score for a document page.

This is NOT a scientifically absolute quality measurement. It is a transparent
weighted combination of measurable submetrics so a reader can see why a page
scored 8.7 vs 4.2.

Each submetric is mapped to 0–10, then combined:

    quality_score =
        0.16 * usable_dimension_score
      + 0.14 * effective_resolution_score
      + 0.16 * sharpness_score
      + 0.10 * blur_score          (high = less blur)
      + 0.12 * contrast_score
      + 0.08 * noise_score         (high = less noise)
      + 0.08 * clipping_score      (high = less clipping)
      + 0.06 * brightness_score
      + 0.06 * text_visibility_score
      + 0.04 * compression_score   (high = fewer JPEG-block artifacts)

Unknown DPI is not invented. Effective resolution then uses pixel geometry
relative to a US-letter page as a fallback estimate and is labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from image_preprocessing.utils.image_utils import to_gray


@dataclass
class QualityResult:
    quality_score: float
    sharpness_score: float
    contrast_score: float
    noise_score: float
    blur_score: float
    brightness_score: float
    clipping_score: float
    text_visibility_score: float
    compression_score: float
    usable_dimension_score: float
    effective_resolution_score: float
    width_px: int
    height_px: int
    input_dpi: float | None
    estimated_dpi_from_pixels: float | None
    warnings: list[str] = field(default_factory=list)
    breakdown: str = ""


def _clip_10(value: float) -> float:
    return float(np.clip(value, 0.0, 10.0))


def _map_range(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clip_10(10.0 * (value - lo) / (hi - lo))


def _usable_dimension_score(width: int, height: int) -> float:
    shortest = min(width, height)
    # Rough OCR usability of the shorter side in pixels.
    return _map_range(float(shortest), 400.0, 2200.0)


def _effective_resolution_score(
    width: int,
    height: int,
    dpi: float | None,
    target_dpi: float,
) -> tuple[float, float | None]:
    if dpi is not None and dpi > 0:
        # 150 DPI → ~3.75, 400 DPI → 10. Do not reward values far above 400.
        score = _clip_10(10.0 * min(dpi, target_dpi) / float(target_dpi))
        return score, None
    # Fallback: assume a letter-size page. This is an estimate, not a claim.
    letter_w, letter_h = 8.5, 11.0
    est = max(width / letter_w, height / letter_h)
    # Swap if the page is landscape-ish.
    if width > height:
        est = max(width / letter_h, height / letter_w)
    score = _clip_10(10.0 * min(est, target_dpi) / float(target_dpi))
    return score, float(est)


def _sharpness_and_blur(gray: np.ndarray) -> tuple[float, float, float]:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(lap.var())
    # Log-scale: very soft scans ~10, typical office scans ~100–800, crisp ~1500+.
    sharpness = _clip_10(np.log1p(variance) / np.log1p(1800.0) * 10.0)
    # FFT high-frequency energy as an independent blur check.
    small = gray
    if max(gray.shape) > 512:
        scale = 512 / float(max(gray.shape))
        small = cv2.resize(
            gray,
            (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    f = np.fft.fftshift(np.fft.fft2(small.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_hi = 0.35 * min(h, w)
    r_lo = 0.10 * min(h, w)
    hi = float(mag[radius >= r_hi].mean()) if np.any(radius >= r_hi) else 0.0
    lo = float(mag[radius <= r_lo].mean()) if np.any(radius <= r_lo) else 1.0
    hf_ratio = hi / (lo + 1e-9)
    blur_clarity = _clip_10(np.log1p(hf_ratio * 50.0) / np.log1p(8.0) * 10.0)
    return sharpness, blur_clarity, variance


def _contrast_score(gray: np.ndarray) -> float:
    p5, p95 = np.percentile(gray, [5, 95])
    spread = float(p95 - p5) / 255.0
    std = float(gray.std()) / 128.0
    return _clip_10(10.0 * (0.7 * spread + 0.3 * min(std, 1.0)))


def _noise_score(gray: np.ndarray) -> float:
    """High score = less noise. Residual vs median filter."""
    med = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, med)
    # Text edges also survive a 3x3 median. Compare residual to edge mask.
    edges = cv2.Canny(gray, 60, 160)
    non_edge = residual[edges == 0]
    if non_edge.size == 0:
        return 8.0
    rms = float(np.sqrt(np.mean(non_edge.astype(np.float32) ** 2)))
    # rms ~ 0–2 clean, ~8+ noisy.
    penalty = _map_range(rms, 1.5, 14.0)
    return _clip_10(10.0 - penalty)


def _clipping_score(gray: np.ndarray) -> float:
    lo = float(np.mean(gray <= 3))
    hi = float(np.mean(gray >= 252))
    clipped = lo + hi
    return _clip_10(10.0 - _map_range(clipped, 0.01, 0.25))


def _brightness_score(gray: np.ndarray) -> float:
    mean = float(gray.mean())
    # Ideal document paper is bright-but-not-blown, ink still visible.
    if 140 <= mean <= 210:
        return 10.0
    if mean < 140:
        return _map_range(mean, 40.0, 140.0)
    return _clip_10(10.0 - _map_range(mean, 210.0, 250.0))


def _text_visibility_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_ratio = float(np.mean(thr == 0))
    if thr.mean() < 127:
        ink_ratio = 1.0 - ink_ratio
    # Usable document ink usually sits between ~0.5% and ~20%.
    if 0.008 <= ink_ratio <= 0.20:
        return 10.0
    if ink_ratio < 0.008:
        return _map_range(ink_ratio, 0.0005, 0.008)
    return _clip_10(10.0 - _map_range(ink_ratio, 0.20, 0.55))


def _compression_score(gray: np.ndarray) -> float:
    """Detect 8x8 JPEG blockiness. High score = fewer artifacts."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return 8.0
    hh, ww = h - (h % 8), w - (w % 8)
    g = gray[:hh, :ww].astype(np.float32)
    # Vertical block boundaries at x = 8, 16, ...
    if ww > 16:
        boundary = np.abs(g[:, 7:-1:8] - g[:, 8::8]).mean()
        interior = np.abs(g[:, 3:-5:8] - g[:, 4:-4:8]).mean()
    else:
        return 8.0
    ratio = float(boundary / (interior + 1e-6))
    # ratio ~1 = no blocking, ~2+ = visible blocks.
    penalty = _map_range(ratio, 1.15, 2.4)
    return _clip_10(10.0 - penalty)


def analyze_quality(
    image: np.ndarray,
    input_dpi: float | None,
    *,
    target_dpi: float = 400.0,
    low_dpi_threshold: float = 150.0,
    quality_review_threshold: float = 4.0,
) -> QualityResult:
    gray = to_gray(image)
    h, w = gray.shape
    usable = _usable_dimension_score(w, h)
    effective, estimated_dpi = _effective_resolution_score(w, h, input_dpi, target_dpi)
    sharpness, blur_clarity, _lap_var = _sharpness_and_blur(gray)
    contrast = _contrast_score(gray)
    noise = _noise_score(gray)
    clipping = _clipping_score(gray)
    brightness = _brightness_score(gray)
    text_vis = _text_visibility_score(gray)
    compression = _compression_score(gray)

    weights = {
        "usable_dimension": 0.16,
        "effective_resolution": 0.14,
        "sharpness": 0.16,
        "blur": 0.10,
        "contrast": 0.12,
        "noise": 0.08,
        "clipping": 0.08,
        "brightness": 0.06,
        "text_visibility": 0.06,
        "compression": 0.04,
    }
    parts = {
        "usable_dimension": usable,
        "effective_resolution": effective,
        "sharpness": sharpness,
        "blur": blur_clarity,
        "contrast": contrast,
        "noise": noise,
        "clipping": clipping,
        "brightness": brightness,
        "text_visibility": text_vis,
        "compression": compression,
    }
    score = sum(weights[k] * parts[k] for k in weights)
    score = round(_clip_10(score), 2)

    warnings: list[str] = []
    if input_dpi is not None and input_dpi < low_dpi_threshold:
        warnings.append("LOW_DPI")
        warnings.append("DPI_WARNING: Input DPI below 150")
    if input_dpi is None:
        warnings.append("DPI_UNKNOWN: Raster DPI metadata unavailable")
    if score < quality_review_threshold:
        warnings.append(f"LOW_QUALITY: score {score:.1f} below {quality_review_threshold:.1f}")
    if min(w, h) < 600:
        warnings.append("LOW_PIXEL_DIMENSION")

    breakdown = (
        f"quality_score={score:.2f} = "
        + " + ".join(f"{weights[k]:.2f}*{k}({parts[k]:.1f})" for k in weights)
    )
    if estimated_dpi is not None:
        breakdown += f"; estimated_dpi_from_letter_assumption={estimated_dpi:.1f} (not used as input_dpi)"

    return QualityResult(
        quality_score=score,
        sharpness_score=round(sharpness, 2),
        contrast_score=round(contrast, 2),
        noise_score=round(noise, 2),
        blur_score=round(blur_clarity, 2),
        brightness_score=round(brightness, 2),
        clipping_score=round(clipping, 2),
        text_visibility_score=round(text_vis, 2),
        compression_score=round(compression, 2),
        usable_dimension_score=round(usable, 2),
        effective_resolution_score=round(effective, 2),
        width_px=int(w),
        height_px=int(h),
        input_dpi=input_dpi,
        estimated_dpi_from_pixels=estimated_dpi,
        warnings=warnings,
        breakdown=breakdown,
    )
