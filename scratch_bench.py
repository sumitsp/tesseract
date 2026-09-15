"""Scratch benchmark for residual-angle estimation variants. Not part of the package."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from image_preprocessing.utils.image_utils import rotate_bound, to_gray  # noqa: E402

IMAGES = Path("/Users/sumit/Algodel/drive-download-20260915T202926Z-1-001/Images")


# ---------------------------------------------------------------- ink pipeline
def analysis_gray(image, max_dim):
    gray = to_gray(image)
    h, w = gray.shape
    s = max_dim / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return gray


def normalize_illumination(gray):
    h, w = gray.shape
    k = max(31, (min(h, w) // 20) | 1)
    bg = np.maximum(cv2.GaussianBlur(gray, (k, k), 0), 1)
    return np.clip(gray.astype(np.float32) / bg.astype(np.float32) * 128.0, 0, 255).astype(np.uint8)


def binarize(gray):
    # No CLAHE: on an illumination-normalized page it amplifies background noise
    # until text lines binarize as solid bars (~23% coverage) instead of strokes,
    # which is what made the angle search bistable on dense-text pages.
    g = normalize_illumination(gray)
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    ink = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


def strip_padding(ink, gray):
    """Zero ink outside the real page area (handles rotate_bound white triangles)."""
    return ink


def remove_rules(ink, frac=0.10):
    h, w = ink.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * frac)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * frac))))
    lines = cv2.bitwise_or(cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk), cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk))
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.bitwise_and(ink, cv2.bitwise_not(lines))


def remove_blobs(ink, min_area=8, max_area_frac=0.02, max_dim_frac=0.35):
    h, w = ink.shape
    max_area = int(h * w * max_area_frac)
    md = int(max(h, w) * max_dim_frac)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area or a > max_area:
            continue
        if cw > md or ch > md:
            continue
        keep[i] = True
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


def clean_ink(gray, *, derule=True, deblob=True):
    ink = binarize(gray)
    if derule:
        ink = remove_rules(ink)
    if deblob:
        ink = remove_blobs(ink)
    return ink


# ------------------------------------------------------------------- scoring
def points(ink, max_pts=80000, seed=0):
    ys, xs = np.nonzero(ink)
    if len(xs) > max_pts:
        i = np.random.default_rng(seed).choice(len(xs), max_pts, replace=False)
        xs, ys = xs[i], ys[i]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    if len(xs):
        xs -= xs.mean()
        ys -= ys.mean()
    return xs, ys


def sharpness(proj, bin_size=1.0):
    lo, hi = proj.min(), proj.max()
    nb = max(8, int((hi - lo) / bin_size) + 1)
    h, _ = np.histogram(proj, bins=nb, range=(lo, hi))
    h = h.astype(np.float64)
    n = h.sum()
    if n <= 0:
        return 0.0
    p = h / n
    return float(np.dot(p, p))


def axis_score(xs, ys, a_deg, bin_size=1.0):
    ar = math.radians(a_deg)
    ca, sa = math.cos(ar), math.sin(ar)
    return max(sharpness(-xs * sa + ys * ca, bin_size), sharpness(xs * ca + ys * sa, bin_size))


def estimate(ink, coarse_step=0.5, min_pts=300):
    xs, ys = points(ink)
    if len(xs) < min_pts:
        return None, 0.0
    coarse = np.arange(-45.0, 45.0, coarse_step)
    sc = np.array([axis_score(xs, ys, a) for a in coarse])
    b = float(coarse[int(sc.argmax())])
    for step, rad in ((0.1, coarse_step), (0.02, 0.1)):
        rng = np.arange(b - rad, b + rad + 1e-9, step)
        s2 = np.array([axis_score(xs, ys, a) for a in rng])
        b = float(rng[int(s2.argmax())])
    peak, med = float(sc.max()), float(np.median(sc))
    return b, (peak - med) / (peak + 1e-9)


# ---------------------------------------------------------------- benchmark
def wrap90(a):
    return ((a + 45.0) % 90.0) - 45.0


def run(variant, names, angles, max_dim=1400):
    errs, bad, nopts = [], [], 0
    for name in names:
        img = cv2.imread(str(IMAGES / f"{name}.jpg"))
        if img is None:
            continue
        # native skew of the page, measured once, becomes the ground-truth offset
        base_ink = clean_ink(analysis_gray(img, max_dim), **variant)
        base, _ = estimate(base_ink)
        if base is None:
            continue
        for th in angles:
            rot = rotate_bound(img, -th) if th else img
            ink = clean_ink(analysis_gray(rot, max_dim), **variant)
            r, m = estimate(ink)
            if r is None:
                nopts += 1
                continue
            err = abs(wrap90(r - wrap90(base + th)))
            errs.append(err)
            if err > 1.0:
                bad.append((name, th, round(wrap90(base + th), 2), round(r, 2), round(err, 2), round(m, 2)))
    return np.array(errs), bad, nopts


if __name__ == "__main__":
    names = [1, 12, 25, 33, 47, 50, 68, 75, 90, 100, 113, 125, 137, 150, 166, 178, 190, 200, 205, 209]
    angles = [0, 2.0, 6.5, 17, 33, 44, 63, 90, 120, 175, 237, 300]
    for label, variant in (
        ("derule+deblob", dict(derule=True, deblob=True)),
        ("derule only", dict(derule=True, deblob=False)),
        ("deblob only", dict(derule=False, deblob=True)),
        ("raw ink", dict(derule=False, deblob=False)),
    ):
        t0 = time.time()
        errs, bad, nopts = run(variant, names, angles)
        if not len(errs):
            print(label, "no results")
            continue
        print(
            f"{label:15s} n={len(errs):4d} med={np.median(errs):.2f} p90={np.percentile(errs,90):6.2f} "
            f"max={errs.max():6.2f} <=1deg={int((errs<=1).sum()):3d}/{len(errs)} "
            f"<=2deg={int((errs<=2).sum()):3d} nopts={nopts} {time.time()-t0:.0f}s"
        )
        for b in bad[:12]:
            print("     ", b)
