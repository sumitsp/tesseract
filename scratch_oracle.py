"""Offline oracle: label each page's true upright orientation via OCR word confidence.

Used ONLY to tune/validate the pipeline's cheap signals. Never shipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

sys.path.insert(0, str(Path(__file__).parent))
from scratch_bench import IMAGES  # noqa: E402

OUT = Path("/tmp/oracle.json")


def ocr_strength(image):
    """(n_confident_words, mean_conf) for an image."""
    try:
        # psm 6 = uniform block, no layout-driven auto-rotation. psm 3 silently
        # OCRs one of the two vertical directions, which destroys axis discrimination.
        d = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6")
    except Exception:
        return 0, 0.0
    confs, texts = d.get("conf", []), d.get("text", [])
    good = []
    for c, t in zip(confs, texts):
        try:
            c = float(c)
        except (TypeError, ValueError):
            continue
        t = (t or "").strip()
        if c >= 60 and len(t) >= 2 and any(ch.isalnum() for ch in t):
            good.append(c)
    return len(good), (float(np.mean(good)) if good else 0.0)


def quad(image, deg):
    if deg == 0:
        return image
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def oracle_content_cw(image, max_dim=1600):
    """Which quadrant is the content rotated to (clockwise), by OCR evidence."""
    h, w = image.shape[:2]
    s = max_dim / float(max(h, w))
    small = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else image
    scores = {}
    for content_cw in (0, 90, 180, 270):
        # to view upright content we rotate the image by -content_cw (i.e. CCW)
        upright = quad(small, (360 - content_cw) % 360)
        n, m = ocr_strength(upright)
        scores[content_cw] = (n, m, n * m)
    best = max(scores, key=lambda k: scores[k][2])
    ranked = sorted(scores.values(), key=lambda v: -v[2])
    top, second = ranked[0][2], ranked[1][2]
    sep = (top - second) / (top + 1e-9) if top > 0 else 0.0
    return best, scores, sep


if __name__ == "__main__":
    names = [int(a) for a in sys.argv[1:]] or list(range(1, 211))
    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text())
    for i, name in enumerate(names):
        if str(name) in out:
            continue
        p = IMAGES / f"{name}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        best, scores, sep = oracle_content_cw(img)
        n, m, _ = scores[best]
        out[str(name)] = {
            "content_cw": best,
            "n_words": n,
            "mean_conf": round(m, 1),
            "separation": round(sep, 3),
            "scores": {str(k): [v[0], round(v[1], 1)] for k, v in scores.items()},
        }
        if i % 10 == 0:
            OUT.write_text(json.dumps(out, indent=1))
            print(f"{i}/{len(names)} {name}: cw={best} words={n} conf={m:.0f} sep={sep:.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    labelled = [v for v in out.values() if v["n_words"] >= 15 and v["separation"] >= 0.35]
    print(f"\ntotal={len(out)} trustworthy={len(labelled)}")
    from collections import Counter

    print("orientation distribution:", Counter(v["content_cw"] for v in labelled))
