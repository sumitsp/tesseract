"""End-to-end: residual -> axis-align -> OSD quadrant -> total rotation recovery."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

sys.path.insert(0, str(Path(__file__).parent))
from image_preprocessing.utils.image_utils import rotate_bound  # noqa: E402
from scratch_bench import IMAGES, analysis_gray, clean_ink, estimate, wrap90  # noqa: E402


def osd_quadrant(image):
    """content_cw quadrant in {0,90,180,270}, or None."""
    try:
        d = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except Exception:
        return None, 0.0
    try:
        rot = int(d.get("rotate", 0)) % 360
        conf = float(d.get("orientation_conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None, 0.0
    if rot not in (0, 90, 180, 270) or conf < 1.0:
        return None, conf
    return (360 - rot) % 360, conf


def recover(image, max_dim=1400):
    """Return (content_cw_estimate, residual, margin, osd_conf)."""
    ink = clean_ink(analysis_gray(image, max_dim), derule=False, deblob=True)
    r, margin = estimate(ink)
    if r is None:
        r, margin = 0.0, 0.0
    # axis-align a working copy (analysis resolution is enough for OSD)
    work = rotate_bound(image, r) if abs(r) > 0.05 else image
    q, conf = osd_quadrant(work)
    if q is None:
        return None, r, margin, conf
    return (r + q) % 360, r, margin, conf


if __name__ == "__main__":
    names = [int(a) for a in sys.argv[1:]] or [1, 12, 25, 33, 47, 50, 68, 75, 90, 100, 113, 125, 137, 150, 166, 178, 190, 200, 205, 209]
    angles = [0, 90, 180, 270, 6.5, 17, 63, 120, 237, 300]
    t0 = time.time()
    errs, undecided, rows = [], 0, []
    for name in names:
        img = cv2.imread(str(IMAGES / f"{name}.jpg"))
        if img is None:
            continue
        base_cw, _, _, _ = recover(img)
        base_native = 0.0 if base_cw is None else wrap90(base_cw)
        for th in angles:
            rot = rotate_bound(img, -th) if th else img
            got, r, margin, conf = recover(rot)
            expect = (th + base_native) % 360
            if got is None:
                undecided += 1
                rows.append((name, th, None, None, round(margin, 2), round(conf, 1)))
                continue
            err = abs(((got - expect + 180) % 360) - 180)
            errs.append(err)
            if err > 2.0:
                rows.append((name, th, round(expect, 1), round(got, 1), round(err, 1), round(conf, 1)))
    errs = np.array(errs)
    print(f"n={len(errs)} undecided={undecided} med={np.median(errs):.2f} p90={np.percentile(errs,90):.2f} max={errs.max():.2f}")
    print(f"within 2deg: {int((errs<=2).sum())}/{len(errs)}   within 5deg: {int((errs<=5).sum())}/{len(errs)}")
    print(f"quadrant-wrong (>45deg): {int((errs>45).sum())}")
    print(f"elapsed {time.time()-t0:.0f}s  ({(time.time()-t0)/max(1,len(errs)+undecided):.2f}s/page)")
    print("\n--- failures / undecided ---")
    for r in rows:
        print("   ", r)
