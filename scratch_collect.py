"""Collect raw rotation signals over oracle-labelled pages x synthetic angles.

Writes /tmp/signals.json so thresholds can be tuned offline without re-running.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

sys.path.insert(0, str(Path(__file__).parent))
from image_preprocessing.utils.image_utils import rotate_bound  # noqa: E402
from scratch_bench import IMAGES, analysis_gray, clean_ink, estimate  # noqa: E402

ORACLE = Path("/tmp/oracle.json")
OUT = Path("/tmp/signals.json")
ANGLES = [0, 90, 180, 270, 6.5, 33, 63, 120, 237, 300]


def osd_raw(image):
    try:
        d = pytesseract.image_to_osd(image, output_type=Output.DICT)
    except Exception:
        return None, 0.0
    try:
        rot = int(d.get("rotate", 0)) % 360
        conf = float(d.get("orientation_conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None, 0.0
    if rot not in (0, 90, 180, 270):
        return None, conf
    return (360 - rot) % 360, conf  # content_cw


def signals(image, max_dim=1400):
    ink = clean_ink(analysis_gray(image, max_dim), derule=False, deblob=True)
    r, margin = estimate(ink)
    if r is None:
        r, margin = 0.0, 0.0
    work = rotate_bound(image, r) if abs(r) > 0.05 else image
    q, conf = osd_raw(work)
    # stability probe: OSD on the same page turned 180 should answer q+180
    q2, conf2 = osd_raw(cv2.rotate(work, cv2.ROTATE_180))
    stable = q is not None and q2 is not None and (q2 - q) % 360 == 180
    return {
        "residual": None if r is None else round(float(r), 3),
        "margin": round(float(margin), 4),
        "osd_q": q,
        "osd_conf": round(conf, 2),
        "osd_q180": q2,
        "osd_conf180": round(conf2, 2),
        "stable": bool(stable),
    }


if __name__ == "__main__":
    oracle = json.loads(ORACLE.read_text())
    upright = sorted(
        int(k) for k, v in oracle.items()
        if v["content_cw"] == 0 and v["n_words"] >= 15 and v["separation"] >= 0.35
    )
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    names = upright[:: max(1, len(upright) // limit)][:limit]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for i, name in enumerate(names):
        img = cv2.imread(str(IMAGES / f"{name}.jpg"))
        if img is None:
            continue
        for th in ANGLES:
            key = f"{name}@{th}"
            if key in out:
                continue
            rot = rotate_bound(img, -th) if th else img
            s = signals(rot)
            s["truth_cw"] = th
            s["name"] = name
            out[key] = s
        if i % 5 == 0:
            OUT.write_text(json.dumps(out, indent=0))
            print(f"{i}/{len(names)} {name} elapsed={time.time()-t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(out, indent=0))
    print(f"collected {len(out)} rows in {time.time()-t0:.0f}s")
