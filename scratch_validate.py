"""End-to-end validation of the real pipeline detectors on oracle-labelled pages.

For each page (known upright) x each synthetic rotation, run the production
rotation + skew stages with validation and measure the residual error of the
corrected page. The number that matters is "made worse": pages whose final
misalignment exceeds the misalignment they started with.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from image_preprocessing.config import default_config  # noqa: E402
from image_preprocessing.orientation.angle_search import wrap_pm180  # noqa: E402
from image_preprocessing.orientation.mirror_detector import detect_mirror  # noqa: E402
from image_preprocessing.orientation.arbitrary_rotation import detect_rotation  # noqa: E402
from image_preprocessing.orientation.skew_detector import detect_skew  # noqa: E402
from image_preprocessing.utils.image_utils import (  # noqa: E402
    rotate_bound,
    rotate_lossless_ccw,
)
from image_preprocessing.validation.correction_validator import validate_candidate  # noqa: E402

IMAGES = Path("/Users/sumit/Algodel/drive-download-20260915T202926Z-1-001/Images")
ORACLE = Path("/tmp/oracle.json")
ANGLES = [0, 90, 180, 270, 3.0, 17, 63, 120, 237, 300]


def correct(image, config, *, check_mirror=False):
    """Mirror of pipeline.process_page's geometry stages. Returns (applied_cw, info)."""
    rot = detect_rotation(image, config)
    applied = 0.0
    info = {
        "rot_status": rot.status,
        "rot_conf": rot.confidence,
        "residual": rot.residual_deg,
        "quadrant": rot.quadrant_deg,
        "mirror": "n/a",
    }
    work = image
    confirmed = False
    if rot.status == "NOT_NEEDED":
        confirmed = True
    elif rot.status == "DETECTED":
        cand = rotate_lossless_ccw(image, int(rot.quadrant_deg or 0))
        res = float(rot.residual_component_deg or 0.0)
        if abs(res) > 0.05:
            cand = rotate_bound(cand, res)
        v = validate_candidate(image, cand, config)
        if v.accepted:
            work, applied, confirmed = cand, float(rot.angle_cw_deg or 0.0), True
            info["rot_status"] = "APPLIED"
        else:
            info["rot_status"] = "REJECTED"

    if confirmed and check_mirror:
        m = detect_mirror(work, config)
        info["mirror"] = m.mirror

    skew = detect_skew(work, config)
    info["tilt_status"] = skew.status
    info["tilt_conf"] = skew.confidence
    if confirmed and skew.status == "DETECTED" and skew.tilt_cw_deg is not None:
        cand = rotate_bound(work, float(skew.tilt_cw_deg))
        if validate_candidate(work, cand, config).accepted:
            work = cand
            applied += float(skew.tilt_cw_deg)
            info["tilt_status"] = "APPLIED"
        else:
            info["tilt_status"] = "REJECTED"
    info["confirmed"] = confirmed
    return applied, info


def main():
    config = default_config()
    oracle = json.loads(ORACLE.read_text())
    upright = sorted(
        int(k) for k, v in oracle.items()
        if v["content_cw"] == 0 and v["n_words"] >= 15 and v["separation"] >= 0.35
    )
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = upright[:: max(1, len(upright) // limit)][:limit]
    check_mirror = "--mirror" in sys.argv

    t0 = time.time()
    rows = []
    for name in names:
        img = cv2.imread(str(IMAGES / f"{name}.jpg"))
        if img is None:
            continue
        for th in ANGLES:
            rot_img = rotate_bound(img, -th) if th else img
            applied, info = correct(rot_img, config, check_mirror=check_mirror)
            # truth: content sits at th CW; a perfect correction applies exactly th
            before = abs(wrap_pm180(th))
            after = abs(wrap_pm180(th - applied))
            rows.append({"name": name, "th": th, "applied": applied,
                         "before": before, "after": after, **info})

    n = len(rows)
    worse = [r for r in rows if r["after"] > r["before"] + 0.5]
    fixed = [r for r in rows if r["before"] > 1.0 and r["after"] <= 2.0]
    needed = [r for r in rows if r["before"] > 1.0]
    upright_rows = [r for r in rows if r["before"] <= 1.0]
    upright_broken = [r for r in upright_rows if r["after"] > 1.0]

    print(f"pages={len(names)} angles={len(ANGLES)} rows={n}  ({time.time()-t0:.0f}s, {(time.time()-t0)/n:.2f}s/page)")
    print()
    print(f"MADE WORSE                 : {len(worse):4d} / {n}   <-- the number that matters")
    print(f"already-upright pages kept : {len(upright_rows)-len(upright_broken):4d} / {len(upright_rows)}")
    print(f"rotated pages recovered    : {len(fixed):4d} / {len(needed)}")
    after = np.array([r["after"] for r in rows])
    print(f"final error: med={np.median(after):.2f} p90={np.percentile(after,90):.2f} max={after.max():.2f}")
    print()
    print("rotation status:", dict(Counter(r["rot_status"] for r in rows)))
    print("tilt status    :", dict(Counter(r["tilt_status"] for r in rows)))
    if check_mirror:
        print("mirror         :", dict(Counter(r["mirror"] for r in rows)))
    print()
    if worse:
        print("--- made worse ---")
        for r in worse[:20]:
            print(f"   img{r['name']:>4} th={r['th']:>5} applied={r['applied']:>7.2f} "
                  f"before={r['before']:.2f} after={r['after']:.2f} {r['rot_status']} "
                  f"resid={r['residual']} quad={r['quadrant']} conf={r['rot_conf']}")
    unrec = [r for r in needed if r["after"] > 2.0]
    if unrec:
        print(f"--- not recovered ({len(unrec)}) ---")
        for r in unrec[:20]:
            print(f"   img{r['name']:>4} th={r['th']:>5} applied={r['applied']:>7.2f} "
                  f"after={r['after']:.2f} {r['rot_status']} resid={r['residual']} "
                  f"quad={r['quadrant']} conf={r['rot_conf']}")


if __name__ == "__main__":
    main()
