"""
Synthetic evaluation for PageOrientationDetector (no OCR).

Usage:
  python eval_orientation.py --source path/to/clean_pages --out eval_out

Creates known transforms (rotation / tilt / mirror + noise) and reports accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from page_orientation import PageOrientationDetector
from page_orientation.correct import rotate_coarse_cw, rotate_fine, flip_horizontal


def load_images(source: Path, limit: int) -> list[np.ndarray]:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        paths.extend(source.rglob(ext))
    paths = sorted(paths)[:limit]
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    return images


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image
    if rng.random() < 0.5:
        # brightness / contrast
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.uniform(-25, 25)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.5:
        noise = rng.gauss(0, 8)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        gauss = np.random.normal(0, 6, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(35, 85)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def apply_gt(
    image: np.ndarray,
    rotation: int,
    tilt: float,
    mirror: bool,
) -> np.ndarray:
    """
    Build a transformed page with known GT.

    We apply inverse of the detector's correction order so that
    detector.correct should recover the original orientation roughly.
    Detector corrects: mirror → rotation → tilt.
    So synthesize with: tilt → inverse-rotation → mirror  (or equivalent).

    Simpler approach: start from upright, apply:
      1) clockwise tilt
      2) clockwise coarse rotation
      3) optional mirror
    Detector should report those values (approx).
    """
    out = rotate_fine(image, -tilt, expand=True)  # introduce clockwise tilt of +tilt
    # rotate_fine(img, t) undoes clockwise t; so to ADD clockwise tilt t, call with -t
    out = rotate_coarse_cw(out, rotation)
    if mirror:
        out = flip_horizontal(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Folder of clean upright pages")
    parser.add_argument("--out", type=Path, default=Path("eval_orientation_out"))
    parser.add_argument("--limit", type=int, default=20, help="Max source pages")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    images = load_images(args.source, args.limit)
    if not images:
        raise SystemExit(f"No images found under {args.source}")

    detector = PageOrientationDetector(analysis_max_dimension=1600, max_tilt_to_apply=None)

    rotations = [0, 90, 180, 270]
    tilts = [-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0]
    mirrors = [False, True]

    rows = []
    confusion = {r: {p: 0 for p in rotations} for r in rotations}
    mirror_tp = mirror_tn = mirror_fp = mirror_fn = 0
    tilt_errs: list[float] = []
    review_flags = 0
    n = 0

    for img in images:
        for rot in rotations:
            for tilt in tilts:
                for mir in mirrors:
                    if rng.random() < 0.55:
                        # subsample combinations for speed
                        continue
                    synth = apply_gt(img, rot, tilt, mir)
                    synth = degrade(synth, rng)
                    result = detector.detect_result(synth)
                    n += 1
                    review_flags += int(result.needs_review)
                    confusion[rot][result.rotation] += 1
                    err = abs(result.tilt - tilt)
                    # After coarse rot/mirror, tilt estimate is residual; allow wrap noise.
                    tilt_errs.append(err)
                    if mir and result.mirror:
                        mirror_tp += 1
                    elif (not mir) and (not result.mirror):
                        mirror_tn += 1
                    elif result.mirror and not mir:
                        mirror_fp += 1
                    else:
                        mirror_fn += 1
                    rows.append(
                        {
                            "gt_rotation": rot,
                            "pred_rotation": result.rotation,
                            "gt_tilt": tilt,
                            "pred_tilt": result.tilt,
                            "gt_mirror": mir,
                            "pred_mirror": result.mirror,
                            "overall_confidence": result.overall_confidence,
                            "needs_review": result.needs_review,
                        }
                    )

    rot_correct = sum(confusion[r][r] for r in rotations)
    rot_acc = rot_correct / max(n, 1)
    mirror_acc = (mirror_tp + mirror_tn) / max(n, 1)
    tilt_arr = np.array(tilt_errs, dtype=np.float64) if tilt_errs else np.array([0.0])
    report = {
        "n_samples": n,
        "rotation_accuracy": rot_acc,
        "mirror_accuracy": mirror_acc,
        "tilt_mae": float(np.mean(tilt_arr)),
        "tilt_median_ae": float(np.median(tilt_arr)),
        "tilt_p95_ae": float(np.percentile(tilt_arr, 95)),
        "review_rate": review_flags / max(n, 1),
        "rotation_confusion": confusion,
        "note": (
            "Synthetic evaluation. Do not claim 99% accuracy unless this report "
            "and real medical-page holdout both support it."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
