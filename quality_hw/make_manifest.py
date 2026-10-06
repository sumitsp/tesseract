"""Turn the labeling CSV plus the downloaded image folders into manifest.csv.

Every labeled page type is kept: Printed, Handwritten, Form, Visual, Blank,
Uncertain. Visibility and the handwritten-area percentage are kept when the
labeler filled them in, and left blank when they were not.

Edit RUN CONFIG, then run:  python quality_hw/make_manifest.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ============================== RUN CONFIG ==============================
# The CSV from training_data_creator.
LABELS_CSV = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\training_labels.csv")

# The folder that contains the chart folders, so this file exists:
#   IMAGES_DIR / <folder name> / <image name>
IMAGES_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\images")

# Written next to the images. train.py reads this file.
MANIFEST_PATH = IMAGES_DIR / "manifest.csv"
# ========================================================================

LABELS = {"Printed", "Handwritten", "Form", "Visual", "Blank", "Uncertain"}
VISIBILITY = {"Visible", "Not visible"}
COLUMNS = ["path", "label", "visibility", "handwritten_percent", "source", "origin"]


def build_manifest(labels_csv: Path = LABELS_CSV, images_dir: Path = IMAGES_DIR, manifest_path: Path = MANIFEST_PATH) -> tuple[int, int, int]:
    """Returns (written, missing files, skipped rows)."""
    labels_csv = Path(labels_csv)
    images_dir = Path(images_dir)
    manifest_path = Path(manifest_path)
    if not labels_csv.is_file():
        raise FileNotFoundError(f"LABELS_CSV does not exist: {labels_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"IMAGES_DIR does not exist: {images_dir}")

    rows: list[dict[str, str]] = []
    missing = 0
    skipped = 0
    with labels_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            folder = (row.get("folder name") or "").strip()
            name = (row.get("image name") or "").strip()
            value = (row.get("value") or "").strip()
            visibility = (row.get("visibility") or "").strip()
            percent = (row.get("handwritten percent") or "").strip()
            if not folder or not name or value not in LABELS:
                skipped += 1
                continue
            if visibility not in VISIBILITY:
                visibility = ""
            if percent:
                try:
                    number = float(percent)
                except ValueError:
                    number = None
                percent = "" if number is None or number < 0 or number > 100 else f"{number:.4f}".rstrip("0").rstrip(".")
            rel = Path(folder) / name
            if not (images_dir / rel).is_file():
                missing += 1
                continue
            rows.append({
                "path": rel.as_posix(),
                "label": value,
                "visibility": visibility,
                "handwritten_percent": percent,
                "source": "labeled",
                "origin": name,
            })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), missing, skipped


def main() -> int:
    written, missing, skipped = build_manifest()
    print(f"wrote {written} rows -> {MANIFEST_PATH}")
    print(f"missing files {missing}   skipped rows {skipped}")
    if missing:
        print("IMAGES_DIR must be the parent of the chart folders.", file=sys.stderr)
        return 1
    if written < 10:
        print("Fewer than 10 images. Check LABELS_CSV and IMAGES_DIR.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
