    """
    Detect and correct skew (tilt), 90/180/270 rotation, and horizontal mirroring
    in scanned JPGs using OpenCV + Tesseract OSD, then write:
    - one CSV row per image with the measured/applied correction values
    - a corrected copy of every image under OUTPUT_DIR, same folder structure as INPUT_DIR

    Pipeline per image (each step is applied to the full-resolution image; detection
    itself runs on a downscaled copy for speed):
    1. Tilt   - Hough line transform on text-line edges -> small-angle deskew.
    2. Rotate - Tesseract OSD on the deskewed image -> 0/90/180/270 correction.
    3. Mirror - OCR word-count/confidence compared normal vs. horizontally flipped
                on the now-upright image; whichever scores better is kept.

    CSV "*_deg" columns are the degrees rotated CLOCKWISE that were actually applied
    to correct the image (not the raw measured tilt of the original).
    """
    from __future__ import annotations

    import csv
    import math
    import sys
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import pytesseract
    from pytesseract import Output

    INPUT_DIR = Path(r"\\ADMPDNLP06\AI_Vendor\Run1\Batch2\DEID_PNGs")
    OUTPUT_DIR = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\corrected_images")
    CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\rotation_report.csv")
    # Set to a path like r"C:\Program Files\Tesseract-OCR\tesseract.exe" if tesseract
    # is not on PATH. Leave as None to use whatever `tesseract` PATH resolves to.
    TESSERACT_CMD: str | None = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # OCR/rotation start at this folder, then continues with later folders.
    # Use only the folder name, not the full path.
    START_FROM = ""
    JPG_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    DETECT_MAX_DIM = 1600       # downscale target (longest side, px) used only for detection
    SKEW_MIN_DEG = 0.1          # ignore measured tilt smaller than this (noise)
    SKEW_MAX_DEG = 30.0         # ignore measured tilt larger than this (Hough noise, not real skew)
    MIN_MIRROR_WORDS = 3        # need at least this many OCR'd words on one side to trust the mirror check

    CSV_FIELDS = [
        "folder", "filename",
        "skew_angle_deg", "rotation_deg", "rotation_confidence",
        "mirrored", "mirror_words_normal", "mirror_conf_normal",
        "mirror_words_flipped", "mirror_conf_flipped",
        "status", "error", "output_path",
    ]


    def log(message: str) -> None:
        print(message, flush=True)


    def require_tesseract() -> None:
        if TESSERACT_CMD:
            cmd_path = Path(TESSERACT_CMD)
            if not cmd_path.is_file():
                raise SystemExit(f"TESSERACT_CMD does not point to a file:\n  {TESSERACT_CMD}")
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        # The very first call to tesseract.exe on a machine can spuriously fail
        # (e.g. Windows Defender/SmartScreen scanning a freshly-installed exe on
        # first launch), even though the install is fine. Retry a few times
        # before giving up.
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                version = pytesseract.get_tesseract_version()
                log(f"Tesseract version: {version}")
                return
            except Exception as exc:
                last_exc = exc
                log(f"  Tesseract check attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
                time.sleep(2)

        raise SystemExit(
            "Tesseract is not reachable after 3 attempts. Install it and/or set TESSERACT_CMD in rotation.py.\n"
            f"  {type(last_exc).__name__}: {last_exc}"
        )


    def jpg_sort_key(path: Path) -> tuple:
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


    def list_chart_folders(input_dir: Path) -> list[Path]:
        try:
            folders = sorted(
                (entry for entry in input_dir.iterdir() if entry.is_dir()),
                key=lambda path: path.name,
            )
        except Exception as exc:
            raise SystemExit(
                "Cannot read the network path. Check the share, VPN, and folder name.\n"
                f"  {input_dir}\n"
                f"  {type(exc).__name__}: {exc}"
            )
        if not START_FROM:
            return folders
        start_index = next(
            (i for i, folder in enumerate(folders) if folder.name == START_FROM),
            None,
        )
        if start_index is None:
            raise SystemExit(f"START_FROM folder not found under input: {START_FROM}")
        log(f"Starting at {START_FROM}; skipping {start_index} earlier folder(s)")
        return folders[start_index:]


    def list_jpgs(folder: Path) -> list[Path]:
        images = [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix in JPG_SUFFIXES
        ]
        images.sort(key=jpg_sort_key)
        return images

    
    def downscale(image: np.ndarray, max_dim: int = DETECT_MAX_DIM) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, max_dim / float(max(h, w)))
        if scale >= 1.0:
            return image
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


    def rotate_bound_cw(image: np.ndarray, angle_cw_deg: float) -> np.ndarray:
        """Rotate `image` by `angle_cw_deg` degrees clockwise, expanding the canvas
        (white fill) so nothing is cropped."""
        if abs(angle_cw_deg) < 1e-6:
            return image
        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D((cx, cy), -angle_cw_deg, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2.0) - cx
        matrix[1, 2] += (new_h / 2.0) - cy
        return cv2.warpAffine(
            image, matrix, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )


    def measure_skew_deg(gray_small: np.ndarray) -> float:
        """Median tilt of near-horizontal text-line edges, as degrees CLOCKWISE
        needed to correct it (0.0 if nothing usable is found)."""
        edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
        min_len = max(30, gray_small.shape[1] // 4)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=150, minLineLength=min_len, maxLineGap=20
        )
        if lines is None:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) <= 30:
                angles.append(angle)
            elif abs(angle) >= 150:
                angles.append(angle - 180 if angle > 0 else angle + 180)
        if not angles:
            return 0.0
        measured_tilt = float(np.median(angles))  # image-space, clockwise-positive tilt of the text
        correction = -measured_tilt               # rotate clockwise by this amount to undo it
        if abs(correction) < SKEW_MIN_DEG or abs(correction) > SKEW_MAX_DEG:
            return 0.0
        return correction


    def measure_rotation(gray: np.ndarray) -> tuple[int, float]:
        """Returns (degrees clockwise to apply, orientation confidence) from Tesseract OSD."""
        try:
            osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
        except pytesseract.TesseractError:
            return 0, 0.0
        return int(osd.get("rotate", 0)) % 360, float(osd.get("orientation_conf", 0.0))


    def ocr_quality(gray: np.ndarray) -> tuple[int, float]:
        """Returns (word_count, mean_confidence) from a plain OCR pass."""
        data = pytesseract.image_to_data(gray, output_type=Output.DICT)
        word_count = 0
        conf_sum = 0.0
        for text, conf in zip(data["text"], data["conf"]):
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_val >= 0 and text.strip():
                word_count += 1
                conf_sum += conf_val
        mean_conf = conf_sum / word_count if word_count else 0.0
        return word_count, mean_conf


    def correct_image(image: np.ndarray) -> dict:
        """Runs the full tilt -> rotation -> mirror pipeline on a full-resolution
        BGR image. Returns a dict of measured values plus the corrected image."""
        gray_small = cv2.cvtColor(downscale(image), cv2.COLOR_BGR2GRAY)
        skew_deg = measure_skew_deg(gray_small)
        deskewed = rotate_bound_cw(image, skew_deg)

        rotation_deg, rotation_conf = measure_rotation(cv2.cvtColor(downscale(deskewed), cv2.COLOR_BGR2GRAY))
        if rotation_deg == 90:
            upright = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_deg == 180:
            upright = cv2.rotate(deskewed, cv2.ROTATE_180)
        elif rotation_deg == 270:
            upright = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            upright = deskewed

        upright_small = cv2.cvtColor(downscale(upright), cv2.COLOR_BGR2GRAY)
        words_normal, conf_normal = ocr_quality(upright_small)
        flipped_small = cv2.flip(upright_small, 1)
        words_flipped, conf_flipped = ocr_quality(flipped_small)

        mirrored = False
        if max(words_normal, words_flipped) >= MIN_MIRROR_WORDS:
            mirrored = (words_flipped, conf_flipped) > (words_normal, conf_normal)
        final = cv2.flip(upright, 1) if mirrored else upright

        return {
            "skew_angle_deg": round(skew_deg, 3),
            "rotation_deg": rotation_deg,
            "rotation_confidence": round(rotation_conf, 3),
            "mirrored": mirrored,
            "mirror_words_normal": words_normal,
            "mirror_conf_normal": round(conf_normal, 2),
            "mirror_words_flipped": words_flipped,
            "mirror_conf_flipped": round(conf_flipped, 2),
            "image": final,
        }


    def process_folder(
        writer: csv.DictWriter,
        csv_file,
        input_dir: Path,
        output_dir: Path,
        folder: Path,
        images: list[Path],
    ) -> None:
        out_folder = output_dir / folder.relative_to(input_dir)
        out_folder.mkdir(parents=True, exist_ok=True)
        log(f"  {len(images)} images -> {out_folder}")

        for image_path in images:
            log(f"  {image_path.name}")
            row = {"folder": folder.name, "filename": image_path.name, "status": "ok", "error": ""}
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (unreadable/corrupt image)")
                result = correct_image(image)
                out_path = out_folder / image_path.name
                # Always write a copy — including images that needed no correction
                # (skew=0, rotation=0, mirrored=False).
                changed = (
                    abs(result["skew_angle_deg"]) > 0
                    or result["rotation_deg"] != 0
                    or result["mirrored"]
                )
                if not cv2.imwrite(str(out_path), result.pop("image")):
                    raise ValueError(f"cv2.imwrite failed: {out_path}")
                row.update(result)
                row["output_path"] = str(out_path)
                log(
                    f"    wrote {'corrected' if changed else 'unchanged (copied)'} "
                    f"skew={row['skew_angle_deg']} rot={row['rotation_deg']} "
                    f"mirror={row['mirrored']} -> {out_path.name}"
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["output_path"] = ""
                log(f"    ERROR: {row['error']}")
            writer.writerow(row)
            csv_file.flush()


    def main() -> None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        log("=== TILT / ROTATION / MIRROR CORRECTION (OpenCV + Tesseract OSD) ===")
        input_dir = INPUT_DIR
        output_dir = OUTPUT_DIR
        log(f"INPUT:  {input_dir}")
        log(f"OUTPUT: {output_dir}")
        log(f"CSV:    {CSV_PATH}")
        log(f"START_FROM: {START_FROM or '(first folder)'}")
        if not input_dir.is_dir():
            raise SystemExit(
                "Input folder does not exist or the network share is not reachable:\n"
                f"  {input_dir}"
            )
        chart_folders = list_chart_folders(input_dir)
        if not chart_folders:
            raise SystemExit(f"No chart folders found under: {input_dir}")
        require_tesseract()
        output_dir.mkdir(parents=True, exist_ok=True)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        write_header = not CSV_PATH.is_file()
        with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
                csv_file.flush()
            for folder in chart_folders:
                images = list_jpgs(folder)
                if not images:
                    log(f"Folder: {folder.name} (no JPGs, skipped)")
                    continue
                log(f"Folder: {folder}")
                process_folder(writer, csv_file, input_dir, output_dir, folder, images)

        log(f"Done. Report: {CSV_PATH}")


    if __name__ == "__main__":
        main()
