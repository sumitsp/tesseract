# Page Orientation Detector (OpenCV only)

Production-oriented **document page orientation** module used before OCR.

**No OCR** (no Tesseract / RapidOCR / EasyOCR / Paddle / Docling OCR).  
**No ML models / no network APIs.** OpenCV + NumPy only.

## Install

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick usage

```python
import cv2
from page_orientation import PageOrientationDetector

image = cv2.imread("page.jpg")
detector = PageOrientationDetector(
    analysis_max_dimension=1800,
    max_tilt_to_apply=5.0,   # measure always; apply tilt only if |tilt| <= 5
)

result = detector.detect(image)
# {
#   "rotation": 0|90|180|270,
#   "tilt": float,
#   "mirror": bool,
#   "rotation_confidence": float,
#   "tilt_confidence": float,
#   "mirror_confidence": float,
#   "overall_confidence": float,
#   "needs_review": bool
# }

corrected = detector.correct(image, result)
# pass `corrected` to RapidOCR / Docling exactly once
```

Batch Azure runner: `rotation.py` (downloads blobs, writes CSV + corrected images).

## Sign / coordinate convention

| Field | Meaning |
|---|---|
| **rotation** | Clockwise coarse correction to apply (`0/90/180/270`) |
| **tilt** | Residual fine skew in degrees **after** mirror+rotation. **Positive = clockwise** (text leans down-to-the-right; image `y` downward) |
| **mirror** | `True` ⇒ apply horizontal flip |

Correction order in `correct()`:

1. Coarse rotation  
2. Mirror (if needed) — applied **after** rotation so LTR mirror cues are valid  
3. Fine tilt (OpenCV CCW by `+tilt` undoes clockwise lean), unless `|tilt| > max_tilt_to_apply`

(Rotation-then-mirror is required for 90°/270° consistency; flip-then-rotate is not equivalent.)

## Algorithm (ensemble)

1. **Preprocess** (analysis resolution): illumination normalize, CLAHE, adaptive+Otsu ink masks, border strip, tiny-CC cleanup.  
2. **Rotation**: vote across projection profiles, morphological H/V response, text-like CC + line grouping, weak 0°/180° layout biases. Confidence from best-vs-second separation + evidence count. Near-ties → `needs_review`, prefer `0` over weak `180`.  
3. **Mirror**: compare structural scores on ink vs `cv2.flip(..., 1)` (margins, L/R density, line left-edge clustering, spacing regularity). Weak separation → `mirror=False`, lower confidence, review.  
4. **Tilt** (on coarsely corrected analysis image): Hough near-horizontal segments (border-aware), CC text-line fits, 3×3 regional consensus (MAD inliers), projection-profile local search around the seed. Aggregate with robust median.

Tables/borders are mitigated via border masking, length/position filters, and preferring text-component masks for projection scoring.

## Confidence

- Per-head confidence ∈ [0, 1] from score separation / regional agreement / evidence.  
- `overall_confidence = 0.40*rot + 0.30*tilt + 0.30*mirror`  
- `needs_review` when ambiguous 0/180, weak mirror, disagreeing tilt regions, sparse ink, or low overall confidence.

Use `detector.detect_result(image)` for diagnostics (`result.diagnostics`).

## Debug

```python
detector = PageOrientationDetector(debug=True, debug_dir="debug_out")
detector.detect(image)
```

Writes `ink_mask.png`, `gray_clahe.png`, `result.json`.

## Evaluation

```powershell
.\venv\Scripts\python.exe eval_orientation.py --source path\to\clean_upright_pages --out eval_out --limit 20
```

Synthesizes known rotation/tilt/mirror (+ JPEG/noise/blur/brightness), reports rotation accuracy, mirror accuracy, tilt MAE / median / p95, review rate, confusion matrix.

**Do not claim ≥99% accuracy** unless this synthetic report **and** a real medical-page holdout both support it. 0° vs 180° and mirror remain fundamentally hard without OCR on symmetric forms.

## Limitations

- Symmetric / stamp-heavy / image-dominated / near-blank pages → low confidence + review.  
- 0° vs 180° uses weak layout priors, not semantics.  
- Mirror assumes mild LTR / left-alignment asymmetry; centered tables may be ambiguous.  
- Extreme perspective / severe warp is out of scope (planar skew only).

## Package layout

```
page_orientation/
  __init__.py
  detector.py          # PageOrientationDetector
  preprocess.py
  components.py        # text-like CCs + line grouping
  rotation_detect.py
  mirror_detect.py
  tilt_detect.py
  correct.py
  result.py
```
