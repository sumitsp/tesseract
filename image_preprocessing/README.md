# Document-page preprocessor for OCR

Reads PDF / JPG / JPEG / PNG / TIF / TIFF (and other common document images) from a local path, processes **each page sequentially**, and writes:

```text
output/
    corrected_pages/
    report/preprocessing_report.xlsx
    debug/                  # only with --debug
    preprocessing.log
```

Uncertain detections **do not force a correction**. A wrong rotation/flip is worse than leaving the page unchanged.

## Setup

```bash
cd image_preprocessing
python -m pip install -r requirements.txt
```

Tesseract must be installed on the system. Coarse rotation is Tesseract OSD only (0/90/180/270). If OSD cannot decide, the page is left unrotated — geometric rotation is not used as a fallback. The handwritten/printed classifier is the existing ConvNeXt-Tiny model under `document_type/` — it is not retrained here.

## Run

```bash
python main.py --input /path/to/file.pdf --output /path/to/output
python main.py --input /path/to/folder --output /path/to/output --debug
python main.py --help
```

## Pipeline (fixed order)

1. Quality / DPI analysis  
2. Standardize DPI (downsample to 400 only; **never upscale**)  
3. Printed vs handwritten (existing `hw_printed.py`)  
4. Coarse rotation from Tesseract OSD only (no geometric fallback)  
5. Mirror is measured and recorded, never flipped  
6. Fine tilt from the geometric detector, then applied  
7. Save PNG  
8. Excel row  

## Layout

```text
image_preprocessing/
├── main.py
├── config.py
├── pipeline.py
├── ingest/                 # loaders (not named io — that shadows the stdlib)
├── quality/
├── document_type/          # existing ConvNeXt classifier, unmodified
├── orientation/
├── validation/
├── reporting/
└── utils/
```

## Sign convention

`rotation_angle` is the **clockwise OSD correction applied** (0 / 90 / 180 / 270), the same number model-repo stores as `rotation_deg`. If OSD cannot decide, the page is left unrotated.

`tilt_angle` is the geometric residual, **positive = clockwise**. It is applied after rotation. Mirror is a YES/NO flag only; the page is never flipped.
