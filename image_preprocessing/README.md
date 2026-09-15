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

Tesseract must be installed on the system (OSD is used only to resolve the 180° ambiguity after classical geometry finds the text-line angle). Without the `tesseract` binary, the pipeline still runs, but rotation is marked `UNCERTAIN` rather than guessing 0° vs 180°. The handwritten/printed classifier is the existing ConvNeXt-Tiny model under `document_type/` — it is not retrained here.

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
4. Arbitrary rotation (classical CV) + Tesseract OSD for 180° only  
5. Mirror detection (after rotation)  
6. Fine tilt / skew ±10° (after rotation + mirror)  
7. Validate; reject worse geometry  
8. Save PNG  
9. Excel row  

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

`rotation_angle` and `tilt_angle` are the **clockwise offset of content from upright**, in degrees. Correction rotates the image **counter-clockwise** by that amount (OpenCV positive angle). Example: content at 120° clockwise → `rotation_angle = 120.4`.
