# quality_hw

Page quality + printed/handwritten classification only. No rotation, tilt, mirror or image saving.

For every page it reports:

- **Quality score** (0-10) with Good/Bad (Bad below `QUALITY_REVIEW_THRESHOLD`, default 4.0), the 10 submetrics and warnings (low DPI, blur, low contrast, ...).
- **Document type**: `PRINTED`, `HANDWRITTEN`, `BLANK` or `UNCERTAIN`, from the latest ConvNeXt-Tiny page model (`models/page_printed_handwritten_convnext_tiny.pth`) plus handwriting ink evidence that upgrades filled-in forms to HANDWRITTEN.
- **Final status**: `REVIEW_REQUIRED` if quality is Bad or the type is UNCERTAIN, `ERROR` if the page failed, otherwise `OK`.

Inputs: JPG, PNG, TIFF (multi-page), BMP, GIF, WEBP, PNM and PDF, from a local folder or Azure Blob.

## Setup

```bash
git lfs pull            # the model is stored in Git LFS
pip install -r quality_hw/requirements.txt
```

## Run

Edit the RUN CONFIG block at the top of `main.py`:

- `INPUT_SOURCE = "local"`: set `LOCAL_INPUT` to a file or folder.
- `INPUT_SOURCE = "blob"`: set `CONTAINER_NAME` and `PREFIX` (`PREFIX/<chart folder>/<file>`); sign in with `az login`. `START_FROM` resumes from a chart folder and appends to the existing report.
- `OUTPUT_DIR`: where `quality_hw_report.csv` is written. A row is added after every page, so it can be opened while the run is going.

```bash
python quality_hw/main.py
```
