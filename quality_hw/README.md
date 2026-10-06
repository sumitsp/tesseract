# quality_hw

Page quality + printed/handwritten classification only. No rotation, tilt, mirror or image saving.

For every page it reports:

- **Quality score** (0-10) with Good/Medium/Bad (Bad below `QUALITY_REVIEW_THRESHOLD`, default 4.0), the 10 submetrics and warnings (low DPI, blur, low contrast, ...). A HANDWRITTEN page that would be Good is reported as Medium; its score is unchanged. BLANK pages get `N/A` (there is no content to judge).
- **Document type**: `PRINTED`, `HANDWRITTEN`, `BLANK` or `UNCERTAIN`.
  - Blank is decided before the model: stroke-sized marks are measured against the local paper brightness, so scanner noise, grey paper, borders and punch holes do not count. No marks → `BLANK` (`blank_page`). Only faint marks (pencil or show-through from the back) → `UNCERTAIN` (`faint_marks_only`) for review.
  - Pages with marks go to the latest ConvNeXt-Tiny page model (`models/page_printed_handwritten_convnext_tiny.pth`) plus handwriting ink evidence that upgrades filled-in forms to HANDWRITTEN.
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

## Retrain

The label CSV can mark a page Printed, Handwritten, Form, Visual, Blank, or Uncertain, plus Visible or Not visible, plus a handwritten-area percentage. A blank percentage stays blank. It is not stored as zero.

Empty scanner pages are still marked `BLANK` from the image before the model runs. The model learns Blank for the pages you labeled that way.

Edit `LABELS_CSV` and `IMAGES_DIR` at the top of `make_manifest.py`. `IMAGES_DIR` is the folder that contains the chart folders, so `IMAGES_DIR\<folder name>\<image name>` is the file.

```bash
python quality_hw/make_manifest.py
python quality_hw/train.py
```

`make_manifest.py` prints how many rows it wrote and how many image files were missing. Missing should be 0. `train.py` trains for 6 epochs and replaces `quality_hw/models/page_printed_handwritten_convnext_tiny.pth`. The first run downloads the ImageNet starting weights, so it needs a network. A GPU finishes in well under an hour. CPU takes a few hours.

Then run `python quality_hw/main.py` as usual. The report adds Visibility and Handwritten Area %. The current two-class model still loads until this retrain replaces it.
