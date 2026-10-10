# quality_hw_run

Run page quality and printed / handwritten / blank on this folder alone. Training files are not included.

The model is `models/page_printed_handwritten_convnext_tiny.pth`.

## Setup

```bash
pip install -r requirements.txt
```

Edit the RUN CONFIG block at the top of `main.py`:

- `INPUT_SOURCE`: `"local"` or `"blob"`.
- Local: `LOCAL_INPUT` is a file or a folder of chart folders.
- Blob: `CONTAINER_NAME` and `PREFIX`. The account is `azsadve2aipoc`. Sign in with `az login` first.
- `OUTPUT_DIR`: where `quality_hw_report.csv` is written.

## Run

From inside this folder:

```bash
python main.py
```
