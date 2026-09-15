# AdvantMed Imaging

## Setup

```powershell
cd advantmed_imaging
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
az login
```

## Scripts

| Script | Output env var |
|---|---|
| `os_ocr.py` | `DOCLING_FORMAT_AND_RAPID_OUTPUT_DIR` |
| `rapid_ocr.py` | `RAPID_OUTPUT_DIR` |
| `ts_ocr.py` | `TESSERACT_OUTPUT_DIR` |
| `rotation.py` | `ROTATION_CORRECTED_OUTPUT_DIR` |
| `classify_hw_printed.py` | `CSV_PATH_HW_PRINTED` |
