# EHR Page Flagging (OCR text only)

Conservative page flagger for EHR packets. **Primary input is OCR text** — not the page image.

Output is a **CSV of flags only**. Nothing is deleted or removed from the packet; pages are tagged `KEEP`, `BLANK`, or `JUNK` for downstream review/routing.

## Status (v0.3 data)

`data/raw/ehr_pages_v0.3.jsonl` merges:

- **v0.2** blank / junk / review pages
- **New KEEP** clinical OCR (human-labeled) from real encounter notes

Output is still **flagging CSV only** (`KEEP` / `BLANK` / `JUNK`) — nothing is deleted.

Rebuild dataset:

```bash
python scripts/build_v03_dataset.py
```

## Status (v0.2 data — historical)

The original label set alone was **not sufficient** (almost no KEEP). Kept under `ehr_pages_v0.2.jsonl` for reference.

## Layout

```text
blank_junk_classifier/
├── configs/default.json
├── data/raw/ehr_pages_v0.2.jsonl
├── src/
│   ├── preprocessing/   # load, empty-OCR routing, taxonomy → flags
│   ├── features/        # OCR stats + TF-IDF
│   ├── models/          # TF-IDF / embeddings, split, decision layer
│   ├── evaluation/      # false-flag + missed-drop metrics
│   ├── explainability/
│   └── inference/       # page API
├── scripts/
│   ├── dataset_report.py
│   ├── train_baseline.py
│   ├── train_embeddings.py
│   ├── compare_models.py
│   └── infer.py         # → CSV
├── models/
└── reports/
```

## Run on another PC (inference only)

```bash
git pull
cd blank_junk_classifier
pip install -r requirements.txt

# Local OCR input: JSONL/JSON/CSV/XLSX with page_id + ocr_text, OR a folder of .txt
python scripts/infer.py \
  --input /path/to/pages.jsonl \
  --output reports/page_flags.xlsx
```

Terminal prints each page as it is flagged. Default output is **Excel** (`.xlsx`). Nothing is deleted — flags only.

Required files in this folder: `src/`, `scripts/infer.py`, `configs/default.json`, `models/tfidf_flat.joblib`, `requirements.txt`.

## Commands

```bash
python scripts/dataset_report.py --jsonl data/raw/ehr_pages_v0.3.jsonl --out reports/dataset_quality_v0.3.json

# 2) Train TF-IDF on KEEP / BLANK / JUNK
python scripts/train_baseline.py

# 3) Flag pages → CSV
python scripts/infer.py \
  --input data/raw/ehr_pages_v0.3.jsonl \
  --output reports/page_flags.csv
```
## CSV columns

| column | meaning |
|--------|---------|
| `page_id` | page id |
| `flag` | `KEEP` \| `BLANK` \| `JUNK` |
| `confidence` | model confidence for the chosen flag |
| `review_required` | `1` if uncertain / vetoed to KEEP |
| `p_keep` / `p_blank` / `p_junk` | bucket probabilities |
| `decision_reason` | why the flag was chosen |
| `top_evidence` | short audit string |
| `model_version` | config model version |

## Safety behaviour

1. Empty / near-empty OCR → `KEEP` + `review_required` (never auto-`BLANK`)
2. `BLANK` / `JUNK` only when confidence is high **and** KEEP probability is low
3. Low confidence → `KEEP` + review (safe default)
4. Split by `split_group` / template to avoid vendor-packet leakage
5. Primary metric: **false flag rate** (KEEP pages tagged BLANK/JUNK)
