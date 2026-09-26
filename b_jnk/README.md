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
b_jnk/
├── configs/default.json
├── data/
├── src/
├── scripts/
│   └── infer.py         # edit INPUT_PATH / OUTPUT_PATH at top
├── models/
└── reports/
```

## Run on another PC (inference only)

1. `git pull` then `cd b_jnk` and `pip install -r requirements.txt`
2. Open `scripts/infer.py` and set **`INPUT_PATH`** / **`OUTPUT_PATH`** at the top
3. Run:

```bash
python scripts/infer.py
```

Terminal prints each page as it is flagged. Output is **Excel** by default. Nothing is deleted — flags only.
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
