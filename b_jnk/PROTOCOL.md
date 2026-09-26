# Blank & Junk elimination protocol (encoded in model stack)

Coarse flags stay **KEEP / BLANK / JUNK** (flagging only — nothing deleted here).
Excel also writes **`audit_tag`** for QC chain of custody.

## Mandatory retention (hard rules)

These **never** become BLANK/JUNK, even if the TF-IDF model scores junk:

| Signal | Audit tag | Rule |
|--------|-----------|------|
| Clinical wound / derm / endoscopy / colonoscopy / retinal / OCT / slit-lamp cues | `KEEP_CLINICAL_IMAGE` | `protocol_retain_clinical_image` |
| Demographics / registration / face sheet / MRN / DOB / guarantor cues | `KEEP_DEMOGRAPHIC` | `protocol_retain_demographic` |

Empty OCR still → `KEEP` + `review_required` (could be absolute blank **or** failed OCR on a clinical image).

## Blank typology → audit tags

- `BLANK_ABSOLUTE` / `BLANK_ABSOLUTE_CANDIDATE` — no text
- `BLANK_TECHNICAL` — scan noise language
- `BLANK_SYSTEM` — “intentionally left blank”
- `BLANK_HEADER_FOOTER_ONLY` — header/footer patterns only

## Junk typology → audit tags

- `JUNK_COVER_REQUEST`, `JUNK_FAX_TRANSMISSION`, `JUNK_SEPARATOR_BARCODE`
- `JUNK_PRINTER_SYSTEM_TEST`, `JUNK_POSTAL_MAIL`, `JUNK_INSURANCE_ID`
- `JUNK_BLACK_SCAN_DEFECT`, `JUNK_NON_CLINICAL_PHOTO`

Implementation: `src/preprocessing/protocol.py` + decision layer in `src/models/decision.py`.
Typology keywords are also TF-IDF hand features in `src/features/ocr_features.py`.
