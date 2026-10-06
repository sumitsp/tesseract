# blank_junk_creator

Label chart images from Azure Blob as Keep, Blank, or a junk subtype. The page runs on this machine. Images are read from blob. The CSV is written locally.

Each row is one image: `value` (`KEEP`, `BLANK`, or `JUNK`) and `subtype` (empty for Keep and Blank; for Junk one of Invoice, Cover page, Record request, Instructions, Letter / fax, Others). Labeling it again updates that row. Empty fields are allowed.

## Setup

```bash
pip install -r blank_junk_creator/requirements.txt
az login
```

Edit the RUN CONFIG block at the top of `main.py`:

- `CONTAINER_NAME` and `PREFIX` (`PREFIX/<chart folder>/<image>`). The account is `azsadve2aipoc`.
- `START_FROM`: leave empty to begin at the first folder. Set a folder name to begin at that folder and continue through the later folders.
- `CSV_PATH`: where `blank_junk_labels.csv` is written. Close it in Excel while labeling, or a save will fail until you close it.

## Run

```bash
python blank_junk_creator/main.py
```

Open the URL it prints (http://127.0.0.1:8767). Images play in order: folder 1 image 1, folder 1 image 2, then the next folder.

Marks stay on the current image until Enter. Enter writes the whole row, including empty fields, and opens the next image. Press the same key again to clear that mark. Going back loads the saved row so it can be changed or cleared.

| Key | Action |
| --- | --- |
| K | Keep |
| B | Blank |
| I | Junk / Invoice |
| C | Junk / Cover page |
| R | Junk / Record request |
| N | Junk / Instructions |
| F | Junk / Letter or fax |
| O | Junk / Others |
| Enter or Right | Save this page, empty fields included, and open the next image |
| Left | Previous image. Unsaved marks on this page are saved first. |
| Up | First image of the previous folder |

Back and Previous folder do the same thing with the mouse. Folder and Image, or an image number (1 is the first image of the first folder), jump to a specific page.

Refresh keeps your place. With no saved place, it opens on the first image that has no row yet.
