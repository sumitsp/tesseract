# training_data_creator

Label chart images from Azure Blob as Handwritten or Printed. The page runs on this machine. Images are read from blob. The CSV is written locally.

Each row is `folder name`, `image name`, `value` (`Handwritten` or `Printed`). One row per image. Labeling it again replaces that row.

## Setup

```bash
pip install -r training_data_creator/requirements.txt
az login
```

Edit the RUN CONFIG block at the top of `main.py`:

- `CONTAINER_NAME` and `PREFIX` (`PREFIX/<chart folder>/<image>`). The account is `azsadve2aipoc`.
- `CSV_PATH`: where `training_labels.csv` is written. Close it in Excel while labeling, or a save will fail until you close it.

## Run

```bash
python training_data_creator/main.py
```

Open the URL it prints (http://127.0.0.1:8765). Images play in order: folder 1 image 1, folder 1 image 2, then the next folder.

| Key | Action |
| --- | --- |
| A | Handwritten, then the next image |
| Space | Printed, then the next image |
| Left / Right | Previous / next image, no change to the CSV |
| Up | First image of the previous folder |

Back and Previous folder do the same thing with the mouse. Folder and Image, or an image number (1 is the first image of the first folder), jump to a specific page. The current label is shown, and pressing A or Space there updates that CSV row.

Refresh keeps your place. With no saved place, it opens on the first image that has no row yet.
