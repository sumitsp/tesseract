# training_data_creator

Label chart images as Handwritten or Printed. The page runs on this machine. Images are read from a local folder or from Azure Blob. The CSV is written locally.

Each row is one image: `value` (Handwritten, Printed, Form, Visual, Blank, or Uncertain), `visibility` (Visible or Not visible), and `handwritten percent` (the drawn box as a percentage of the page, or 0 when no box is drawn). Labeling it again updates that row.

## Setup

```bash
pip install -r training_data_creator/requirements.txt
az login
```

Edit the RUN CONFIG block at the top of `main.py`:

- `INPUT_SOURCE`: `"local"` or `"blob"`.
- Local: `LOCAL_INPUT` is the folder that contains the chart folders, so `LOCAL_INPUT\<folder name>\<image name>` is the file.
- Blob: `CONTAINER_NAME` and `PREFIX` (`PREFIX/<chart folder>/<image>`). The account is `azsadve2aipoc`. Sign in with `az login` first.
- `START_FROM`: leave empty to begin at the first folder. Set a folder name to begin at that folder and continue through the later folders.
- `CSV_PATH`: where `training_labels.csv` is written. Close it in Excel while labeling, or a save will fail until you close it.

## Run

From inside this folder:

```bash
python main.py
```

From the project folder it is the same file:

```bash
python training_data_creator/main.py
```

Open the URL it prints (http://127.0.0.1:8765). Images play in order: folder 1 image 1, folder 1 image 2, then the next folder.

Marks stay on the current image until Enter. Enter writes the whole row, including empty fields, and opens the next image. Press the same key again to clear that mark back to empty. Going back loads the saved row so any mark can be changed or cleared.

| Key | Action |
| --- | --- |
| A / Space / W / N / D / O | Set the page type. Press it again to clear. |
| F / J | Visible or Not visible. Press it again to clear. |
| Drag | Box around the writing. Its area is `handwritten percent`. No box leaves that field empty. |
| Enter or Right | Save this page, empty fields included, and open the next image |
| Left | Previous image. Unsaved marks on this page are saved first. |
| Up | First image of the previous folder |

Back and Previous folder do the same thing with the mouse. Folder and Image, or an image number (1 is the first image of the first folder), jump to a specific page. The current label is shown, and pressing A or Space there updates that CSV row.

Refresh keeps your place. With no saved place, it opens on the first image that has no row yet.
