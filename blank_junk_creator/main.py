"""Label local or blob images as Keep, Blank, or a junk subtype. The CSV is written on this machine.

Edit RUN CONFIG, then run:  python blank_junk_creator/main.py
Open the printed URL. Enter saves the page and opens the next image.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blank_junk_creator.catalog import (
    connect_container,
    list_images,
    list_local_images,
    read_blob_image,
    read_local_image,
)
from blank_junk_creator.labels import LabelStore
from blank_junk_creator.server import App, serve

# ============================== RUN CONFIG ==============================
INPUT_SOURCE = "local"  # "local" or "blob"

# local: a folder of chart folders, so this file exists:
#   LOCAL_INPUT / <folder name> / <image name>
LOCAL_INPUT = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\images")

# blob: PREFIX/<chart folder>/<image>
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"
# Empty starts at the first folder. A folder name starts there and continues downward.
START_FROM = ""

# Labels stay on this machine. Close the file in Excel while labeling.
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\blank_junk_labels.csv")

HOST = "127.0.0.1"
PORT = 8767
# ========================================================================

LOGGER = logging.getLogger("blank_junk_creator")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = INPUT_SOURCE.strip().lower()
    if source == "blob" and (not CONTAINER_NAME or CONTAINER_NAME == "YOUR_CONTAINER_NAME"):
        raise SystemExit("Set CONTAINER_NAME in blank_junk_creator/main.py")
    if source not in {"local", "blob"}:
        raise SystemExit('INPUT_SOURCE must be "local" or "blob"')

    store = LabelStore(CSV_PATH)
    app = App(store)

    def load() -> None:
        try:
            if source == "local":
                LOGGER.info("Listing local images under %s", LOCAL_INPUT)
                app.read_image = read_local_image
                items = list_local_images(LOCAL_INPUT, START_FROM)
            else:
                LOGGER.info("Connecting to %s / %s", STORAGE_ACCOUNT, CONTAINER_NAME)
                container = connect_container(STORAGE_ACCOUNT, CONTAINER_NAME)
                app.read_image = lambda item: read_blob_image(container, item)
                LOGGER.info("Listing images under %s", PREFIX)
                items = list_images(container, PREFIX, START_FROM)
            if not items:
                raise FileNotFoundError("No images found")
            app.set_items(items)
        except Exception as exc:
            LOGGER.exception("Could not read images")
            app.fail(f"Could not read images: {type(exc).__name__}: {exc}")

    threading.Thread(target=load, name="blob-list", daemon=True).start()
    LOGGER.info("CSV will be written to %s", CSV_PATH)
    serve(app, HOST, PORT)


if __name__ == "__main__":
    main()
