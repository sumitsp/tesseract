"""Label blob images as Handwritten or Printed. The CSV is written on this machine.

Edit RUN CONFIG, then run:  python main.py
Open the printed URL. Keys: A handwritten, Space printed, Left previous image.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_data_creator.catalog import connect_container, list_images, read_blob_image
from training_data_creator.labels import LabelStore
from training_data_creator.server import App, serve

# ============================== RUN CONFIG ==============================
STORAGE_ACCOUNT = "azsadve2aipoc"
CONTAINER_NAME = "YOUR_CONTAINER_NAME"
PREFIX = "Run1/Batch1/DEID_PNGs/"  # PREFIX/<chart folder>/<image>
# Empty starts at the first folder. A folder name starts there and continues downward.
START_FROM = ""

# Labels stay on this machine. Close the file in Excel while labeling.
CSV_PATH = Path(r"C:\Users\sumit.pandey\Desktop\Imaging\training_labels.csv")

HOST = "127.0.0.1"
PORT = 8765
# ========================================================================

LOGGER = logging.getLogger("training_data_creator")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not CONTAINER_NAME or CONTAINER_NAME == "YOUR_CONTAINER_NAME":
        raise SystemExit("Set CONTAINER_NAME in training_data_creator/main.py")

    store = LabelStore(CSV_PATH)
    app = App(store)

    def load() -> None:
        try:
            LOGGER.info("Connecting to %s / %s", STORAGE_ACCOUNT, CONTAINER_NAME)
            container = connect_container(STORAGE_ACCOUNT, CONTAINER_NAME)
            app.read_image = lambda item: read_blob_image(container, item)
            LOGGER.info("Listing images under %s", PREFIX)
            app.set_items(list_images(container, PREFIX, START_FROM))
        except Exception as exc:
            LOGGER.exception("Could not read blob")
            app.fail(f"Could not read blob: {type(exc).__name__}: {exc}")

    threading.Thread(target=load, name="blob-list", daemon=True).start()
    LOGGER.info("CSV will be written to %s", CSV_PATH)
    serve(app, HOST, PORT)


if __name__ == "__main__":
    main()
