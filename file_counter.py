import csv
import os
import sys

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# ============================================================
# AZURE CONFIGURATION
# ============================================================

STORAGE_ACCOUNT = "azsadve2aipoc"

# CHANGE THIS to your actual container name
CONTAINER_NAME = "YOUR_CONTAINER_NAME"

# Make sure this EXACTLY matches the folder structure in Azure
PREFIX = "Run1/Batch1/DEID_PNGs/"


# ============================================================
# FILE EXTENSIONS TO COUNT
# ============================================================

EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
    ".jp2",
    ".j2k",
    ".pdf",
)


# ============================================================
# CSV OUTPUT
# ============================================================

CSV_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "jpg_counts_batch1.csv",
)


# ============================================================
# AZURE CONNECTION
# ============================================================

print("Connecting to Azure Blob Storage...", flush=True)

ACCOUNT_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"

try:
    credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )

    container_client = blob_service_client.get_container_client(
        CONTAINER_NAME
    )

    print("Azure connection created.", flush=True)

except Exception as e:
    print("\nERROR connecting to Azure:")
    print(e)
    input("\nPress Enter to close...")
    sys.exit(1)


# ============================================================
# SCAN BLOB STORAGE
# ============================================================

print()
print("=" * 70)
print("Scanning Azure Blob")
print("=" * 70)
print(f"Storage Account : {STORAGE_ACCOUNT}")
print(f"Container       : {CONTAINER_NAME}")
print(f"Prefix          : {PREFIX}")
print("=" * 70)


folder_counts = {}

total_blobs_found = 0
total_files_counted = 0


try:

    blobs = container_client.list_blobs(
        name_starts_with=PREFIX
    )

    for blob in blobs:

        total_blobs_found += 1

        blob_name = blob.name

        # Remove the DEID_PNGs prefix
        relative_path = blob_name[len(PREFIX):]

        # Example:
        #
        # FolderA/image1.png
        #
        # becomes:
        #
        # ["FolderA", "image1.png"]
        #
        # Or:
        #
        # FolderA/SubFolder/image1.png
        #
        # becomes:
        #
        # ["FolderA", "SubFolder", "image1.png"]

        parts = relative_path.split("/")

        # Ignore files directly inside DEID_PNGs
        if len(parts) < 2:
            continue

        # First folder immediately under DEID_PNGs
        folder_name = parts[0]

        # Actual filename
        filename = parts[-1]

        # Ignore anything that isn't one of the supported extensions
        if not filename.lower().endswith(EXTENSIONS):
            continue

        # Count it under the immediate parent folder
        folder_counts[folder_name] = (
            folder_counts.get(folder_name, 0) + 1
        )

        total_files_counted += 1


except Exception as e:

    print("\nERROR while reading blobs:")
    print(e)

    input("\nPress Enter to close...")
    sys.exit(1)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SCAN RESULTS")
print("=" * 70)

print(f"Total blobs found under prefix : {total_blobs_found}")
print(f"Total files counted            : {total_files_counted}")
print(f"Total folders found            : {len(folder_counts)}")
print()


if not folder_counts:

    print("NO FOLDERS / FILES WERE COUNTED.")
    print()
    print("This usually means one of these is wrong:")
    print()
    print("1. CONTAINER_NAME")
    print("2. PREFIX")
    print("3. Blob folder structure")
    print()
    print("To debug, uncomment the DEBUG section at the")
    print("bottom of this code.")
    print()

else:

    rows = sorted(
        folder_counts.items(),
        key=lambda x: x[0].lower()
    )

    print(f"{'Folder Name':<50} {'Count':>10}")
    print("-" * 50, "-" * 10)

    for folder_name, count in rows:
        print(f"{folder_name:<50} {count:>10}")

    print("-" * 50, "-" * 10)

    total = sum(count for _, count in rows)

    print(f"{'Total':<50} {total:>10}")


    # ========================================================
    # SAVE CSV
    # ========================================================

    try:

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["Folder Name", "Count"]
            )

            for folder_name, count in rows:

                writer.writerow(
                    [folder_name, count]
                )

            writer.writerow(
                ["Total", total]
            )

        print()
        print(f"CSV saved to:")
        print(CSV_PATH)

    except Exception as e:

        print()
        print("ERROR saving CSV:")
        print(e)


# ============================================================
# DEBUG
# ============================================================
#
# If the result is still 0, uncomment this section.
#
# It will print the actual blob names Azure is returning.
#
# ============================================================

# print()
# print("=" * 70)
# print("DEBUG - ACTUAL BLOB PATHS")
# print("=" * 70)
#
# try:
#     debug_blobs = container_client.list_blobs(
#         name_starts_with=PREFIX
#     )
#
#     debug_count = 0
#
#     for blob in debug_blobs:
#         print(blob.name)
#         debug_count += 1
#
#         if debug_count >= 50:
#             print("... showing first 50 blobs only ...")
#             break
#
# except Exception as e:
#     print("DEBUG ERROR:")
#     print(e)


print()
input("Press Enter to close...")